//! A fragment's partitions run a few at a time. Run all at once, as
//! `CoalescePartitionsExec` does, a worker's chunks contend for its one or two
//! cores and share its scan budget; one at a time, nothing overlaps a chunk's
//! read with the previous one's compute.

use std::fmt;
use std::sync::Arc;

use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::stream::RecordBatchReceiverStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;
use tokio::sync::Semaphore;

/// `input`'s partitions as one, running at most `in_flight` of them at once,
/// each on its own task, started in order.
#[derive(Debug)]
pub(crate) struct BoundedCoalesceExec {
    input: Arc<dyn ExecutionPlan>,
    in_flight: usize,
    properties: Arc<PlanProperties>,
}

impl BoundedCoalesceExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, in_flight: usize) -> Self {
        let mut equivalence = input.equivalence_properties().clone();
        equivalence.clear_orderings();
        equivalence.clear_per_partition_constants();
        let properties = PlanProperties::new(
            equivalence,
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Self {
            input,
            in_flight: in_flight.max(1),
            properties: Arc::new(properties),
        }
    }
}

impl DisplayAs for BoundedCoalesceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BoundedCoalesceExec: in_flight={}", self.in_flight)
    }
}

impl ExecutionPlan for BoundedCoalesceExec {
    fn name(&self) -> &str {
        "BoundedCoalesceExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(children.swap_remove(0), self.in_flight)))
    }

    fn execute(
        &self,
        _partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let mut builder = RecordBatchReceiverStream::builder(self.schema(), self.in_flight);
        let permits = Arc::new(Semaphore::new(self.in_flight));
        let partitions = self.input.output_partitioning().partition_count();
        let (input, tx) = (self.input.clone(), builder.tx());
        builder.spawn(async move {
            let mut running = tokio::task::JoinSet::new();
            for partition in 0..partitions {
                let permit = permits.clone().acquire_owned().await.expect("never closed");
                let (input, context, tx) = (input.clone(), context.clone(), tx.clone());
                running.spawn(async move {
                    let _permit = permit;
                    let mut stream = match input.execute(partition, context) {
                        Ok(stream) => stream,
                        Err(e) => {
                            tx.send(Err(e)).await.ok();
                            return;
                        }
                    };
                    while let Some(item) = stream.next().await {
                        let failed = item.is_err();
                        if tx.send(item).await.is_err() || failed {
                            return;
                        }
                    }
                });
            }
            while running.join_next().await.is_some() {}
            Ok(())
        });
        Ok(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::physical_expr::EquivalenceProperties;
    use datafusion::physical_plan::execution_plan::EmissionType;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_plan::{collect, execution_plan::Boundedness};

    use super::*;

    /// Partitions that each yield their index after a pause, counting how many
    /// are running at once.
    #[derive(Debug)]
    struct Counting {
        running: Arc<AtomicUsize>,
        most: Arc<AtomicUsize>,
        properties: Arc<PlanProperties>,
    }

    impl Counting {
        fn new(partitions: usize) -> Self {
            let schema: SchemaRef =
                Arc::new(Schema::new(vec![Field::new("p", DataType::Int64, false)]));
            Self {
                running: Arc::default(),
                most: Arc::default(),
                properties: Arc::new(PlanProperties::new(
                    EquivalenceProperties::new(schema),
                    Partitioning::UnknownPartitioning(partitions),
                    EmissionType::Incremental,
                    Boundedness::Bounded,
                )),
            }
        }
    }

    impl DisplayAs for Counting {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Counting")
        }
    }

    impl ExecutionPlan for Counting {
        fn name(&self) -> &str {
            "Counting"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            let (running, most, schema) = (self.running.clone(), self.most.clone(), self.schema());
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vec![partition as i64]))],
            )?;
            let output = futures::stream::once(async move {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                running.fetch_sub(1, Ordering::SeqCst);
                Ok(batch)
            });
            Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn runs_every_partition_but_only_so_many_at_once() {
        let input = Arc::new(Counting::new(7));
        let most = input.most.clone();
        let plan = Arc::new(BoundedCoalesceExec::new(input, 2));

        let batches = collect(plan, Arc::new(TaskContext::default()))
            .await
            .unwrap();

        let mut seen: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        seen.sort();
        assert_eq!(seen, (0..7).collect::<Vec<_>>());
        assert_eq!(most.load(Ordering::SeqCst), 2);
    }
}
