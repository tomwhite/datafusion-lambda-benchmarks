//! A fragment's partial rows merged across its partitions. Each partition, one
//! per chunk, emits its own partial row per group, so without this a fragment's
//! response grows with its chunks.

use std::sync::Arc;

use datafusion::error::Result;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::bounded::BoundedCoalesceExec;

/// `plan`'s partitions as one, run `in_flight` at a time. A partial aggregate's
/// are merged under a `PartialReduce`, which keeps its schema: one partial row
/// per group.
pub fn merge_partials(
    plan: Arc<dyn ExecutionPlan>,
    in_flight: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    if plan.output_partitioning().partition_count() < 2 {
        return Ok(plan);
    }
    let coalesced = Arc::new(BoundedCoalesceExec::new(plan.clone(), in_flight));
    let Some(partial) = plan.downcast_ref::<AggregateExec>() else {
        return Ok(coalesced);
    };
    if *partial.mode() != AggregateMode::Partial {
        return Ok(coalesced);
    }
    // as DataFusion's planner builds the stage above a partial aggregate
    let merged = AggregateExec::try_new(
        AggregateMode::PartialReduce,
        partial.group_expr().as_final(),
        partial.aggr_expr().to_vec(),
        partial.filter_expr().to_vec(),
        coalesced.clone(),
        partial.input_schema(),
    )?;
    Ok(if merged.schema() == plan.schema() {
        Arc::new(merged)
    } else {
        coalesced
    })
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int64Array};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::{collect, displayable};
    use datafusion::prelude::{SessionConfig, SessionContext};

    use super::*;

    /// Three partitions, each holding every one of the keys 0 to 3.
    async fn context() -> SessionContext {
        let partitions = (0..3)
            .map(|p| {
                let k = Int64Array::from_iter_values((0..12).map(|i| i % 4));
                let v = Float64Array::from_iter_values((0..12).map(|i| (p * 12 + i) as f64));
                vec![
                    RecordBatch::try_from_iter([("k", Arc::new(k) as _), ("v", Arc::new(v) as _)])
                        .unwrap(),
                ]
            })
            .collect::<Vec<_>>();
        let table = MemTable::try_new(partitions[0][0].schema(), partitions).unwrap();
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(3));
        ctx.register_table("t", Arc::new(table)).unwrap();
        ctx
    }

    fn partial_aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|a| *a.mode() == AggregateMode::Partial)
        {
            return Some(plan.clone());
        }
        plan.children().into_iter().find_map(partial_aggregate)
    }

    /// `query`'s answer, and that of its partial aggregate merged and then
    /// finished as DataFusion's planner would, with how many rows the merge
    /// returned.
    async fn both_ways(query: &str) -> (String, String, usize) {
        let ctx = context().await;
        let df = ctx.sql(query).await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let expected = collect(plan.clone(), ctx.task_ctx()).await.unwrap();

        let partial = partial_aggregate(&plan).expect("a partial aggregate");
        assert_eq!(partial.output_partitioning().partition_count(), 3);
        let merged = merge_partials(partial.clone(), 2).unwrap();
        assert!(
            merged
                .downcast_ref::<AggregateExec>()
                .is_some_and(|a| *a.mode() == AggregateMode::PartialReduce),
            "not merged:\n{}",
            displayable(merged.as_ref()).indent(true)
        );
        let rows = collect(merged.clone(), ctx.task_ctx())
            .await
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        let source = partial.downcast_ref::<AggregateExec>().unwrap();
        let finished = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Final,
                source.group_expr().as_final(),
                source.aggr_expr().to_vec(),
                source.filter_expr().to_vec(),
                merged,
                source.input_schema(),
            )
            .unwrap(),
        );
        let actual = collect(finished, ctx.task_ctx()).await.unwrap();
        // the rows alone: the query's projection renames the columns
        let sorted = |batches: &[RecordBatch]| {
            let mut lines = arrow::util::pretty::pretty_format_batches(batches)
                .unwrap()
                .to_string()
                .lines()
                .skip(3)
                .filter(|line| !line.starts_with('+'))
                .map(|line| line.split('|').map(str::trim).collect::<Vec<_>>().join("|"))
                .collect::<Vec<_>>();
            lines.sort();
            lines.join("\n")
        };
        (sorted(&actual), sorted(&expected), rows)
    }

    #[tokio::test]
    async fn one_partial_row_per_group_with_the_same_answer() {
        let (actual, expected, rows) = both_ways(
            "SELECT k, sum(v) AS s, avg(v) AS a, count(*) AS n, min(v) AS lo, \
             count(*) FILTER (WHERE v > 20) AS big FROM t GROUP BY k",
        )
        .await;

        assert_eq!(rows, 4);
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn one_partial_row_without_groups() {
        let (actual, expected, rows) =
            both_ways("SELECT sum(v) AS s, stddev(v) AS sd, max(v) AS hi FROM t").await;

        assert_eq!(rows, 1);
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn one_partition_is_left_as_it_is() {
        let ctx = context().await;
        let plan = ctx
            .sql("SELECT k, sum(v) FROM t GROUP BY k")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let partial = partial_aggregate(&plan).unwrap();
        let one = partial
            .clone()
            .with_new_children(vec![Arc::new(CoalescePartitionsExec::new(
                partial.children()[0].clone(),
            ))])
            .unwrap();

        assert!(Arc::ptr_eq(&merge_partials(one.clone(), 2).unwrap(), &one));
    }
}
