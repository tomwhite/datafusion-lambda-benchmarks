//! Fanning a query out on map-lambda. [`FanOutRule`] cuts a physical plan at
//! each partial aggregate over a Zarr scan, and [`MapLambdaZarrExec`] runs what it
//! cut off on `fragment` lambdas, a run of chunks each, streaming the partial
//! states back to the rest of the plan as workers finish. A plan with nothing
//! to cut runs in this process.

pub mod region;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use map_lambda_client::{Lambda, Lambdas, Transport};
use serde::Serialize;
use serde_json::{json, Map, Value};
use split::PartitionSpec;
use zarr_datafusion::optimizer::CardinalityRule;
use zarr_datafusion::physical_plan::partition::split_selection;
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;

/// Where fragments run, and how many at once.
#[derive(Debug, Clone)]
pub struct Settings {
    pub dispatcher_function: String,
    pub worker_function: String,
    /// Worker invocations in flight.
    pub concurrency: usize,
    /// How many dispatchers share them, or the fewest that hold them. Each
    /// streams its workers' results back at 2 MB/s after its first 6 MB.
    pub dispatchers: Option<usize>,
    pub retries: usize,
    /// A local `cargo lambda watch`, for dispatchers and workers both.
    pub endpoint_url: Option<String>,
    pub chunks_per_invocation: usize,
    /// How many of an invocation's chunks a worker runs at once, else its default.
    pub chunks_in_flight: Option<usize>,
    /// Fan out over S3 buckets in other regions than the workers, which is
    /// billed per GB read.
    pub allow_cross_region: bool,
    /// Keep every invocation's record and what its worker measured in the [`Report`].
    pub return_stats: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            dispatcher_function: "dispatcher".into(),
            worker_function: "fragment".into(),
            concurrency: 1000,
            dispatchers: None,
            retries: 2,
            endpoint_url: None,
            chunks_per_invocation: 1,
            chunks_in_flight: None,
            allow_cross_region: false,
            return_stats: false,
        }
    }
}

/// A map-lambda session and the settings its fan-outs run with.
pub struct FanOut {
    lambdas: Lambdas,
    settings: Settings,
}

impl fmt::Debug for FanOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FanOut")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl FanOut {
    pub fn new(transport: Arc<dyn Transport>, settings: Settings) -> Self {
        let mut calls = map_lambda_client::Settings::default();
        calls.dispatcher_function = settings.dispatcher_function.clone();
        calls.dispatchers = settings.dispatchers;
        calls.concurrency = settings.concurrency;
        calls.retries = settings.retries;
        calls.return_stats = settings.return_stats;
        let lambdas = Lambdas::new(transport, settings.endpoint_url.clone()).with_settings(calls);
        Self { lambdas, settings }
    }

    /// Against real Lambda, or the emulator at `settings.endpoint_url`.
    pub async fn connect(settings: Settings) -> Self {
        let lambda = Lambda::connect(settings.endpoint_url.as_deref()).await;
        Self::new(Arc::new(lambda), settings)
    }
}

/// A session over Zarr tables that fans out through `fan_out`, planning what
/// runs here for `target_partitions`, or by default for this machine's cores.
pub fn session(fan_out: Arc<FanOut>, target_partitions: Option<usize>) -> SessionContext {
    split::session_from(
        split::builder()
            .with_physical_optimizer_rule(Arc::new(FanOutRule(fan_out)))
            // After the cut, so it only pushes down aggregates that stay here.
            .with_physical_optimizer_rule(Arc::new(CardinalityRule::new()))
            .with_physical_optimizer_rule(Arc::new(split::PerValueRule)),
        target_partitions,
    )
}

/// A Zarr table's location, and its zarr-datafusion options, such as the
/// `group` to read in an icechunk repo.
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub location: String,
    pub options: BTreeMap<String, String>,
}

impl Table {
    fn create(&self, name: &str) -> String {
        let quoted = |s: &str| format!("'{}'", s.replace('\'', "''"));
        let mut sql = format!(
            "CREATE EXTERNAL TABLE \"{}\" STORED AS ZARR LOCATION {}",
            name.replace('"', "\"\""),
            quoted(&self.location)
        );
        if !self.options.is_empty() {
            let options = self
                .options
                .iter()
                .map(|(key, value)| format!("{} {}", quoted(key), quoted(value)))
                .collect::<Vec<_>>();
            sql += &format!(" OPTIONS ({})", options.join(", "));
        }
        sql
    }
}

/// Register each Zarr table by name.
pub async fn register(ctx: &SessionContext, tables: &HashMap<String, Table>) -> Result<()> {
    for (name, table) in tables {
        ctx.sql(&table.create(name)).await?;
    }
    Ok(())
}

/// Run `query`: its result, and a report from each fan-out in its plan.
pub async fn run(
    ctx: &SessionContext,
    query: &str,
) -> Result<(SchemaRef, Vec<RecordBatch>, Vec<Report>)> {
    let plan = ctx.sql(query).await?.create_physical_plan().await?;
    let batches = collect(plan.clone(), ctx.task_ctx()).await?;
    let mut reports = vec![];
    for_each_fan_out(&plan, |exec| reports.push(exec.report()))?;
    Ok((plan.schema(), batches, reports))
}

/// Call `f` on each fan-out in `plan`.
pub fn for_each_fan_out(
    plan: &Arc<dyn ExecutionPlan>,
    mut f: impl FnMut(&MapLambdaZarrExec),
) -> Result<()> {
    plan.apply(|node| {
        if let Some(exec) = node.downcast_ref::<MapLambdaZarrExec>() {
            f(exec);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

/// Replaces each partial aggregate over a chunkwise Zarr scan with a
/// [`MapLambdaZarrExec`] that runs it on workers.
#[derive(Debug)]
pub struct FanOutRule(pub Arc<FanOut>);

impl PhysicalOptimizerRule for FanOutRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            Ok(match self.cut(&node)? {
                Some(exec) => Transformed::yes(exec),
                None => Transformed::no(node),
            })
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "fan_out"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

impl FanOutRule {
    fn cut(&self, node: &Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let partial = node
            .downcast_ref::<AggregateExec>()
            .is_some_and(|a| *a.mode() == AggregateMode::Partial);
        if !partial {
            return Ok(None);
        }
        let Some(chunks) = chunkwise_scan(node).and_then(chunks) else {
            return Ok(None);
        };
        Ok(Some(Arc::new(MapLambdaZarrExec::new(
            without_round_robin(node.clone())?,
            chunks,
            self.0.clone(),
        )?)))
    }
}

/// `fragment` without the round-robin repartitions planned for this process's
/// cores, so each worker runs one partition per chunk it is given.
// TODO: plan for the worker's own vCPUs, by running `EnforceDistribution` over
// the bound fragment there.
fn without_round_robin(fragment: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    fragment
        .transform_up(|node| {
            Ok(match node.downcast_ref::<RepartitionExec>() {
                Some(r) if matches!(r.partitioning(), Partitioning::RoundRobinBatch(_)) => {
                    Transformed::yes(r.input().clone())
                }
                _ => Transformed::no(node),
            })
        })
        .map(|t| t.data)
}

/// The Zarr scan under a partial aggregate, if every node between them works
/// row by row, so that a worker given some chunks computes exactly their share
/// of the partial states.
// TODO: a broadcast join, where the other side is small enough for every worker to read whole.
fn chunkwise_scan(partial: &Arc<dyn ExecutionPlan>) -> Option<&ZarrExec> {
    let mut node = *partial.children().first()?;
    loop {
        if let Some(scan) = node.downcast_ref::<ZarrExec>() {
            return Some(scan);
        }
        let rowwise = node.is::<ProjectionExec>()
            || node.is::<FilterExec>()
            || node.is::<RepartitionExec>()
            || node.is::<CoalescePartitionsExec>()
            || node.is::<CooperativeExec>();
        let [child] = node.children()[..] else {
            return None;
        };
        if !rowwise {
            return None;
        }
        node = child;
    }
}

/// The scan's partitions cut into single chunks along the outer axis, less any
/// that read nothing.
fn chunks(scan: &ZarrExec) -> Option<Vec<PartitionSpec>> {
    // No partitions means one unsliced read: VirtualiZarr, Icechunk, or a
    // limit, whose pushdown is only right over the whole table.
    if scan.limit().is_some() || scan.partitions().is_empty() {
        return None;
    }
    // Unknown chunking (0) leaves each partition whole.
    let chunk_len = scan
        .store_meta()?
        .data_vars
        .first()?
        .chunks
        .as_ref()
        .and_then(|c| c.first().copied())
        .unwrap_or(0);
    Some(
        scan.partitions()
            .iter()
            .flat_map(|spec| {
                split_selection(&spec.outer, chunk_len, usize::MAX)
                    .into_iter()
                    .map(|outer| PartitionSpec {
                        outer,
                        extra: spec.extra.clone(),
                    })
            })
            .filter(|spec| !spec.is_empty())
            .collect(),
    )
}

/// What a fan-out did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub chunks: usize,
    pub invocations: usize,
    /// From dispatching to the last answer.
    pub elapsed_s: f64,
    /// With `return_stats`: each invocation's record, as map-lambda keeps it.
    pub invocation_stats: Vec<Value>,
    /// With `return_stats`: each worker's `index`, `timings_ms`,
    /// `s3_requests` and `chunks_in_flight`.
    pub workers: Vec<Value>,
    /// Each dispatcher's record.
    pub dispatcher_stats: Vec<Value>,
    /// With `return_stats`: each frame from the dispatchers as it arrived
    /// here, `at_s` from dispatching, with its workers' result `bytes`.
    pub arrivals: Vec<Value>,
}

/// A fragment run on map-lambda workers, one invocation per run of chunks,
/// yielding every worker's batches in one partition as they arrive.
#[derive(Debug)]
pub struct MapLambdaZarrExec {
    schema: SchemaRef,
    plan: Arc<dyn ExecutionPlan>,
    fragment: String,
    chunks: Vec<PartitionSpec>,
    fan_out: Arc<FanOut>,
    properties: Arc<PlanProperties>,
    report: Arc<Mutex<Report>>,
}

impl MapLambdaZarrExec {
    pub fn new(
        plan: Arc<dyn ExecutionPlan>,
        chunks: Vec<PartitionSpec>,
        fan_out: Arc<FanOut>,
    ) -> Result<Self> {
        let schema = plan.schema();
        let fragment = split::encode_plan(plan.clone())?;
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            schema,
            plan,
            fragment,
            chunks,
            fan_out,
            properties,
            report: Default::default(),
        })
    }

    /// The fragment each worker runs, as base64 datafusion-proto bytes.
    pub fn fragment(&self) -> &str {
        &self.fragment
    }

    /// The fragment each worker runs, before it is bound to its chunks.
    pub fn fragment_plan(&self) -> &Arc<dyn ExecutionPlan> {
        &self.plan
    }

    /// Each invocation's chunks.
    pub fn inputs(&self) -> Vec<Value> {
        self.chunks
            .chunks(self.fan_out.settings.chunks_per_invocation.max(1))
            .map(|run| {
                serde_json::to_value(split::compact::compact(run))
                    .expect("partition specs serialize")
            })
            .collect()
    }

    pub fn report(&self) -> Report {
        self.report.lock().expect("unpoisoned").clone()
    }

    /// What every invocation is sent besides its chunks.
    fn template(&self) -> Map<String, Value> {
        let mut template =
            Map::from_iter([("plan".to_string(), Value::from(self.fragment.clone()))]);
        if let Some(n) = self.fan_out.settings.chunks_in_flight {
            template.insert("chunks_in_flight".into(), n.into());
        }
        template
    }
}

impl DisplayAs for MapLambdaZarrExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let function = &self.fan_out.settings.worker_function;
        let invocations = self
            .chunks
            .len()
            .div_ceil(self.fan_out.settings.chunks_per_invocation.max(1));
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "MapLambdaZarrExec: function={function}, chunks={}, invocations={invocations}",
                self.chunks.len()
            ),
            DisplayFormatType::TreeRender => write!(
                f,
                "function={function}\nchunks={}\ninvocations={invocations}",
                self.chunks.len()
            ),
        }
    }
}

fn external(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

fn to_value(record: &impl Serialize) -> Value {
    serde_json::to_value(record).expect("records serialize")
}

fn decode(output: &Value) -> Result<Vec<RecordBatch>> {
    let batches = output["batches"].as_str().ok_or_else(|| {
        DataFusionError::Execution(format!("a fragment answered without batches: {output}"))
    })?;
    split::decode_batches(batches)
}

impl ExecutionPlan for MapLambdaZarrExec {
    fn name(&self) -> &str {
        "MapLambdaZarrExec"
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
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let inputs = self.inputs();
        *self.report.lock().expect("unpoisoned") = Report {
            chunks: self.chunks.len(),
            invocations: inputs.len(),
            ..Report::default()
        };
        if inputs.is_empty() {
            let none = futures::stream::empty();
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                none,
            )));
        }

        let settings = &self.fan_out.settings;
        let locations = if settings.allow_cross_region || settings.endpoint_url.is_some() {
            vec![]
        } else {
            split::zarr_scans(&self.plan)?
                .iter()
                .filter_map(|scan| scan.downcast_ref::<ZarrExec>())
                .map(|scan| scan.path().to_string())
                .collect()
        };
        let (fan_out, template) = (self.fan_out.clone(), self.template());
        let run = futures::stream::once(async move {
            region::check(&locations).await?;
            let worker = &fan_out.settings.worker_function;
            Ok::<_, DataFusionError>(
                fan_out
                    .lambdas
                    .map_unordered(worker, inputs, "partitions", template)
                    .map_err(external)?
                    .map_err(external),
            )
        })
        .try_flatten();
        let report = self.report.clone();
        let return_stats = self.fan_out.settings.return_stats;
        let started = Instant::now();
        let batches = run
            .and_then(move |batch| {
                let decoded = batch
                    .answers
                    .iter()
                    .map(|(index, answer)| {
                        let answer: Value = serde_json::from_str(answer.get()).map_err(external)?;
                        Ok((*index, answer))
                    })
                    .collect::<Result<Vec<_>>>()
                    .and_then(|answers| {
                        let mut report = report.lock().expect("unpoisoned");
                        report.elapsed_s = started.elapsed().as_secs_f64();
                        report
                            .dispatcher_stats
                            .extend(batch.dispatcher_stats.iter().map(to_value));
                        if return_stats {
                            let bytes: usize = answers
                                .iter()
                                .map(|(_, answer)| answer["batches"].as_str().map_or(0, str::len))
                                .sum();
                            let at_s = report.elapsed_s;
                            report.arrivals.push(json!({"at_s": at_s, "bytes": bytes}));
                            report
                                .invocation_stats
                                .extend(batch.invocation_stats.iter().map(to_value));
                            report.workers.extend(answers.iter().map(|(index, answer)| {
                                json!({
                                    "index": index,
                                    "timings_ms": answer["timings_ms"],
                                    "s3_requests": answer["s3_requests"],
                                    "chunks_in_flight": answer["chunks_in_flight"],
                                })
                            }));
                        }
                        answers
                            .iter()
                            .map(|(_, answer)| decode(answer))
                            .collect::<Result<Vec<_>>>()
                    })
                    .map(|batches| futures::stream::iter(batches.into_iter().flatten().map(Ok)));
                futures::future::ready(decoded)
            })
            .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            batches,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_is_created_with_its_options_quoted() {
        let table = Table {
            location: "s3://b/it's".to_string(),
            options: BTreeMap::from([
                ("group".to_string(), "single/spatial".to_string()),
                ("anonymous".to_string(), "true".to_string()),
            ]),
        };
        assert_eq!(
            table.create("era5"),
            "CREATE EXTERNAL TABLE \"era5\" STORED AS ZARR LOCATION 's3://b/it''s' \
             OPTIONS ('anonymous' 'true', 'group' 'single/spatial')"
        );
        assert_eq!(
            Table {
                location: "/data/era5".to_string(),
                ..Table::default()
            }
            .create("e"),
            "CREATE EXTERNAL TABLE \"e\" STORED AS ZARR LOCATION '/data/era5'"
        );
    }
}
