//! Fanning a query out on map-lambda. [`FanOutRule`] cuts a physical plan at
//! each partial aggregate over a Zarr scan, and [`MapLambdaZarrExec`] runs what it
//! cut off on `fragment` lambdas, a run of chunks each, streaming the partial
//! states back to the rest of the plan as workers finish. When no group's rows
//! span two invocations, the final aggregate runs on the workers too. A plan
//! with nothing to cut runs in this process.

pub mod region;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use datafusion::arrow::array::{AsArray, UInt64Array};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::datatypes::UInt64Type;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::datasource::sink::DataSinkExec;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::prelude::SessionContext;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use map_lambda_client::{connect, Invoker, Session};
use serde::Serialize;
use serde_json::{json, Map, Value};
use split::PartitionSpec;
use zarr_datafusion::optimizer::CardinalityRule;
use zarr_datafusion::physical_plan::partition::split_selection;
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;
use zarr_datafusion::reader::filter::CoordSelection;
use zarr_datafusion::reader::schema_inference::{ZarrArrayMeta, ZarrStoreMeta};
use zarr_datafusion::writer::{create_skeleton, SkeletonSpec, ZarrDataSink};

/// Where fragments run, and how many at once.
#[derive(Debug, Clone)]
pub struct Settings {
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

/// Called with how many of a fan-out's invocations have succeeded, and how
/// many it has: with none before it invokes, then as they arrive.
pub type OnProgress = Arc<dyn Fn(usize, usize) + Send + Sync>;

/// A map-lambda session and the settings its fan-outs run with.
pub struct FanOut {
    session: Session,
    settings: Settings,
    on_progress: Option<OnProgress>,
}

impl fmt::Debug for FanOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FanOut")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl FanOut {
    pub fn new(invoker: Arc<dyn Invoker>, settings: Settings) -> Self {
        let mut calls = map_lambda_client::Settings::default();
        calls.dispatchers = settings.dispatchers;
        calls.concurrency = settings.concurrency;
        calls.retries = settings.retries;
        calls.return_stats = settings.return_stats;
        let session = Session::new(invoker, settings.endpoint_url.clone()).with_settings(calls);
        Self {
            session,
            settings,
            on_progress: None,
        }
    }

    pub fn with_progress(mut self, on_progress: OnProgress) -> Self {
        self.on_progress = Some(on_progress);
        self
    }

    /// Against real Lambda, or the emulator at `settings.endpoint_url`.
    pub async fn connect(settings: Settings) -> Self {
        let invoker = connect(settings.endpoint_url.as_deref()).await;
        Self::new(invoker, settings)
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
            let exec = match self.cut(&node)? {
                Some(exec) => Some(exec),
                None => match finish(&node)? {
                    Some(exec) => Some(exec),
                    None => write(&node)?,
                },
            };
            Ok(match exec {
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
        let Some(partial) = node
            .downcast_ref::<AggregateExec>()
            .filter(|a| *a.mode() == AggregateMode::Partial)
        else {
            return Ok(None);
        };
        let Some(scan) = chunkwise_scan(node) else {
            return Ok(None);
        };
        let Some(chunks) = chunks(scan) else {
            return Ok(None);
        };
        let per_invocation = self.0.settings.chunks_per_invocation.max(1);
        let (runs, whole_groups) = if groups_within_tiles(partial, scan, &chunks) {
            by_tile(chunks, per_invocation)
        } else {
            let runs = chunks.chunks(per_invocation).map(<[_]>::to_vec).collect();
            (runs, false)
        };
        Ok(Some(Arc::new(MapLambdaZarrExec::new(
            without_round_robin(node.clone())?,
            runs,
            whole_groups,
            self.0.clone(),
        )?)))
    }
}

/// A final aggregate over a fan-out whose invocations each hold all of their
/// groups' rows, moved onto the workers.
fn finish(node: &Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(last) = node.downcast_ref::<AggregateExec>().filter(|a| {
        matches!(
            a.mode(),
            AggregateMode::Final | AggregateMode::FinalPartitioned
        )
    }) else {
        return Ok(None);
    };
    let mut input = last.input();
    while input.is::<RepartitionExec>() || input.is::<CoalescePartitionsExec>() {
        input = input.children()[0];
    }
    let Some(fan_out) = input
        .downcast_ref::<MapLambdaZarrExec>()
        .filter(|f| f.whole_groups)
    else {
        return Ok(None);
    };
    let plan = Arc::new(AggregateExec::try_new(
        AggregateMode::Final,
        last.group_expr().clone(),
        last.aggr_expr().to_vec(),
        last.filter_expr().to_vec(),
        fan_out.plan.clone(),
        last.input_schema(),
    )?);
    let mut finished =
        MapLambdaZarrExec::new(plan, fan_out.runs.clone(), false, fan_out.fan_out.clone())?;
    finished.finished = true;
    Ok(Some(Arc::new(finished)))
}

/// A Zarr sink over a fan-out of finished rows, moved onto the workers, each
/// writing the chunks its tiles fall in.
fn write(node: &Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(sink) = node.downcast_ref::<DataSinkExec>() else {
        return Ok(None);
    };
    let mut input = sink.input();
    loop {
        let rowwise = input.is::<ProjectionExec>()
            || input.is::<FilterExec>()
            || input.is::<SortExec>()
            || input.is::<SortPreservingMergeExec>()
            || input.is::<RepartitionExec>()
            || input.is::<CoalescePartitionsExec>()
            || input.is::<CooperativeExec>();
        if !rowwise {
            break;
        }
        input = input.children()[0];
    }
    let Some(fan_out) = input
        .downcast_ref::<MapLambdaZarrExec>()
        .filter(|f| f.finished)
    else {
        return Ok(None);
    };
    let Some(zarr) = sink.sink().downcast_ref::<ZarrDataSink>() else {
        return Ok(None);
    };
    if !fan_out.tiles_chunks(zarr.spec()) {
        return Ok(None);
    }
    let fragment = Arc::new(DataSinkExec::new(
        inline(sink.input().clone(), &fan_out.plan)?,
        Arc::new(zarr.clone().writing_chunks()),
        sink.sort_order().clone(),
    ));
    Ok(Some(Arc::new(MapLambdaZarrExec::new(
        fragment,
        fan_out.runs.clone(),
        false,
        fan_out.fan_out.clone(),
    )?)))
}

/// `plan` with the fan-out in it replaced by the fragment it runs.
fn inline(
    plan: Arc<dyn ExecutionPlan>,
    fragment: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if plan.is::<MapLambdaZarrExec>() {
        return Ok(fragment.clone());
    }
    let children = plan
        .children()
        .into_iter()
        .map(|child| inline(child.clone(), fragment))
        .collect::<Result<_>>()?;
    plan.with_new_children(children)
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

/// The scan's partitions cut into single chunks, less any that read nothing.
fn chunks(scan: &ZarrExec) -> Option<Vec<PartitionSpec>> {
    // No partitions means one unsliced read: VirtualiZarr, Icechunk, or a
    // limit, whose pushdown is only right over the whole table.
    if scan.limit().is_some() || scan.partitions().is_empty() {
        return None;
    }
    let meta = scan.store_meta()?;
    let var = meta.data_vars.first()?;
    // Unknown chunking (0) leaves each partition whole.
    let chunk_len = var
        .chunks
        .as_ref()
        .and_then(|c| c.first().copied())
        .unwrap_or(0);
    let inner = inner_axes(scan, meta, var);
    Some(
        scan.partitions()
            .iter()
            .flat_map(|spec| {
                let boxes = inner_boxes(&spec.extra, &inner);
                split_selection(&spec.outer, chunk_len, usize::MAX)
                    .into_iter()
                    .flat_map(move |outer| {
                        boxes.clone().into_iter().map(move |extra| PartitionSpec {
                            outer: outer.clone(),
                            extra,
                        })
                    })
            })
            .filter(|spec| !spec.is_empty())
            .collect(),
    )
}

/// The inner axes to cut at chunk boundaries, as `(coordinate index, extent,
/// chunk length)`: those of more than one chunk, and unfiltered, since a
/// partition's selection on an axis replaces the filter's.
fn inner_axes(
    scan: &ZarrExec,
    meta: &ZarrStoreMeta,
    var: &ZarrArrayMeta,
) -> Vec<(usize, u64, u64)> {
    let (Some(dims), Some(chunks)) = (&var.dimensions, &var.chunks) else {
        return vec![];
    };
    (1..var.shape.len())
        .filter_map(|axis| {
            let name = dims.get(axis)?;
            let filtered = scan.coord_filters().is_some_and(|f| f.get(name).is_some());
            let coord = meta.coords.iter().position(|c| &c.name == name)?;
            let (extent, chunk) = (var.shape[axis], *chunks.get(axis)?);
            (!filtered && chunk > 0 && extent > chunk).then_some((coord, extent, chunk))
        })
        .collect()
}

/// Each chunk-aligned box within `extra`, over the `inner` axes.
fn inner_boxes(
    extra: &[(usize, CoordSelection)],
    inner: &[(usize, u64, u64)],
) -> Vec<Vec<(usize, CoordSelection)>> {
    let mut boxes = vec![extra.to_vec()];
    for &(coord, extent, chunk) in inner {
        let whole = CoordSelection::Range(0, extent as usize);
        boxes = boxes
            .into_iter()
            .flat_map(|b| {
                let selected = b
                    .iter()
                    .find(|(c, _)| *c == coord)
                    .map_or(&whole, |(_, s)| s);
                split_selection(selected, chunk, usize::MAX)
                    .into_iter()
                    .map(|piece| {
                        let mut b = b.clone();
                        b.retain(|(c, _)| *c != coord);
                        b.push((coord, piece));
                        b
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
    }
    boxes
}

/// Whether the partial aggregate groups by every inner axis its chunks are cut
/// along, so that no group spans two tiles.
fn groups_within_tiles(partial: &AggregateExec, scan: &ZarrExec, chunks: &[PartitionSpec]) -> bool {
    let (Some(meta), Some(first)) = (scan.store_meta(), chunks.first()) else {
        return false;
    };
    let keys: Vec<&str> = partial
        .group_expr()
        .expr()
        .iter()
        .filter_map(|(expr, _)| expr.downcast_ref::<Column>().map(Column::name))
        .collect();
    partial.group_expr().is_single()
        && !first.extra.is_empty()
        && first.extra.iter().all(|(coord, _)| {
            meta.coords
                .get(*coord)
                .is_some_and(|c| keys.contains(&c.name.as_str()))
        })
}

/// `chunks` dealt a tile at a time, through time, in runs of at most
/// `per_invocation`, and whether each tile took only one.
fn by_tile(
    mut chunks: Vec<PartitionSpec>,
    per_invocation: usize,
) -> (Vec<Vec<PartitionSpec>>, bool) {
    let tile = |spec: &PartitionSpec| {
        let mut key: Vec<_> = spec.extra.iter().map(|(c, s)| (*c, s.as_range())).collect();
        key.sort();
        key
    };
    chunks.sort_by_cached_key(tile);
    let stacks: Vec<_> = chunks.chunk_by(|a, b| tile(a) == tile(b)).collect();
    let whole = stacks.iter().all(|stack| stack.len() <= per_invocation);
    let runs = stacks
        .into_iter()
        .flat_map(|stack| stack.chunks(per_invocation).map(<[_]>::to_vec))
        .collect();
    (runs, whole)
}

/// What a fan-out did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub chunks: usize,
    pub invocations: usize,
    /// From dispatching to the last output.
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
    runs: Vec<Vec<PartitionSpec>>,
    /// Whether no group's rows span two runs.
    whole_groups: bool,
    /// Whether the fragment ends in the final aggregate.
    finished: bool,
    fan_out: Arc<FanOut>,
    properties: Arc<PlanProperties>,
    report: Arc<Mutex<Report>>,
}

impl MapLambdaZarrExec {
    pub fn new(
        plan: Arc<dyn ExecutionPlan>,
        runs: Vec<Vec<PartitionSpec>>,
        whole_groups: bool,
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
            runs,
            whole_groups,
            finished: false,
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
        self.runs
            .iter()
            .map(|run| {
                serde_json::to_value(split::compact::compact(run))
                    .expect("partition specs serialize")
            })
            .collect()
    }

    fn chunks(&self) -> usize {
        self.runs.iter().map(Vec::len).sum()
    }

    /// Whether each of `spec`'s chunks lies in one tile, so that one worker writes
    /// it: along each axis the tiles are cut on, the grid is the whole source axis,
    /// and its chunks divide the source's.
    fn tiles_chunks(&self, spec: &SkeletonSpec) -> bool {
        let Some(tile) = self.runs.first().and_then(|run| run.first()) else {
            return false;
        };
        let scans = split::zarr_scans(&self.plan).unwrap_or_default();
        let Some(meta) = scans
            .first()
            .and_then(|scan| scan.downcast_ref::<ZarrExec>())
            .and_then(ZarrExec::store_meta)
        else {
            return false;
        };
        let Some(var) = meta.data_vars.first() else {
            return false;
        };
        tile.extra.iter().all(|(coord, _)| {
            let name = &meta.coords[*coord].name;
            let source_chunk = var
                .dimensions
                .as_ref()
                .and_then(|dims| dims.iter().position(|d| d == name))
                .and_then(|axis| var.chunks.as_ref()?.get(axis).copied());
            let extent = meta.coords[*coord].shape.first().copied();
            let axis = spec.coords.iter().position(|c| &c.name == name);
            match (source_chunk, extent, axis) {
                (Some(source_chunk), Some(extent), Some(axis)) => {
                    spec.coords[axis].values.len() as u64 == extent
                        && source_chunk % spec.chunks[axis] == 0
                }
                _ => false,
            }
        })
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
        let (chunks, invocations) = (self.chunks(), self.runs.len());
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "MapLambdaZarrExec: function={function}, chunks={chunks}, invocations={invocations}",
            ),
            DisplayFormatType::TreeRender => write!(
                f,
                "function={function}\nchunks={chunks}\ninvocations={invocations}",
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
            chunks: self.chunks(),
            invocations: inputs.len(),
            ..Report::default()
        };
        if inputs.is_empty() {
            return Ok(self.answer(futures::stream::empty().boxed()));
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
        let on_progress = self.fan_out.on_progress.clone();
        let total = inputs.len();
        // The workers each write their own chunks to a store made here first.
        let store = self
            .plan
            .downcast_ref::<DataSinkExec>()
            .and_then(|sink| sink.sink().downcast_ref::<ZarrDataSink>())
            .map(|zarr| (zarr.store_path().to_string(), zarr.spec().clone()));
        let run = futures::stream::once(async move {
            region::check(&locations).await?;
            if let Some((path, spec)) = store {
                tokio::task::spawn_blocking(move || create_skeleton(&path, &spec))
                    .await
                    .map_err(external)?
                    .map_err(DataFusionError::External)?;
            }
            if let Some(on_progress) = &on_progress {
                on_progress(0, total);
            }
            let worker = &fan_out.settings.worker_function;
            Ok::<_, DataFusionError>(
                fan_out
                    .session
                    .map_unordered(worker, inputs, "partitions", template)
                    .map_err(external)?
                    .map_err(external),
            )
        })
        .try_flatten();
        let report = self.report.clone();
        let return_stats = self.fan_out.settings.return_stats;
        let on_progress = self.fan_out.on_progress.clone();
        let mut succeeded = 0;
        let started = Instant::now();
        let batches = run
            .and_then(move |batch| {
                if let (Some(on_progress), 1..) = (&on_progress, batch.succeeded) {
                    succeeded += batch.succeeded;
                    on_progress(succeeded, total);
                }
                let decoded = batch
                    .outputs
                    .iter()
                    .map(|(index, output)| {
                        let output: Value = serde_json::from_str(output.get()).map_err(external)?;
                        Ok((*index, output))
                    })
                    .collect::<Result<Vec<_>>>()
                    .and_then(|outputs| {
                        let mut report = report.lock().expect("unpoisoned");
                        report.elapsed_s = started.elapsed().as_secs_f64();
                        report
                            .dispatcher_stats
                            .extend(batch.dispatcher_stats.iter().map(to_value));
                        if return_stats {
                            let bytes: usize = outputs
                                .iter()
                                .map(|(_, output)| output["batches"].as_str().map_or(0, str::len))
                                .sum();
                            let at_s = report.elapsed_s;
                            report.arrivals.push(json!({"at_s": at_s, "bytes": bytes}));
                            report
                                .invocation_stats
                                .extend(batch.invocation_stats.iter().map(to_value));
                            report.workers.extend(outputs.iter().map(|(index, output)| {
                                json!({
                                    "index": index,
                                    "timings_ms": output["timings_ms"],
                                    "s3_requests": output["s3_requests"],
                                    "chunks_in_flight": output["chunks_in_flight"],
                                })
                            }));
                        }
                        outputs
                            .iter()
                            .map(|(_, output)| decode(output))
                            .collect::<Result<Vec<_>>>()
                    })
                    .map(|batches| futures::stream::iter(batches.into_iter().flatten().map(Ok)));
                futures::future::ready(decoded)
            })
            .try_flatten();
        Ok(self.answer(batches.boxed()))
    }
}

impl MapLambdaZarrExec {
    /// The workers' batches, or when they write, one count of the rows they
    /// wrote, as the sink they each ran would answer.
    fn answer(
        &self,
        batches: BoxStream<'static, Result<RecordBatch>>,
    ) -> SendableRecordBatchStream {
        let schema = self.schema.clone();
        if !self.plan.is::<DataSinkExec>() {
            return Box::pin(RecordBatchStreamAdapter::new(schema, batches));
        }
        let total = futures::stream::once({
            let schema = schema.clone();
            async move {
                let written = batches
                    .try_fold(0, |total, batch| async move {
                        let counts = batch.column(0).as_primitive::<UInt64Type>();
                        Ok(total + counts.iter().flatten().sum::<u64>())
                    })
                    .await?;
                let count = Arc::new(UInt64Array::from(vec![written]));
                Ok(RecordBatch::try_new(schema, vec![count])?)
            }
        });
        Box::pin(RecordBatchStreamAdapter::new(schema, total))
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
