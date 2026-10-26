//! What the client and the worker share: a session over Zarr tables, the
//! fragment's wire format, and running a fragment over some of its scan's
//! partitions.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use base64::prelude::{Engine, BASE64_STANDARD};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
pub use datafusion::error::DataFusionError;
use datafusion::error::Result;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::SessionContext;
use datafusion_proto::bytes::{
    physical_plan_from_bytes_with_extension_codec, physical_plan_to_bytes_with_extension_codec,
};
use object_store::path::Path as ObjectPath;
use zarr_datafusion::datasource::factory::ZarrTableFactory;
use zarr_datafusion::optimizer::cardinality::budget::MemoryBudget;
use zarr_datafusion::optimizer::{
    CountStatisticsRule, MinMaxStatisticsRule, ZarrLimitPushdownRule,
};
use zarr_datafusion::physical_plan::codec::ZarrPhysicalCodec;
use zarr_datafusion::physical_plan::zarr_exec::{CachedRemoteStore, StoreKind, ZarrExec};
use zarr_datafusion::reader::icechunk_store::{open_icechunk_async, open_icechunk_remote};
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta_async;
use zarr_datafusion::reader::storage::{create_async_store, is_remote_url};
use zarrs_storage::AsyncReadableListableStorage;

pub use zarr_datafusion::physical_plan::partition::PartitionSpec;

mod bounded;
pub mod compact;
mod merge;
mod per_value;
mod requests;

pub use merge::merge_partials;
pub use per_value::PerValueRule;

/// A session state that reads Zarr tables, without zarr-datafusion's
/// `CardinalityRule`: its aggregate pushdown replaces an aggregate with one node
/// that has no partial form to split at.
pub fn builder() -> SessionStateBuilder {
    SessionStateBuilder::new()
        .with_default_features()
        .with_table_factory("ZARR".to_string(), Arc::new(ZarrTableFactory))
        .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))
        .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))
        .with_physical_optimizer_rule(Arc::new(ZarrLimitPushdownRule::new()))
}

/// A session from `builder`, planning for `target_partitions`, or by default
/// for this machine's cores.
pub fn session_from(
    builder: SessionStateBuilder,
    target_partitions: Option<usize>,
) -> SessionContext {
    let state = builder.build();
    let mut config = state.config().clone().with_information_schema(true);
    if let Some(n) = target_partitions {
        config = config.with_target_partitions(n);
    }
    SessionContext::new_with_state(
        SessionStateBuilder::new_from_existing(state)
            .with_config(config)
            .build(),
    )
}

/// A session that reads Zarr tables, planning for `target_partitions`.
pub fn session(target_partitions: usize) -> SessionContext {
    session_from(builder(), Some(target_partitions))
}

pub fn zarr_scans(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let mut scans = vec![];
    plan.apply(|node| {
        if node.downcast_ref::<ZarrExec>().is_some() {
            scans.push(node.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(scans)
}

/// Why `plan` can't be bound to some chunks, if it can't. Every scan in it would
/// be bound to the same partitions, which is only right for a single scan.
fn single_scan_problem(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<&'static str>> {
    Ok(match zarr_scans(plan)?.len() {
        0 => Some("it reads no Zarr chunks"),
        1 => None,
        _ => Some("it scans Zarr tables more than once, which needs a join across chunks"),
    })
}

/// Remote stores and their metadata, kept for the life of the execution
/// environment. A decoded scan has neither, so would otherwise make a new
/// connection and rediscover the store's arrays on every invocation.
async fn cached_remote(path: &str, kind: &StoreKind) -> Result<CachedRemoteStore> {
    type Cache = HashMap<(String, StoreKind), CachedRemoteStore>;
    static CACHE: OnceLock<tokio::sync::Mutex<Cache>> = OnceLock::new();
    if !is_remote_url(path) && *kind == StoreKind::Zarr {
        return Ok(None);
    }
    let mut cache = CACHE.get_or_init(Default::default).lock().await;
    let key = (path.to_string(), kind.clone());
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }
    let (store, prefix) = match kind {
        StoreKind::Zarr => create_async_store(path)
            .await
            .map_err(|e| DataFusionError::External(e.into()))?,
        StoreKind::Icechunk { group, anonymous } => (
            icechunk_store(path, *anonymous).await?,
            ObjectPath::from(group.as_deref().unwrap_or_default()),
        ),
    };
    let (_, meta) = infer_schema_with_meta_async(&store, &prefix)
        .await
        .map_err(DataFusionError::External)?;
    let cached = Some((requests::counted(store), prefix, meta));
    cache.insert(key, cached.clone());
    Ok(cached)
}

async fn icechunk_store(path: &str, anonymous: bool) -> Result<AsyncReadableListableStorage> {
    if is_remote_url(path) {
        open_icechunk_remote(path, anonymous).await
    } else {
        open_icechunk_async(path).await
    }
    .map_err(DataFusionError::External)
}

/// `plan` with each Zarr scan reading `partitions` instead of its own, through
/// a cached store when its location is remote. Also returns the scans, whose
/// I/O counters fill in once the plan runs.
async fn bind(
    plan: Arc<dyn ExecutionPlan>,
    partitions: &[PartitionSpec],
    budget: Option<u64>,
) -> Result<(Arc<dyn ExecutionPlan>, Vec<Arc<ZarrExec>>)> {
    if let Some(problem) = single_scan_problem(&plan)? {
        return Err(DataFusionError::Plan(format!(
            "can't run this fragment: {problem}"
        )));
    }
    let stores = zarr_scans(&plan)?
        .iter()
        .map(|scan| {
            let scan = scan.downcast_ref::<ZarrExec>().unwrap();
            (scan.path().to_string(), scan.store_kind().clone())
        })
        .collect::<Vec<_>>();
    let mut caches = HashMap::new();
    for (path, kind) in stores {
        let cached = cached_remote(&path, &kind).await?;
        caches.insert(path, cached);
    }
    let mut scans = vec![];
    let plan = plan
        .transform_down(|node| {
            let Some(scan) = node.downcast_ref::<ZarrExec>() else {
                return Ok(Transformed::no(node));
            };
            let cached = caches[scan.path()].clone();
            let meta = cached.as_ref().map(|(_, _, meta)| meta.clone());
            let bound = Arc::new(
                ZarrExec::new(
                    scan.schema().clone(),
                    scan.path().to_string(),
                    scan.projection().cloned(),
                    scan.limit(),
                    cached,
                    scan.coord_filters().cloned(),
                    None,
                )
                .with_partitions(partitions.to_vec())
                .with_store_meta(meta.or_else(|| scan.store_meta().cloned()))
                .with_stream_budget_bytes(budget.or(scan.stream_budget_bytes()))
                .with_store_kind(scan.store_kind().clone()),
            );
            scans.push(bound.clone());
            Ok(Transformed::yes(bound as Arc<dyn ExecutionPlan>))
        })?
        .data;
    Ok((plan, scans))
}

/// The memory a worker's scans may hold, shared by the partitions running:
/// `ZARR_MEM_BUDGET_BYTES` if set, else half of a Lambda's memory, leaving the
/// rest for decoding and the aggregate. zarr-datafusion's default, 256 MiB,
/// is narrower than a full-globe chunk row, which it would then read more than
/// once.
fn scan_budget() -> Option<u64> {
    MemoryBudget::from_env().map(|b| b.bytes).or_else(|| {
        let mb: u64 = std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
            .ok()?
            .parse()
            .ok()?;
        Some(mb * 1024 * 1024 / 2)
    })
}

pub fn encode_plan(plan: Arc<dyn ExecutionPlan>) -> Result<String> {
    Ok(
        BASE64_STANDARD.encode(physical_plan_to_bytes_with_extension_codec(
            plan,
            &ZarrPhysicalCodec,
        )?),
    )
}

pub fn decode_plan(ctx: &SessionContext, plan: &str) -> Result<Arc<dyn ExecutionPlan>> {
    let bytes = BASE64_STANDARD
        .decode(plan)
        .map_err(|e| DataFusionError::External(e.into()))?;
    physical_plan_from_bytes_with_extension_codec(&bytes, &ctx.task_ctx(), &ZarrPhysicalCodec)
}

pub fn encode_batches(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<u8>> {
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(Vec::new(), schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(writer.into_inner()?)
}

pub fn decode_batches(batches: &str) -> Result<Vec<RecordBatch>> {
    let bytes = BASE64_STANDARD
        .decode(batches)
        .map_err(|e| DataFusionError::External(e.into()))?;
    arrow::ipc::reader::StreamReader::try_new(bytes.as_slice(), None)?
        .collect::<std::result::Result<_, _>>()
        .map_err(Into::into)
}

/// Where a fragment's time went, in milliseconds.
#[derive(Debug, Default, serde::Serialize)]
pub struct Timings {
    pub decode: f64,
    /// Opening remote stores and reading their metadata, the first time only.
    pub bind: f64,
    /// Running the fragment, which includes the scan's reads below.
    pub execute: f64,
    pub encode: f64,
    /// The scan's own counters, summed over the fragment's scans.
    pub scan_metadata: f64,
    pub scan_coords: f64,
    pub scan_data: f64,
    /// Inside requests to a remote store, bodies included, summed over
    /// concurrent ones: the network's part of the scan's reads.
    pub fetch: f64,
}

/// The object store requests a fragment made, by S3's billing tiers.
#[derive(Debug, Default, serde::Serialize)]
pub struct Requests {
    /// GETs and HEADs.
    pub gets: u64,
    pub lists: u64,
}

/// A fragment's run over some partitions of its scan.
pub struct Run {
    /// Its output as base64 Arrow IPC.
    pub batches: String,
    pub timings: Timings,
    /// To a remote store, opening it included.
    pub requests: Requests,
    /// The plan as it ran, holding each operator's metrics.
    pub plan: Arc<dyn ExecutionPlan>,
    /// Chunks run at once.
    pub in_flight: usize,
}

/// The chunks a worker runs at once by default, so one is read while another
/// is computed on. More contend for the CPU and the scan budget: a 1769 MB
/// Lambda reports two cores but has one vCPU's time, and at three a
/// full-globe chunk no longer fit its share of the budget.
pub const DEFAULT_IN_FLIGHT: usize = 2;

/// Run a fragment over `partitions` of its scan, one per chunk, as a worker
/// does, `in_flight` of them at once, or [`DEFAULT_IN_FLIGHT`].
pub async fn run_fragment(
    plan: &str,
    partitions: &[PartitionSpec],
    in_flight: Option<usize>,
) -> Result<Run> {
    let in_flight = in_flight.unwrap_or(DEFAULT_IN_FLIGHT).max(1);
    let (gets, lists, fetched) = requests::made();
    let mut timings = Timings::default();
    let mut lap = Instant::now();
    let mut elapsed_ms = || {
        let ms = lap.elapsed().as_secs_f64() * 1000.0;
        lap = Instant::now();
        ms
    };
    let ctx = session(1);
    let plan = decode_plan(&ctx, plan)?;
    timings.decode = elapsed_ms();
    // zarr-datafusion divides a scan's budget among all its partitions
    let running = in_flight.min(partitions.len()).max(1) as u64;
    let budget = scan_budget().map(|b| b * partitions.len().max(1) as u64 / running);
    let (plan, scans) = bind(plan, partitions, budget).await?;
    let plan = merge_partials(per_value::rewrite_plan(plan)?, in_flight)?;
    timings.bind = elapsed_ms();
    let batches = collect(plan.clone(), ctx.task_ctx()).await?;
    timings.execute = elapsed_ms();
    let batches = BASE64_STANDARD.encode(encode_batches(&plan.schema(), &batches)?);
    timings.encode = elapsed_ms();
    for scan in scans {
        let stats = scan.io_stats();
        timings.scan_metadata += stats.metadata_time().as_secs_f64() * 1000.0;
        timings.scan_coords += stats.coord_time().as_secs_f64() * 1000.0;
        timings.scan_data += stats.data_time().as_secs_f64() * 1000.0;
    }
    let (gets_now, lists_now, fetched_now) = requests::made();
    timings.fetch = (fetched_now - fetched).as_secs_f64() * 1000.0;
    // a worker runs one fragment at a time, so the difference is this one's
    let requests = Requests {
        gets: gets_now - gets,
        lists: lists_now - lists,
    };
    Ok(Run {
        batches,
        timings,
        requests,
        plan,
        in_flight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scan_budget_comes_from_the_setting_then_the_lambdas_memory() {
        // nothing else reads these two
        std::env::remove_var("ZARR_MEM_BUDGET_BYTES");
        std::env::remove_var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE");
        assert_eq!(scan_budget(), None);

        std::env::set_var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "1769");
        assert_eq!(scan_budget(), Some(1769 * 1024 * 1024 / 2));

        std::env::set_var("ZARR_MEM_BUDGET_BYTES", "1000");
        assert_eq!(scan_budget(), Some(1000));

        std::env::remove_var("ZARR_MEM_BUDGET_BYTES");
        std::env::remove_var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE");
    }
}
