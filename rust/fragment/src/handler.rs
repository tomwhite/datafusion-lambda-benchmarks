use lambda_runtime::{Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use split::compact::{expand, CompactSpec};

/// Run a plan fragment over some partitions of its Zarr scan.
#[derive(Deserialize)]
pub(crate) struct Request {
    /// The fragment as base64 datafusion-proto bytes, the same for every invocation.
    plan: String,
    /// The scan partitions this invocation reads, replacing the fragment's own.
    partitions: Vec<CompactSpec>,
    /// How many of its chunks to run at once, else the worker's default.
    #[serde(default)]
    chunks_in_flight: Option<usize>,
}

#[derive(Serialize)]
pub(crate) struct Response {
    /// The fragment's output as a base64 Arrow IPC stream.
    batches: String,
    timings_ms: split::Timings,
    s3_requests: split::Requests,
    chunks_in_flight: usize,
}

pub(crate) async fn function_handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    let request = event.payload;
    let partitions = expand(request.partitions)?;
    let run = split::run_fragment(&request.plan, &partitions, request.chunks_in_flight).await?;
    Ok(Response {
        batches: run.batches,
        timings_ms: run.timings,
        s3_requests: run.requests,
        chunks_in_flight: run.in_flight,
    })
}
