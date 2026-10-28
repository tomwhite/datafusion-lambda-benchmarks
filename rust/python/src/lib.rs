//! `datafusion_lambda_benchmarks._native`: a session over Zarr tables that fans
//! out on map-lambda, as Python sees it.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use datafusion::physical_plan::{displayable, ExecutionPlan};
use datafusion::prelude::SessionContext;
use fanout::{FanOut, Report, Settings, Table};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Zarr tables, queried with each partial aggregate over a chunkwise scan
/// fanned out on `fragment` lambdas, and everything else in this process.
#[pyclass(frozen)]
struct Session {
    runtime: tokio::runtime::Runtime,
    ctx: SessionContext,
}

#[pymethods]
impl Session {
    #[new]
    #[pyo3(signature = (
        tables,
        endpoint_url = None,
        concurrency = 1000,
        dispatchers = None,
        chunks_per_invocation = 1,
        chunks_in_flight = None,
        allow_cross_region = false,
        return_stats = false,
        target_partitions = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        tables: HashMap<String, (String, BTreeMap<String, String>)>,
        endpoint_url: Option<String>,
        concurrency: usize,
        dispatchers: Option<usize>,
        chunks_per_invocation: usize,
        chunks_in_flight: Option<usize>,
        allow_cross_region: bool,
        return_stats: bool,
        target_partitions: Option<usize>,
    ) -> PyResult<Self> {
        let tables = tables
            .into_iter()
            .map(|(name, (location, options))| (name, Table { location, options }))
            .collect::<HashMap<_, _>>();
        let runtime = tokio::runtime::Runtime::new().map_err(err)?;
        let settings = Settings {
            concurrency,
            dispatchers,
            endpoint_url,
            chunks_per_invocation,
            chunks_in_flight,
            allow_cross_region,
            return_stats,
            ..Settings::default()
        };
        let ctx = py.detach(|| {
            runtime.block_on(async {
                let fan_out = Arc::new(FanOut::connect(settings).await);
                let ctx = fanout::session(fan_out, target_partitions);
                fanout::register(&ctx, &tables).await.map(|_| ctx)
            })
        });
        Ok(Self {
            runtime,
            ctx: ctx.map_err(err)?,
        })
    }

    /// The result as an Arrow IPC stream, and what each fan-out did as JSON.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let (schema, batches, reports) = py
            .detach(|| self.runtime.block_on(fanout::run(&self.ctx, query)))
            .map_err(err)?;
        let stream = split::encode_batches(&schema, &batches).map_err(err)?;
        let reports = serde_json::to_string::<Vec<Report>>(&reports).map_err(err)?;
        Ok((PyBytes::new(py, &stream), reports))
    }

    /// What each fan-out in the query's plan sends: its fragment, and each
    /// invocation's partitions as JSON.
    fn fragments(&self, py: Python<'_>, query: &str) -> PyResult<Vec<(String, Vec<String>)>> {
        let mut fragments = vec![];
        fanout::for_each_fan_out(&self.plan(py, query)?, |exec| {
            let inputs = exec.inputs().iter().map(|i| i.to_string()).collect();
            fragments.push((exec.fragment().to_string(), inputs));
        })
        .map_err(err)?;
        Ok(fragments)
    }

    /// Each fan-out's fragment, as `EXPLAIN` would show it.
    fn explain_fragments(&self, py: Python<'_>, query: &str) -> PyResult<Vec<String>> {
        let mut plans = vec![];
        fanout::for_each_fan_out(&self.plan(py, query)?, |exec| {
            plans.push(
                displayable(exec.fragment_plan().as_ref())
                    .indent(false)
                    .to_string(),
            );
        })
        .map_err(err)?;
        Ok(plans)
    }
}

impl Session {
    fn plan(&self, py: Python<'_>, query: &str) -> PyResult<Arc<dyn ExecutionPlan>> {
        py.detach(|| {
            self.runtime
                .block_on(async { self.ctx.sql(query).await?.create_physical_plan().await })
        })
        .map_err(err)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Session>()
}
