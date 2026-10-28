import json

import datafusion
import numpy as np
import pyarrow as pa
import pytest
import zarr
from map_lambda import LOCAL_ENDPOINT_URL

from datafusion_lambda_benchmarks.sql import location, run, session

LOCAL = {"endpoint_url": LOCAL_ENDPOINT_URL, "concurrency": 100}

VARIABLE = "t2m"
DIMS = ("time", "latitude", "longitude")
TIME_UNITS = "hours since 1979-01-01 00:00:00"
TIME_START = np.datetime64("1979-01-01T00", "h")

# 60 chunks of 31 hours, January to mid-March 1979 on a 10° grid
SHAPE = (31 * 60, 19, 36)
CHUNKS = (31, 19, 36)
# neither bound falls on a chunk boundary
WINDOW = "time >= TIMESTAMP '1979-01-15 03:00:00' AND time < TIMESTAMP '1979-03-01 00:00:00'"

# The monthly global mean, and xarray-sql's zonal mean and climatology
# (https://xql.systems/xarray-sql/latest/geospatial/), over a window.
# TODO: the CASTs work around zarr-datafusion's dictionary-encoded time, which
# date_trunc and date_part reject.
WINDOWED = {
    "monthly-mean": f"""
SELECT date_trunc('month', CAST(time AS TIMESTAMP))                    AS month,
       sum(t2m * cos(radians(latitude))) / sum(cos(radians(latitude))) AS mean_t2m,
       min(t2m)                                                        AS min_t2m,
       max(t2m)                                                        AS max_t2m,
       count(*)                                                        AS n
FROM era5
WHERE {WINDOW}
GROUP BY 1
ORDER BY 1
""",
    "zonal-mean": f"""
SELECT latitude, avg(t2m) - 273.15 AS air_mean_c, count(*) AS n
FROM era5
WHERE {WINDOW}
GROUP BY latitude
ORDER BY latitude DESC
""",
    "climatology": f"""
SELECT latitude,
       longitude,
       date_part('hour', CAST(time AS TIMESTAMP)) AS hour,
       avg(t2m) - 273.15                          AS clim_c,
       count(*)                                   AS n
FROM era5
WHERE {WINDOW}
  AND latitude BETWEEN 25 AND 50
  AND longitude BETWEEN 235 AND 290
GROUP BY latitude, longitude, date_part('hour', CAST(time AS TIMESTAMP))
ORDER BY latitude DESC, longitude, hour
""",
}


def write_group(path, with_values=True):
    """An ERA5-shaped group, hourly from 1979 on a global grid, as xarray writes one."""
    group = zarr.open_group(path, mode="w", zarr_format=3)
    n_time, n_lat, n_lon = SHAPE
    coordinates = {
        "time": np.arange(n_time, dtype="int64"),
        "latitude": 90.0 - 180.0 * np.arange(n_lat) / (n_lat - 1),
        "longitude": 360.0 * np.arange(n_lon) / n_lon,
    }
    for name, values in coordinates.items():
        group.create_array(name, shape=values.shape, dtype=values.dtype, dimension_names=[name])[
            :
        ] = values
    group["time"].attrs.update({"units": TIME_UNITS, "calendar": "standard"})
    variable = group.create_array(
        VARIABLE, shape=SHAPE, chunks=CHUNKS, dtype="float32", dimension_names=DIMS
    )
    if with_values:
        # warm at the equator, cold at the poles, plus noise
        profile = 288 - 40 * np.sin(np.radians(coordinates["latitude"])) ** 2
        noise = np.random.default_rng(0).uniform(-15, 15, SHAPE)
        variable[:] = (profile[:, None] + noise).astype("float32")


@pytest.fixture
def era5(tmp_path):
    write_group(tmp_path)
    return tmp_path.as_uri()


@pytest.fixture
def era5_metadata(tmp_path):
    """A group with coordinates and no chunks written, which is all planning reads."""
    write_group(tmp_path, with_values=False)
    return tmp_path.as_uri()


def query_locally(url, query):
    """`query` over the whole Zarr group at `url` as table `era5`, flattened in one process."""
    group = zarr.open_group(location(url), mode="r")
    variable = group[VARIABLE]
    names = variable.metadata.dimension_names
    grids = np.meshgrid(*(group[name][:] for name in names), indexing="ij")
    columns = {name: grid.ravel() for name, grid in zip(names, grids, strict=True)}
    times = TIME_START + columns["time"].astype("timedelta64[h]")
    columns["time"] = pa.array(times.astype("datetime64[s]"))
    table = pa.table({**columns, VARIABLE: variable[:].ravel()})
    ctx = datafusion.SessionContext()
    ctx.from_arrow(table, name="era5")
    return ctx.sql(query).to_pylist()


# A spread of aggregates and groupings from the strong rows of "What fits".
# TODO: CAST(time AS TIMESTAMP) works around date_trunc rejecting
# zarr-datafusion's dictionary-encoded time.
QUERIES = [
    "SELECT latitude, avg(t2m) AS mean, stddev(t2m) AS sd, count(*) AS n "
    "FROM era5 GROUP BY latitude ORDER BY latitude",
    "SELECT count(*) AS n, sum(t2m) AS s FROM era5 WHERE latitude > 60 AND longitude < 90",
    "SELECT date_part('day', CAST(time AS TIMESTAMP)) AS day, max(t2m) - min(t2m) AS range "
    "FROM era5 WHERE time < TIMESTAMP '1979-01-08 00:00:00' GROUP BY 1 ORDER BY 1",
    # only the inner aggregate fans out
    "SELECT max(mean) AS warmest, count(*) AS n FROM "
    "(SELECT latitude, avg(t2m) AS mean FROM era5 GROUP BY latitude)",
]


def rows_close(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a.keys() == e.keys()
        for k in a:
            if isinstance(e[k], float):
                assert a[k] == pytest.approx(e[k], rel=1e-9), k
            else:
                assert a[k] == e[k], k


@pytest.mark.lambda_integration
@pytest.mark.parametrize("query", [*WINDOWED.values(), *QUERIES])
def test_queries_match_one_process(era5, query):
    result = run(query, {"era5": era5}, **LOCAL, chunks_per_invocation=7)

    assert result.chunks
    rows_close(result.table.to_pylist(), query_locally(era5, query))


@pytest.mark.lambda_integration
def test_every_invocation_has_its_workers_measurements(era5):
    result = run(QUERIES[0], {"era5": era5}, **LOCAL, chunks_per_invocation=7, return_stats=True)

    [report] = result.reports
    indexes = sorted(r["index"] for r in report["invocation_stats"])
    assert indexes == list(range(result.invocations))
    assert sorted(w["index"] for w in report["workers"]) == indexes
    # a local store isn't counted
    assert all(w["s3_requests"] == {"gets": 0, "lists": 0} for w in report["workers"])


@pytest.mark.lambda_integration
def test_dispatchers_share_the_invocations_with_the_same_answer(era5):
    result = run(QUERIES[0], {"era5": era5}, **LOCAL, dispatchers=3, return_stats=True)

    [report] = result.reports
    assert len(report["dispatcher_stats"]) == 3
    assert sum(c["dispatched"] for c in report["dispatcher_stats"]) == result.invocations
    rows_close(result.table.to_pylist(), query_locally(era5, QUERIES[0]))


@pytest.mark.lambda_integration
@pytest.mark.parametrize("in_flight", [1, 3])
def test_chunks_in_flight_give_the_same_answer(era5, in_flight):
    query = WINDOWED["zonal-mean"]
    result = run(
        query,
        {"era5": era5},
        **LOCAL,
        chunks_per_invocation=7,
        chunks_in_flight=in_flight,
        return_stats=True,
    )

    [report] = result.reports
    assert {w["chunks_in_flight"] for w in report["workers"]} == {in_flight}
    rows_close(result.table.to_pylist(), query_locally(era5, query))


# A persistence-forecast error: each cell against itself a day later, which
# joins chunks along time.
# TODO: the CASTs work around interval arithmetic rejecting zarr-datafusion's
# dictionary-encoded time.
SELF_JOIN = (
    "SELECT count(*) AS n, avg(abs(a.t2m - b.t2m)) AS mae FROM era5 a JOIN era5 b "
    "ON a.latitude = b.latitude AND a.longitude = b.longitude "
    "AND CAST(b.time AS TIMESTAMP) = CAST(a.time AS TIMESTAMP) + INTERVAL '24 hours'"
)


def fragments(tables, query):
    return session(tables, **LOCAL).fragments(query)


def test_a_partial_aggregate_fans_out_one_chunk_per_invocation(era5_metadata):
    [(_, inputs)] = fragments({"era5": era5_metadata}, QUERIES[0])

    starts = [json.loads(i)[0]["outer"]["Range"][0] for i in inputs]
    assert starts == [k * CHUNKS[0] for k in range(SHAPE[0] // CHUNKS[0])]


def test_only_the_chunks_a_filter_keeps_fan_out(era5_metadata):
    [(_, inputs)] = fragments({"era5": era5_metadata}, QUERIES[2])

    # hours 0 to 167
    assert len(inputs) == 168 // CHUNKS[0] + 1


def test_a_fragment_runs_one_partition_per_chunk_whatever_the_clients_cores(era5_metadata):
    # 6 chunks survive, fewer than the partitions planned for here
    [plan] = session({"era5": era5_metadata}, **LOCAL, target_partitions=16).explain_fragments(
        QUERIES[2]
    )

    assert "RoundRobinBatch" not in plan


@pytest.mark.parametrize(
    ("query", "why"),
    [
        (SELF_JOIN, "two scans, joined across chunks"),
        ("SELECT t2m FROM era5 LIMIT 3", "no aggregate"),
        (
            "SELECT max(r) AS r FROM (SELECT row_number() OVER (ORDER BY time) AS r FROM era5)",
            "a window over the whole table",
        ),
        (
            "SELECT count(*) AS n FROM (SELECT t2m FROM era5 LIMIT 100)",
            "a limit over the whole table",
        ),
    ],
)
def test_a_query_whose_aggregate_doesnt_work_chunk_by_chunk_doesnt_fan_out(
    era5_metadata, query, why
):
    assert fragments({"era5": era5_metadata}, query) == [], why


# TODO: CAST(time AS TIMESTAMP) because zarr-datafusion's time is UTC, and the
# single-process check's is naive.
UNSPLITTABLE = [
    "SELECT CAST(time AS TIMESTAMP) AS time, latitude, longitude, t2m FROM era5 "
    "ORDER BY time, latitude, longitude LIMIT 5",
    # answered from statistics, so there is no scan to distribute
    "SELECT count(*) AS n FROM era5",
    SELF_JOIN,
]


@pytest.mark.lambda_integration
@pytest.mark.parametrize("query", UNSPLITTABLE)
def test_a_query_that_cant_be_split_runs_locally(era5, query):
    result = run(query, {"era5": era5}, **LOCAL)

    assert result.chunks is None
    rows_close(result.table.to_pylist(), query_locally(era5, query))
