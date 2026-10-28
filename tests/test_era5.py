import json
from operator import itemgetter

import datafusion
import numpy as np
import pyarrow as pa
import pytest
import zarr
from map_lambda import LOCAL_ENDPOINT_URL

from datafusion_lambda_benchmarks.sql import explain, location, run, session

LOCAL = {"endpoint_url": LOCAL_ENDPOINT_URL, "concurrency": 100}

VARIABLE = "t2m"
DIMS = ("time", "latitude", "longitude")
TIME_UNITS = "hours since 1979-01-01 00:00:00"
TIME_START = np.datetime64("1979-01-01T00", "h")

# 60 chunks of 31 hours, January to mid-March 1979 on a 10° grid
SHAPE = (31 * 60, 19, 36)
CHUNKS = (31, 19, 36)
# as `single/temporal` is chunked: a run of hours of a tile, here 6 runs of 4 by 6 tiles
PENCILS = (310, 5, 6)
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


def write_group(path, with_values=True, chunks=CHUNKS):
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
        VARIABLE, shape=SHAPE, chunks=chunks, dtype="float32", dimension_names=DIMS
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
def era5_pencils(tmp_path):
    write_group(tmp_path, chunks=PENCILS)
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
GLOBAL_CLIMATOLOGY = """
SELECT latitude,
       longitude,
       date_part('hour', CAST(time AS TIMESTAMP)) AS hour,
       avg(t2m) - 273.15                          AS clim_c,
       count(*)                                   AS n
FROM era5
GROUP BY latitude, longitude, date_part('hour', CAST(time AS TIMESTAMP))
ORDER BY latitude DESC, longitude, hour
"""


@pytest.mark.lambda_integration
@pytest.mark.parametrize(
    "query", [WINDOWED["climatology"], GLOBAL_CLIMATOLOGY], ids=["windowed", "global"]
)
@pytest.mark.parametrize("chunks_per_invocation", [1, 4, 6])
def test_the_climatology_over_pencils_matches_one_process(
    era5_pencils, query, chunks_per_invocation
):
    result = run(
        query, {"era5": era5_pencils}, **LOCAL, chunks_per_invocation=chunks_per_invocation
    )

    assert result.chunks
    rows_close(result.table.to_pylist(), query_locally(era5_pencils, query))


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


def boxes(inputs):
    """Each input's one box, as its sorted `(start, end)` on every axis it restricts."""
    specs = [json.loads(i) for i in inputs]
    assert all(len(spec) == 1 for spec in specs)
    return [
        sorted([tuple(s["outer"]["Range"]), *(tuple(r["Range"]) for _, r in s["extra"])])
        for [s] in specs
    ]


def test_a_scan_of_pencils_fans_out_one_pencil_per_invocation(tmp_path):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    [(_, inputs)] = fragments({"era5": tmp_path.as_uri()}, QUERIES[0])

    pencils = [
        sorted([(t, t + PENCILS[0]), (y, min(y + PENCILS[1], SHAPE[1])), (x, x + PENCILS[2])])
        for t in range(0, SHAPE[0], PENCILS[0])
        for y in range(0, SHAPE[1], PENCILS[1])
        for x in range(0, SHAPE[2], PENCILS[2])
    ]
    assert sorted(boxes(inputs)) == sorted(pencils)


def tiles(inputs):
    """Each input's tiles, as the sorted inner ranges of its boxes."""
    return [
        {tuple(sorted(tuple(r["Range"]) for _, r in spec["extra"])) for spec in json.loads(i)}
        for i in inputs
    ]


def pencil_inputs(tmp_path, query, chunks_per_invocation):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    tables = {"era5": tmp_path.as_uri()}
    settings = {**LOCAL, "chunks_per_invocation": chunks_per_invocation}
    [(_, inputs)] = session(tables, **settings).fragments(query)
    return inputs


def test_a_climatology_over_pencils_is_dealt_a_tile_at_a_time(tmp_path):
    bands = SHAPE[0] // PENCILS[0]
    inputs = pencil_inputs(tmp_path, GLOBAL_CLIMATOLOGY, chunks_per_invocation=bands)

    assert len(inputs) == 4 * 6
    assert all(len(t) == 1 for t in tiles(inputs))
    # each tile's run of bands, compacted into one spec
    assert all(len(json.loads(i)) == 1 for i in inputs)


def test_a_tile_longer_than_an_invocation_takes_several(tmp_path):
    inputs = pencil_inputs(tmp_path, GLOBAL_CLIMATOLOGY, chunks_per_invocation=4)

    # 6 bands a tile: 4, then 2
    assert len(inputs) == 4 * 6 * 2
    assert all(len(t) == 1 for t in tiles(inputs))


def test_a_group_key_without_the_tiles_axes_is_dealt_in_listing_order(tmp_path):
    inputs = pencil_inputs(tmp_path, QUERIES[0], chunks_per_invocation=6)

    # the 6 tiles of a row of one band
    assert all(len(t) == 6 for t in tiles(inputs))


@pytest.mark.lambda_integration
@pytest.mark.parametrize(
    "climatology",
    [GLOBAL_CLIMATOLOGY, GLOBAL_CLIMATOLOGY.replace("FROM era5\n", f"FROM era5\nWHERE {WINDOW}\n")],
    ids=["all-time", "windowed"],
)
@pytest.mark.parametrize(("chunks", "on_the_workers"), [("24,5,6", True), ("24,10,6", False)])
def test_a_zarr_copy_of_the_climatology_over_pencils_matches_one_process(
    era5_pencils, tmp_path, climatology, chunks, on_the_workers
):
    out = tmp_path / "climatology.zarr"
    query = f"COPY ({climatology}) TO '{out}' STORED AS ZARR OPTIONS ('chunks' '{chunks}')"
    tables = {"era5": era5_pencils}
    settings = {**LOCAL, "chunks_per_invocation": SHAPE[0] // PENCILS[0]}

    # a chunk of 10 latitudes spans two tiles, so the store is written here
    [fragment] = session(tables, **settings).explain_fragments(query)
    assert ("ZarrDataSink" in fragment) == on_the_workers
    result = run(query, tables, **settings)

    expected = query_locally(era5_pencils, climatology)
    assert result.table.to_pylist() == [{"count": len(expected)}]
    group = zarr.open_group(out, mode="r")
    hours, lats, lons = (group[name][:] for name in ("hour", "latitude", "longitude"))
    assert list(group["clim_c"].metadata.dimension_names) == ["hour", "latitude", "longitude"]
    clim, n = group["clim_c"][:], group["n"][:]
    written = [
        {
            "latitude": lats[y],
            "longitude": lons[x],
            "hour": int(hours[h]),
            "clim_c": float(clim[h, y, x]),
            "n": int(n[h, y, x]),
        }
        for h in range(len(hours))
        for y in range(len(lats))
        for x in range(len(lons))
    ]
    key = itemgetter("latitude", "longitude", "hour")
    rows_close(sorted(written, key=key), sorted(expected, key=key))


@pytest.mark.parametrize(("chunks_per_invocation", "on_the_workers"), [(6, True), (4, False)])
def test_the_final_aggregate_runs_on_the_workers_when_each_tile_takes_one_invocation(
    tmp_path, chunks_per_invocation, on_the_workers
):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    tables = {"era5": tmp_path.as_uri()}
    settings = {**LOCAL, "chunks_per_invocation": chunks_per_invocation}

    [fragment] = session(tables, **settings).explain_fragments(GLOBAL_CLIMATOLOGY)
    plan = explain(GLOBAL_CLIMATOLOGY, tables, **settings).split("fragment:")[0]

    assert ("AggregateExec: mode=Final," in fragment) == on_the_workers
    assert ("mode=FinalPartitioned" in plan) != on_the_workers


def test_a_filtered_inner_axis_is_not_cut(tmp_path):
    write_group(tmp_path, with_values=False, chunks=PENCILS)
    [(_, inputs)] = fragments({"era5": tmp_path.as_uri()}, WINDOWED["climatology"])

    # the window's four runs of hours, each over the whole grid, where the filter applies
    assert len(inputs) == 4
    assert all(json.loads(i)[0].get("extra", []) == [] for i in inputs)


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
