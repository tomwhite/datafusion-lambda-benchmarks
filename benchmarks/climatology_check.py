"""
The climatology over one tile of Earthmover's ERA5 pencils, `single/temporal`,
through the fan-out, against xarray's `groupby("valid_time.hour").mean()` in
one process:

    export AWS_SKIP_SIGNATURE=true AWS_REGION=us-east-1
    uv run --with xarray --with "numcodecs[pcodec]" --with obstore \
        python benchmarks/climatology_check.py --local

With `--local` the workers are a `cargo lambda watch` here, reading a pencil
or two anonymously from us-east-1; without it they are the deployed lambdas,
as for `demo.py`, and billed. With `--output` it runs nothing, and checks the
tile's chunk of a `climatology.py` output instead, over the same window. A
tile is 12 by 12 cells, numbered from the north pole and the prime meridian.
"""

import argparse
import time
from datetime import datetime
from urllib.parse import urlparse

import icechunk
import numpy as np
import xarray as xr
import zarr
from map_lambda import LOCAL_ENDPOINT_URL

from datafusion_lambda_benchmarks.sql import run

BUCKET, PREFIX = "earthmover-icechunk-era5", "icechunkV2"
GROUP = "single/temporal"
TILE = 12
STEP = 0.25

# xarray sums in float64 too, so only the order of summation differs
TOLERANCE_K = 1e-6

QUERY = """
SELECT latitude,
       longitude,
       date_part('hour', CAST(valid_time AS TIMESTAMP)) AS hour,
       avg(t2m)                                         AS clim,
       count(*)                                         AS n
FROM era5
WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
  AND latitude BETWEEN {south} AND {north}
  AND longitude BETWEEN {west} AND {east}
GROUP BY 1, 2, 3
"""


def bounds(row: int, column: int) -> dict[str, float]:
    north = 90 - row * TILE * STEP
    west = column * TILE * STEP
    return {
        "north": north,
        "south": max(north - (TILE - 1) * STEP, -90),
        "west": west,
        "east": west + (TILE - 1) * STEP,
    }


def zarr_store(output: str) -> str | zarr.abc.store.Store:
    parsed = urlparse(output)
    if parsed.scheme != "s3":
        return output
    from obstore.store import S3Store

    store = S3Store(parsed.netloc, prefix=parsed.path.strip("/"))
    return zarr.storage.ObjectStore(store, read_only=True)


def from_zarr(output: str, row: int, column: int) -> dict:
    """The tile's rows in a Zarr store with the whole grid, reading its chunk alone."""
    group = zarr.open_group(zarr_store(output), mode="r")
    lats, lons, hours = (group[name][:] for name in ("latitude", "longitude", "hour"))
    rows = slice(row * TILE, (row + 1) * TILE)
    columns = slice(column * TILE, (column + 1) * TILE)
    clim, n = (group[name][:, rows, columns] for name in ("clim", "n"))
    return {
        (lat.item(), lon.item(), hour.item()): (clim[h, i, j].item(), n[h, i, j].item())
        for h, hour in enumerate(hours)
        for i, lat in enumerate(lats[rows])
        for j, lon in enumerate(lons[columns])
    }


def by_xarray(box: dict[str, float], start: datetime, end: datetime) -> dict:
    storage = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX, region="us-east-1", anonymous=True)
    session = icechunk.Repository.open(storage).readonly_session("main")
    ds = xr.open_zarr(session.store, group=GROUP, chunks=None, consolidated=False)
    t2m = ds.t2m.sel(
        valid_time=slice(start, np.datetime64(end) - np.timedelta64(1, "h")),
        latitude=slice(box["north"], box["south"]),
        longitude=slice(box["west"], box["east"]),
    )
    # in memory first: grouping the lazy array reads the store once a group
    by_hour = t2m.load().astype("float64").groupby("valid_time.hour")
    clim, n = by_hour.mean(), by_hour.count()
    return {
        (lat, lon, hour): (
            clim.sel(hour=hour, latitude=lat, longitude=lon).item(),
            n.sel(hour=hour, latitude=lat, longitude=lon).item(),
        )
        for hour in clim.hour.values
        for lat in clim.latitude.values
        for lon in clim.longitude.values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, default=10, help="the tile's row, from the north")
    parser.add_argument("--column", type=int, default=0, help="the tile's column, from 0°E")
    parser.add_argument("--start", type=datetime.fromisoformat, default=datetime(2020, 2, 1))
    parser.add_argument("--end", type=datetime.fromisoformat, default=datetime(2020, 3, 1))
    parser.add_argument("--local", action="store_true", help=f"against {LOCAL_ENDPOINT_URL}")
    parser.add_argument("--output", help="a climatology.py output to check, run over the window")
    args = parser.parse_args()

    box = bounds(args.row, args.column)
    query = QUERY.format(
        start=f"{args.start:%Y-%m-%d %H:%M:%S}", end=f"{args.end:%Y-%m-%d %H:%M:%S}", **box
    )
    started = time.monotonic()
    if args.output:
        rows = from_zarr(args.output, args.row, args.column)
        source = args.output
    else:
        tables = {"era5": {"url": f"s3://{BUCKET}/{PREFIX}", "group": GROUP}}
        endpoint_url = LOCAL_ENDPOINT_URL if args.local else None
        result = run(query, tables, endpoint_url=endpoint_url, concurrency=100)
        rows = {
            (r["latitude"], r["longitude"], r["hour"]): (r["clim"], r["n"])
            for r in result.table.to_pylist()
        }
        source = f"{result.invocations} invocation(s)"
    read = time.monotonic()
    expected = by_xarray(box, args.start, args.end)
    computed = time.monotonic()

    print(
        f"{box['north']}°N to {box['south']}°N, {box['west']}°E to {box['east']}°E, "
        f"{args.start:%Y-%m-%d} to {args.end:%Y-%m-%d}: {len(rows)} cell-hours"
    )
    assert rows.keys() == expected.keys(), "different cell-hours"
    worst = max(abs(rows[k][0] - expected[k][0]) for k in rows)
    print(f"  largest difference from xarray {worst:.2e} K, from {source}")
    print(f"  {read - started:.1f}s reading, {computed - read:.1f}s in xarray")
    assert all(rows[k][1] == expected[k][1] for k in rows), "counts differ"
    assert worst < TOLERANCE_K, f"means differ by {worst:.2e} K"
    print("all as expected")


if __name__ == "__main__":
    main()
