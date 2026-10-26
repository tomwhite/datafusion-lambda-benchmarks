# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk>=2", "numcodecs[pcodec]", "numpy", "xarray", "zarr>=3.1.3"]
# ///
"""
One month of a `demo.py` run, computed by xarray from Earthmover's ERA5 in one
process, against the month's row from the run:

    uv run benchmarks/demo_check.py results/demo-....json --month 2020-02

It reads the month from the run's repo a day at a time: from Earthmover's,
about 3 GB anonymously from us-east-1, so it is quickest run from there.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import icechunk
import numpy as np
import xarray as xr
import zarr
from zarr.codecs.numcodecs import PCodec


class BarePCodec(PCodec):
    """
    numcodecs' pcodec under the bare name the repo's metadata uses: zarr
    registers it only as `numcodecs.pcodec`, and refuses any other name.
    """

    @classmethod
    def from_dict(cls, data):
        return super().from_dict({**data, "name": "numcodecs.pcodec"})


# both sum in Float64, so only the order of summation differs
TOLERANCE_K = 1e-6


def storage(url: str) -> icechunk.Storage:
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        return icechunk.s3_storage(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/"),
            region="us-east-1",
            anonymous=True,
        )
    return icechunk.local_filesystem_storage(parsed.path)


def month_of(rows: list[dict], month: str | None) -> dict:
    for row in rows:
        if month is None or row["month"].startswith(month):
            return row
    raise SystemExit(f"no row for {month} in the run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="a saved demo.py run")
    parser.add_argument("--month", help="YYYY-MM (default: the run's first)")
    args = parser.parse_args()
    run = json.loads(args.results.read_text())
    row = month_of(run["rows"], args.month)
    start = max(datetime.fromisoformat(row["month"]), datetime.fromisoformat(run["start"]))
    next_month = datetime(start.year + start.month // 12, start.month % 12 + 1, 1)
    end = min(next_month, datetime.fromisoformat(run["end"]))

    zarr.registry.register_codec("pcodec", BarePCodec)
    session = icechunk.Repository.open(storage(run["url"])).readonly_session("main")
    ds = xr.open_zarr(session.store, group="single/spatial", chunks=None, consolidated=False)
    t2m = ds.t2m.sel(valid_time=slice(start, np.datetime64(end) - np.timedelta64(1, "h")))
    weights = np.cos(np.radians(ds.latitude.values.astype("float64")))[:, None]

    weighted = total_weight = 0.0
    low, high, n = np.inf, -np.inf, 0
    for day in range(0, t2m.sizes["valid_time"], 24):
        values = t2m.isel(valid_time=slice(day, day + 24)).values.astype("float64")
        weighted += (values * weights).sum()
        total_weight += weights.sum() * values.shape[0] * values.shape[2]
        low, high, n = min(low, values.min()), max(high, values.max()), n + values.size
        print(f"  {day // 24 + 1} days read", end="\r", flush=True)
    mean = weighted / total_weight

    print(f"\n{start:%Y-%m-%d} to {end:%Y-%m-%d}, {t2m.sizes['valid_time']} hours")
    print(f"  mean  xarray {mean:.9f}  map-lambda {row['mean_t2m']:.9f}")
    print(f"  min   xarray {low:.6f}  map-lambda {row['min_t2m']:.6f}")
    print(f"  max   xarray {high:.6f}  map-lambda {row['max_t2m']:.6f}")
    print(f"  n     xarray {n:,}  map-lambda {row['n']:,}")
    assert n == row["n"], "counts differ"
    assert low == row["min_t2m"] and high == row["max_t2m"], "extremes differ"
    difference = mean - row["mean_t2m"]
    assert abs(difference) < TOLERANCE_K, f"means differ by {difference:.2e} K"
    print("all as expected")


if __name__ == "__main__":
    main()
