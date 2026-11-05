# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.1",
#     "icechunk>=2",
#     "numcodecs[pcodec]",
#     "numpy",
#     "xarray",
#     "zarr>=3.1.3",
# ]
# ///
"""
One month of a `demo.py` run, computed by xarray from Earthmover's ERA5 in one
process, against the month's row from the run:

\b
    uv run benchmarks/demo_check.py results/demo-....json --month 2020-02

It reads the month from the run's repo a day at a time: from Earthmover's,
about 3 GB anonymously from us-east-1, so it is quickest run from there.
"""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import click
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
    raise click.ClickException(f"no row for {month} in the run")


@click.command(help=__doc__)
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--month", metavar="YYYY-MM", help="The month to check. [default: the run's first]")
def main(results: Path, month: str | None) -> None:
    run = json.loads(results.read_text())
    row = month_of(run["rows"], month)
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
        click.echo(f"  {day // 24 + 1} days read\r", nl=False)
    mean = weighted / total_weight

    click.echo(f"\n{start:%Y-%m-%d} to {end:%Y-%m-%d}, {t2m.sizes['valid_time']} hours")
    click.echo(f"  mean  xarray {mean:.9f}  map-lambda {row['mean_t2m']:.9f}")
    click.echo(f"  min   xarray {low:.6f}  map-lambda {row['min_t2m']:.6f}")
    click.echo(f"  max   xarray {high:.6f}  map-lambda {row['max_t2m']:.6f}")
    click.echo(f"  n     xarray {n:,}  map-lambda {row['n']:,}")
    if n != row["n"]:
        raise click.ClickException("the counts differ")
    if low != row["min_t2m"] or high != row["max_t2m"]:
        raise click.ClickException("the extremes differ")
    difference = mean - row["mean_t2m"]
    if abs(difference) >= TOLERANCE_K:
        raise click.ClickException(f"the means differ by {difference:.2e} K")
    click.secho("all as expected", fg="green")


if __name__ == "__main__":
    main()
