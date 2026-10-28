"""
The demo: the monthly area-weighted global mean of `t2m` over Earthmover's
ERA5, an Icechunk repo in us-east-1, fanned out from a window's hourly
chunks. The workers must be in the bucket's region:

    AWS_REGION=us-east-1 ./aws/deploy.sh
    eval "$(aws configure export-credentials --format env)"; export AWS_REGION=us-east-1
    uv run python benchmarks/demo.py --start 2020-01-01 --end 2020-04-01

It checks that pruning kept exactly the window's chunks, each month's count,
and that the temperatures are plausible; `demo_check.py` checks a month's
values against xarray, and `copernicus_check.py` every month's mean against
Copernicus's published daily means. **This invokes real Lambdas and is billed.**
"""

import argparse
import asyncio
import calendar
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import icechunk
import numpy as np
from map_lambda import LOCAL_ENDPOINT_URL
from zarr.core.buffer import default_buffer_prototype

from datafusion_lambda_benchmarks import runs, sql

URL = "s3://earthmover-icechunk-era5/icechunkV2"
GROUP = "single/spatial"
VARIABLE = "t2m"

RESULTS = Path(__file__).parent.parent / "results"

# TODO: the CASTs work around zarr-datafusion's dictionary-encoded time, as in
# era5.MONTHLY_MEAN.
QUERY = """
SELECT date_trunc('month', CAST(valid_time AS TIMESTAMP))              AS month,
       sum(t2m * cos(radians(latitude))) / sum(cos(radians(latitude))) AS mean_t2m,
       min(t2m)                                                        AS min_t2m,
       max(t2m)                                                        AS max_t2m,
       count(*)                                                        AS n
FROM era5
WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
GROUP BY 1
ORDER BY 1
"""

# kelvin, far wider than any month's global mean or extremes in ERA5
PLAUSIBLE_MEAN = (280, 295)
PLAUSIBLE_RANGE = (170, 340)


def bounds(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": f"{start:%Y-%m-%d %H:%M:%S}", "end": f"{end:%Y-%m-%d %H:%M:%S}"}


def metadata(url: str) -> dict[str, Any]:
    """The variable's array metadata, read without its codecs, which zarr doesn't know."""
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        # TODO: the bucket's own region, once the workers can take one
        storage = icechunk.s3_storage(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/"),
            region="us-east-1",
            anonymous=True,
        )
    else:
        storage = icechunk.local_filesystem_storage(parsed.path)
    store = icechunk.Repository.open(storage).readonly_session("main").store
    buffer = asyncio.run(
        store.get(f"{GROUP}/{VARIABLE}/zarr.json", prototype=default_buffer_prototype())
    )
    return json.loads(buffer.to_bytes())


def months(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Each calendar month overlapping [start, end), clipped to it."""
    spans = []
    first = datetime(start.year, start.month, 1)
    while first < end:
        last = first + timedelta(days=calendar.monthrange(first.year, first.month)[1])
        spans.append((max(first, start), min(last, end)))
        first = last
    return spans


def check(rows: list[dict[str, Any]], start: datetime, end: datetime, cells: int) -> list[str]:
    """What is wrong with the rows, if anything, for hourly steps of `cells` values."""
    spans = months(start, end)
    expected = [(s.year, s.month) for s, _ in spans]
    if [(r["month"].year, r["month"].month) for r in rows] != expected:
        return [f"expected the {len(expected)} months from {start:%Y-%m}"]
    problems = []
    for row, (first, last) in zip(rows, spans, strict=True):
        label = f"{first:%Y-%m}"
        n = int((last - first) / timedelta(hours=1)) * cells
        if row["n"] != n:
            problems.append(f"{label}: {row['n']:,} values, expected {n:,}")
        if not PLAUSIBLE_MEAN[0] < row["mean_t2m"] < PLAUSIBLE_MEAN[1]:
            problems.append(f"{label}: mean {row['mean_t2m']:.2f} K")
        if not PLAUSIBLE_RANGE[0] < row["min_t2m"] <= row["max_t2m"] < PLAUSIBLE_RANGE[1]:
            problems.append(f"{label}: from {row['min_t2m']:.2f} to {row['max_t2m']:.2f} K")
    return problems


def settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "endpoint_url": LOCAL_ENDPOINT_URL if args.local else None,
        "concurrency": args.concurrency,
        "dispatchers": args.dispatchers,
        "chunks_per_invocation": args.chunks_per_invocation,
        "chunks_in_flight": args.chunks_in_flight,
        "allow_cross_region": args.allow_cross_region,
    }


def print_problems(rows: list[dict[str, Any]], problems: list[str]) -> None:
    print(f"{len(rows)} rows: " + ("all as expected" if not problems else "PROBLEMS"))
    for problem in problems[:20]:
        print(f"  {problem}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL, help="the Icechunk repo")
    parser.add_argument("--start", type=datetime.fromisoformat, default=datetime(2020, 1, 1))
    parser.add_argument("--end", type=datetime.fromisoformat, default=datetime(2020, 4, 1))
    parser.add_argument("--concurrency", type=int, default=1000)
    parser.add_argument("--dispatchers", type=int)
    parser.add_argument("--chunks-per-invocation", type=int, default=24)
    parser.add_argument("--chunks-in-flight", type=int)
    parser.add_argument(
        "--local", action="store_true", help=f"against {LOCAL_ENDPOINT_URL}, with no CloudWatch"
    )
    parser.add_argument(
        "--allow-cross-region",
        action="store_true",
        help="read a bucket in another region than the workers, billed per GB",
    )
    parser.add_argument(
        "--explain", action="store_true", help="print the plan and the workers' fragment, and stop"
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="RESULTS",
        help="report on a run saved in RESULTS from CloudWatch, without running it again",
    )
    parser.set_defaults(query="demo")
    args = parser.parse_args()
    if args.report:
        runs.report(json.loads(args.report.read_text()), args.report)
        return

    meta = metadata(args.url)
    chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
    chunk_bytes = math.prod(chunk_shape) * np.dtype(meta["data_type"]).itemsize
    cells = math.prod(meta["shape"][1:])
    tables: dict[str, sql.Table] = {"era5": {"url": args.url, "group": GROUP}}
    query = QUERY.format(**bounds(args.start, args.end))
    if args.explain:
        print(sql.explain(query, tables, **settings(args)))
        return
    result = {**vars(args), **runs.run(query, tables, **settings(args))}

    hours = (args.end - args.start) / timedelta(hours=1)
    runs.print_run(result, args.concurrency, math.ceil(hours / chunk_shape[0]), chunk_bytes)
    for row in result["rows"]:
        print(
            f"  {row['month']:%Y-%m}  mean {row['mean_t2m']:.4f} K  "
            f"min {row['min_t2m']:.2f}  max {row['max_t2m']:.2f}  n {row['n']:,}"
        )
    print_problems(result["rows"], check(result["rows"], args.start, args.end, cells))
    saved = runs.save(RESULTS, "demo", result)
    if not args.local:
        runs.report(result, saved)


if __name__ == "__main__":
    main()
