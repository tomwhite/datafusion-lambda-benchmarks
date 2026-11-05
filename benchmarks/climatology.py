"""
The hourly climatology of `t2m` over Earthmover's ERA5 pencils,
`single/temporal`, written to a Zarr store by the workers, a chunk a tile.
The workers must be in the bucket's region, as the deploy spec fixes, and
allowed to write to the output's:

\b
    uv run datafusion-lambda-benchmarks deploy --from-source --allow-bucket ...
    eval "$(aws configure export-credentials --format env)"; export AWS_REGION=us-east-1
    uv run python benchmarks/climatology.py s3://.../climatology.zarr

By default a week, inside one band of pencils: it still writes every tile, but
reads one pencil a tile, as a rehearsal. `--start 1940-01-01 --end 2026-01-01`
is the whole record, about $3. Each invocation takes a tile's whole stack,
so that its rows are final and its own to write.
`climatology_check.py --output` checks a tile of the output against xarray.
**This invokes real Lambdas and is billed.**
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import click
from map_lambda import LOCAL_ENDPOINT_URL

from datafusion_lambda_benchmarks import runs, sql

URL = "s3://earthmover-icechunk-era5/icechunkV2"
GROUP = "single/temporal"

# a pencil: 8736 hours of a 12 x 12 tile, from the repo's first hour
TIME_START = datetime(1940, 1, 1)
BAND = 8736
TILES = 61 * 120
CELLS = 721 * 1440
CHUNK_BYTES = BAND * 12 * 12 * 4

RESULTS = Path(__file__).parent.parent / "results"

# TODO: the CAST works around zarr-datafusion's dictionary-encoded time.
QUERY = """
COPY (
  SELECT latitude,
         longitude,
         date_part('hour', CAST(valid_time AS TIMESTAMP)) AS hour,
         avg(t2m)                                         AS clim,
         count(*)                                         AS n
  FROM era5
  WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
  GROUP BY 1, 2, 3
) TO '{output}' STORED AS ZARR OPTIONS ('chunks' '{chunks}')
"""

# a chunk a tile, so that each worker writes its own
ZARR_CHUNKS = "24,12,12"


def bands(start: datetime, end: datetime) -> int:
    """How many bands of pencils `[start, end)` reaches into."""
    first = (start - TIME_START) // timedelta(hours=1) // BAND
    last = ((end - TIME_START) // timedelta(hours=1) - 1) // BAND
    return last - first + 1


def objects(output: str) -> int:
    """How many objects are under `output` now."""
    parsed = urlparse(output)
    if parsed.scheme != "s3":
        return sum(path.is_file() for path in Path(output).rglob("*"))
    pages = boto3.client("s3").get_paginator("list_objects_v2")
    pages = pages.paginate(Bucket=parsed.netloc, Prefix=parsed.path.lstrip("/"))
    return sum(page.get("KeyCount", 0) for page in pages)


@click.command(help=__doc__)
@click.argument("output")
@click.option(
    "--start",
    type=click.DateTime(),
    metavar="DATE",
    default=datetime(2020, 1, 6),
    show_default=True,
)
@click.option(
    "--end", type=click.DateTime(), metavar="DATE", default=datetime(2020, 1, 13), show_default=True
)
@click.option("--concurrency", default=990, show_default=True)
@click.option("--dispatchers", type=int)
@click.option("--chunks-in-flight", type=int)
@click.option("--local", is_flag=True, help=f"Against {LOCAL_ENDPOINT_URL}, with no CloudWatch.")
@click.option("--explain", is_flag=True, help="Print the plan and the workers' fragment, and stop.")
def main(
    output: str,
    start: datetime,
    end: datetime,
    concurrency: int,
    dispatchers: int | None,
    chunks_in_flight: int | None,
    local: bool,
    explain: bool,
) -> None:
    chunks_per_invocation = bands(start, end)
    tables: dict[str, sql.Table] = {"era5": {"url": URL, "group": GROUP}}
    query = QUERY.format(
        start=f"{start:%Y-%m-%d %H:%M:%S}",
        end=f"{end:%Y-%m-%d %H:%M:%S}",
        output=output,
        chunks=ZARR_CHUNKS,
    )
    settings: dict[str, Any] = {
        "endpoint_url": LOCAL_ENDPOINT_URL if local else None,
        "concurrency": concurrency,
        "dispatchers": dispatchers,
        "chunks_per_invocation": chunks_per_invocation,
        "chunks_in_flight": chunks_in_flight,
    }
    if explain:
        click.echo(sql.explain(query, tables, **settings))
        return
    chunks = f"{output.rstrip('/')}/clim/c/"
    if objects(chunks):
        # its chunks would be rewritten in place, and not counted
        raise click.BadParameter(
            f"{output} already has chunks: write to a new store", param_hint="OUTPUT"
        )
    options = click.get_current_context().params
    result = {
        "query": "climatology",
        "chunks_per_invocation": chunks_per_invocation,
        **options,
        **runs.run(query, tables, **settings),
    }
    written = objects(chunks)

    expected_chunks = chunks_per_invocation * TILES
    runs.print_run(result, concurrency, expected_chunks, CHUNK_BYTES)
    [row] = result["rows"]
    click.echo(f"{row['count']:,} rows in {written:,} chunks, expected {CELLS * 24:,} in {TILES:,}")
    ok = row["count"] == CELLS * 24 and written == TILES
    click.secho("all as expected" if ok else "PROBLEMS", fg="green" if ok else "red")
    saved = runs.save(RESULTS, "climatology", result)
    if not local:
        runs.report(json.loads(saved.read_text()), saved)


if __name__ == "__main__":
    main()
