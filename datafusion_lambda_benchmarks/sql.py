"""
SQL over Zarr tables. Each partial aggregate over a chunkwise Zarr scan runs on
`fragment` lambdas, a run of chunks each, and the rest of the query here. A
query with nothing to fan out runs here entirely.
"""

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

import pyarrow as pa

from datafusion_lambda_benchmarks._native import Session


def location(url: str) -> str:
    """A table location as zarr-datafusion takes it: a path for `file://`, else the URL."""
    parsed = urlparse(url)
    return unquote(parsed.path) if parsed.scheme == "file" else url


Table = str | dict[str, str]
"""A Zarr group's URL, or a dict of its `url` and zarr-datafusion's table
options, such as the `group` to read in an icechunk repo."""


def _table(table: Table) -> tuple[str, dict[str, str]]:
    if isinstance(table, str):
        return location(table), {}
    options = dict(table)
    return location(options.pop("url")), options


def session(tables: dict[str, Table], **settings: Any) -> Session:
    """
    A session over `tables`, by name. `settings` are
    `endpoint_url`, for a local `cargo lambda watch`, `concurrency`,
    `coordinators` (by default the fewest that hold `concurrency`),
    `chunks_per_invocation`, `chunks_in_flight` (how many of those a
    worker runs at once, by default 2), `allow_cross_region` (else a fan-out
    over an S3 bucket in another region than the workers is refused before
    it starts), `return_stats` and `target_partitions`.
    """
    return Session({name: _table(table) for name, table in tables.items()}, **settings)


@dataclass
class Result:
    table: pa.Table
    reports: list[dict[str, Any]]
    """What each fan-out did: none when the query ran in this process."""

    @property
    def chunks(self) -> int | None:
        return sum(r["chunks"] for r in self.reports) if self.reports else None

    @property
    def invocations(self) -> int:
        return sum(r["invocations"] for r in self.reports)


def explain(query: str, tables: dict[str, Table], **settings: Any) -> str:
    """`query`'s physical plan, then the fragment each of its fan-outs runs on the workers."""
    s = session(tables, **settings)
    stream, _ = s.sql(f"EXPLAIN {query}")
    rows = pa.ipc.open_stream(stream).read_all().to_pylist()
    [plan] = [r["plan"] for r in rows if r["plan_type"] == "physical_plan"]
    return "\n".join([plan, *(f"fragment:\n{f}" for f in s.explain_fragments(query))])


def run(query: str, tables: dict[str, Table], **settings: Any) -> Result:
    """Run `query` over `tables`, with `session`'s settings."""
    stream, reports = session(tables, **settings).sql(query)
    return Result(pa.ipc.open_stream(stream).read_all(), json.loads(reports))
