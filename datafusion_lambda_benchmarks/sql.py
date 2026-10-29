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
from tqdm import tqdm

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
    `dispatchers` (by default the fewest that hold `concurrency`),
    `chunks_per_invocation`, `chunks_in_flight` (how many of those a
    worker runs at once, by default 2), `allow_cross_region` (else a fan-out
    over an S3 bucket in another region than the workers is refused before
    it starts), `return_stats`, `target_partitions` and `on_progress`, called
    with how many of a fan-out's invocations have succeeded and how many it
    has: with none before it invokes, then as they arrive.
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


class _Progress:
    """A bar for each fan-out in turn, of its invocations that have succeeded."""

    def __init__(self) -> None:
        self.bar: tqdm | None = None

    def __call__(self, done: int, total: int) -> None:
        if done == 0:
            self.close()
            # no ETA: invocations finish in waves, so the rate swings
            self.bar = tqdm(
                total=total,
                bar_format="{n_fmt}/{total_fmt} invocations |{bar}| {elapsed}",
                disable=None,
            )
        elif self.bar is not None:
            self.bar.update(done - self.bar.n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def run(query: str, tables: dict[str, Table], progress: bool = False, **settings: Any) -> Result:
    """
    Run `query` over `tables`, with `session`'s settings, and with `progress`
    a bar on stderr for each fan-out, when it is a terminal.
    """
    bar = _Progress() if progress else None
    if bar is not None:
        settings["on_progress"] = bar
    try:
        stream, reports = session(tables, **settings).sql(query)
    finally:
        if bar is not None:
            bar.close()
    return Result(pa.ipc.open_stream(stream).read_all(), json.loads(reports))
