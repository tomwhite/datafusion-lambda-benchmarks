"""
A fanned-out query run with its workers' measurements, and reported on: what
it did, what CloudWatch's REPORT lines say it cost, and its timeline.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import boto3
import numpy as np

from datafusion_lambda_benchmarks import cost, sql
from datafusion_lambda_benchmarks.cloudwatch import join, query_all_reports, query_windows

FUNCTION = "fragment"
DISPATCHER = "dispatcher"


def describe(name: str, values: list[float]) -> str:
    """Percentiles of `values`, in milliseconds, on one line."""
    if not values:
        return f"{name:>9}: -"
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return f"{name:>9}: p50 {p50:7.1f}  p90 {p90:7.1f}  p99 {p99:7.1f}  max {max(values):7.1f} ms"


def run(query: str, tables: dict[str, sql.Table], **settings: Any) -> dict[str, Any]:
    """
    Run `query` as `sql.run` does, with one fan-out: its rows, and what each
    invocation, worker and dispatcher did.
    """
    started = time.monotonic()
    result = sql.run(query, tables, return_stats=True, **settings)
    wall = time.monotonic() - started
    [report] = result.reports
    workers = {w["index"]: w for w in report["workers"]}
    return {
        "chunks": report["chunks"],
        "invocations": [
            {
                **r,
                "s3_gets": workers[r["index"]]["s3_requests"]["gets"],
                "s3_lists": workers[r["index"]]["s3_requests"]["lists"],
            }
            for r in report["invocation_stats"]
        ],
        "dispatchers": report["dispatcher_stats"],
        "arrivals": report["arrivals"],
        "timings": [w["timings_ms"] for w in report["workers"]],
        "chunks_in_flight": sorted({w["chunks_in_flight"] for w in report["workers"]}),
        "rows": result.table.to_pylist(),
        "wall_s": wall,
        "fan_out_s": report["elapsed_s"],
    }


def print_run(result: dict[str, Any], concurrency: int, expected: int, chunk_bytes: int) -> None:
    """How `run`'s run went: its pruning, timings and requests."""
    wall = result["wall_s"]
    print(
        f"{result['chunks']:,} chunks after pruning (expected {expected:,}), "
        f"{len(result['invocations']):,} invocations at concurrency {concurrency} "
        f"through {len(result['dispatchers'])} dispatcher(s)"
    )
    print(
        f"wall {wall:.1f}s: fan-out {result['fan_out_s']:.1f}s, "
        f"planning and combine {wall - result['fan_out_s']:.1f}s"
    )
    scanned = result["chunks"] * chunk_bytes
    print(f"scanned {scanned / 1e12:.2f} TB uncompressed, {scanned / 1e9 / wall:.1f} GB/s")

    invokes = [(r["invoke_end"] - r["invoke_start"]) * 1000 for r in result["invocations"]]
    print(describe("invoke", invokes))
    for step in result["timings"][0]:
        print(describe(f"- {step}", [t[step] for t in result["timings"]]))
    print(f"chunks in flight on the workers: {result['chunks_in_flight']}")
    print(arrived(result))
    gets = sum(r["s3_gets"] for r in result["invocations"])
    lists = sum(r["s3_lists"] for r in result["invocations"])
    print(f"S3 requests from workers: {gets:,} GETs, {lists:,} LISTs")


def save(directory: Path, name: str, result: dict[str, Any]) -> Path:
    """`result` in `directory`, as JSON named for `name` and the time."""
    saved = directory / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    saved.parent.mkdir(exist_ok=True)
    saved.write_text(json.dumps(result, default=str))
    print(f"saved to {saved}")
    return saved


def arrived(result: dict[str, Any]) -> str:
    """When the workers' results reached the client, against when the workers finished."""
    arrivals = sorted(result["arrivals"], key=lambda a: a["at_s"])
    total = sum(a["bytes"] for a in arrivals)
    start = min(r["invoke_start"] for r in result["invocations"])
    last_worker = max(r["invoke_end"] for r in result["invocations"]) - start
    cumulative = np.cumsum([a["bytes"] for a in arrivals])
    at = {q: arrivals[int(np.searchsorted(cumulative, q * total))]["at_s"] for q in (0.5, 0.9, 1)}
    return (
        f" received: {total / 1e6:.1f} MB of results in {len(arrivals):,} frames, half by "
        f"{at[0.5]:.1f}s, 90% by {at[0.9]:.1f}s, all by {at[1]:.1f}s; "
        f"the last worker finished at {last_worker:.1f}s"
    )


def draw(result: dict[str, Any], invocations: list[dict[str, Any]], saved: Path) -> None:
    """The run's timeline, as an SVG beside its results."""
    # needs map-lambda's diagnostics extra, for matplotlib
    from map_lambda import timeline

    figure = timeline.generate_timeline(
        invocations,
        title=f"{result['query']}: {result['chunks']:,} chunks, "
        f"{result['chunks_per_invocation']} an invocation",
    )
    path = saved.with_suffix(".svg")
    figure.savefig(path)
    print(f"timeline in {path}")


def report(result: dict[str, Any], saved: Path) -> None:
    """
    What CloudWatch's REPORT lines say about a saved run, with its `query`
    and `chunks_per_invocation`, and what it cost, and its timeline.
    """
    print("waiting for CloudWatch Logs Insights...", flush=True)
    # where the Lambda client invoked: boto3 reads only AWS_DEFAULT_REGION
    logs = boto3.client("logs", region_name=os.environ.get("AWS_REGION"))
    # the dispatchers' REPORT lines fall in their workers' windows too
    windows = query_windows(result["invocations"])
    workers = join(
        result["invocations"],
        query_all_reports(logs, FUNCTION, {r["req_id"] for r in result["invocations"]}, windows),
    )
    dispatchers = join(
        result["dispatchers"],
        query_all_reports(logs, DISPATCHER, {c["req_id"] for c in result["dispatchers"]}, windows),
    )
    cold = sum(w["init_ms"] is not None for w in workers)
    print(
        f"{len(workers):,} of {len(result['invocations']):,} workers, {cold:,} cold; "
        f"{len(dispatchers)} of {len(result['dispatchers'])} dispatchers"
        + (
            " (the rest are missing from the cost)"
            if len(workers) + len(dispatchers)
            < len(result["invocations"]) + len(result["dispatchers"])
            else ""
        )
    )
    print(describe("handler", [w["duration_ms"] for w in workers]))
    # past its own elapsed_ms, it is still streaming results to the client
    print(describe("dispatch", [c["duration_ms"] for c in dispatchers]))
    if workers:
        print(
            f"   memory: max {max(w['max_memory_mb'] for w in workers):.0f} MB "
            f"of {max(w['memory_size_mb'] for w in workers):.0f} MB"
        )
    # dispatchers make no S3 requests
    spend = cost.actual(workers + [{**c, "s3_gets": 0, "s3_lists": 0} for c in dispatchers])
    print(cost.summarize_actual(spend) if spend else "no REPORT lines found, so no cost")
    ran = {w["req_id"]: w for w in workers}
    draw(
        result,
        [
            {**r, "execution_ms": w["duration_ms"], "init_ms": w["init_ms"]}
            if (w := ran.get(r["req_id"]))
            else r
            for r in result["invocations"]
        ],
        saved,
    )
