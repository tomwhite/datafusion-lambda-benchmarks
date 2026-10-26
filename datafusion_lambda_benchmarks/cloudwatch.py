"""Lambda's REPORT lines from CloudWatch Logs Insights."""

import time
from itertools import pairwise
from typing import Any

# @initDuration is absent on warm invocations, which is how a cold start is detected.
REPORT_QUERY = """
fields @requestId, @duration, @billedDuration, @initDuration
| fields @maxMemoryUsed, @memorySize
| filter @type = "REPORT"
| limit 10000
"""

# Logs Insights' row cap per query, as in REPORT_QUERY's `limit`. It truncates
# silently, and the subset left is not a random one.
REPORT_LIMIT = 10_000

# Well under REPORT_LIMIT: a window also holds REPORT lines from other runs.
SLICE_RECORDS = 5_000

# The REPORT line is written after the invocation returns, on Lambda's clock.
WINDOW_PADDING_SECONDS = 60

# Insights gives @memorySize and @maxMemoryUsed in bytes of decimal megabytes.
MEGABYTE = 1_000_000


def query_windows(records: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """
    Contiguous epoch-second windows over the run, one per `SLICE_RECORDS`
    invocations, padded because the client's and Lambda's clocks only
    roughly agree.
    """
    start = int(min(r["invoke_start"] for r in records)) - WINDOW_PADDING_SECONDS
    end = int(max(r["invoke_end"] for r in records)) + WINDOW_PADDING_SECONDS
    ends = sorted(r["invoke_end"] for r in records)
    cuts = {int(ends[i]) for i in range(SLICE_RECORDS, len(ends), SLICE_RECORDS)}
    bounds = sorted({start, end, *(c for c in cuts if start < c < end)})
    return list(pairwise(bounds))


def query_reports(logs: Any, function: str, start: int, end: int) -> dict[str, dict[str, float]]:
    """`{req_id: report}` for every REPORT line of `function` in the window."""
    query_id = logs.start_query(
        logGroupName=f"/aws/lambda/{function}",
        startTime=start,
        endTime=end,
        queryString=REPORT_QUERY,
    )["queryId"]
    while (response := logs.get_query_results(queryId=query_id))["status"] in (
        "Scheduled",
        "Running",
    ):
        time.sleep(1)
    if response["status"] != "Complete":
        raise RuntimeError(f"Logs Insights query {response['status'].lower()} ({query_id})")
    found = {}
    for row in response["results"]:
        fields = {f["field"]: f["value"] for f in row}
        if (req_id := fields.pop("@requestId", None)) is not None:
            found[req_id] = {k: float(v) for k, v in fields.items() if k != "@ptr"}
    if len(found) >= REPORT_LIMIT:
        raise RuntimeError(
            f"{function}'s REPORT lines from {start} to {end} hit Logs Insights' row limit"
        )
    return found


def query_all_reports(
    logs: Any, function: str, ids: set[str], windows: list[tuple[int, int]], wait_s: float = 300
) -> dict[str, dict[str, float]]:
    """
    REPORT lines for `ids` in `windows`, rerunning the queries while
    CloudWatch, which ingests them some seconds behind, is still adding
    them: until every one is found, a rerun finds no more, or `wait_s`
    passes. Lambda occasionally never writes one.
    """
    deadline = time.monotonic() + wait_s
    matched = 0
    while True:
        found: dict[str, dict[str, float]] = {}
        for start, end in windows:
            found.update(query_reports(logs, function, start, end))
        before, matched = matched, len(ids & found.keys())
        print(f"  {function}: {matched:,} of {len(ids):,} REPORT lines", flush=True)
        if matched == len(ids) or 0 < matched == before or time.monotonic() > deadline:
            if missing := sorted(ids - found.keys()):
                print(f"  no REPORT line for {', '.join(missing[:5])}", flush=True)
            return found
        time.sleep(10)


def join(records: list[dict[str, Any]], found: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """
    Each of `records`, which carry map-lambda's `req_id`, with its REPORT
    line's numbers. Records with no REPORT line in `found` are dropped.
    """
    return [
        {
            **r,
            "duration_ms": line["@duration"],
            "billed_ms": line["@billedDuration"],
            "memory_size_mb": line["@memorySize"] / MEGABYTE,
            "max_memory_mb": line["@maxMemoryUsed"] / MEGABYTE,
            "init_ms": line.get("@initDuration"),
        }
        for r in records
        if (line := found.get(r["req_id"])) is not None
    ]
