"""
What a run cost, priced from the CloudWatch REPORT lines of its invocations.
"""

from typing import Any

from datafusion_lambda_benchmarks.prices import (
    GB_SECOND_USD,
    PRICING_AS_OF,
    REQUEST_USD,
    S3_GET_USD,
    S3_LIST_USD,
    gigabytes,
)


def actual(joined: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Cost of the invocations in `joined`, each with its REPORT line's
    `billed_ms` and `memory_size_mb`, or None if none has them.

    Counts only the invocations joined here: any whose REPORT line was
    missed is not included. S3 requests are each invocation's `s3_gets` and
    `s3_lists`, as its worker counted them through its store, so exclude the
    client's planning reads.
    """
    billable = [
        j for j in joined if j.get("billed_ms") is not None and j.get("memory_size_mb") is not None
    ]
    if not billable:
        return None

    gb_seconds = sum(gigabytes(j["memory_size_mb"]) * j["billed_ms"] / 1000 for j in billable)
    compute_usd = gb_seconds * GB_SECOND_USD
    request_usd = len(billable) * REQUEST_USD
    gets = round(sum(j["s3_gets"] for j in billable))
    lists = round(sum(j["s3_lists"] for j in billable))
    s3_usd = gets * S3_GET_USD + lists * S3_LIST_USD
    return {
        "invocations": len(billable),
        "gb_seconds": gb_seconds,
        "compute_usd": compute_usd,
        "request_usd": request_usd,
        "gets": gets,
        "lists": lists,
        "s3_usd": s3_usd,
        "total_usd": compute_usd + request_usd + s3_usd,
    }


def summarize_actual(cost: dict[str, Any]) -> str:
    """The cost lines for a run's report."""
    lambda_usd = cost["compute_usd"] + cost["request_usd"]
    s3 = f"${cost['s3_usd']:.4f} ({cost['gets']:,} GETs, {cost['lists']:,} LISTs)"
    return "\n".join(
        [
            f"cost:           ${cost['total_usd']:.4f} total (approximate)",
            f"                  lambda  ${lambda_usd:.4f} "
            f"({cost['gb_seconds']:.1f} GB-s, {cost['invocations']:,} requests)",
            f"                  S3      {s3}",
            f"                (arm64 and S3 Standard list prices as of {PRICING_AS_OF};",
            "                 excludes the free tier, retries and the client's own requests)",
        ]
    )
