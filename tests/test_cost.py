from typing import Any

import pytest

from datafusion_lambda_benchmarks.cost import (
    GB_SECOND_USD,
    REQUEST_USD,
    S3_GET_USD,
    S3_LIST_USD,
    actual,
)


def test_actual():
    joined: list[dict[str, Any]] = [
        {"billed_ms": 1000, "memory_size_mb": 1024, "s3_gets": 0, "s3_lists": 0},
        {"billed_ms": 3000, "memory_size_mb": 1024, "s3_gets": 0, "s3_lists": 0},
        {"billed_ms": None, "memory_size_mb": 1024, "s3_gets": 0, "s3_lists": 0},
    ]
    spend = actual(joined)
    assert spend is not None
    assert spend["invocations"] == 2
    assert spend["gb_seconds"] == pytest.approx(4.0)
    assert spend["total_usd"] == pytest.approx(4.0 * GB_SECOND_USD + 2 * REQUEST_USD)


def test_actual_without_billing_data():
    assert actual([{"billed_ms": None, "memory_size_mb": None}]) is None


def test_actual_with_s3_requests():
    joined = [
        {"billed_ms": 1000, "memory_size_mb": 512, "s3_gets": 3, "s3_lists": 1},
        {"billed_ms": 1000, "memory_size_mb": 512, "s3_gets": 2, "s3_lists": 0},
    ]
    spend = actual(joined)
    assert spend is not None
    assert (spend["gets"], spend["lists"]) == (5, 1)
    assert spend["s3_usd"] == pytest.approx(5 * S3_GET_USD + 1 * S3_LIST_USD)
    assert spend["total_usd"] == pytest.approx(
        1.0 * GB_SECOND_USD + 2 * REQUEST_USD + spend["s3_usd"]
    )
