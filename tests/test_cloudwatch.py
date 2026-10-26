from itertools import pairwise

from datafusion_lambda_benchmarks.cloudwatch import (
    SLICE_RECORDS,
    join,
    query_all_reports,
    query_windows,
)


class Logs:
    """Logs Insights over `lines`, one REPORT line per request id at a time."""

    def __init__(self, lines: dict[str, tuple[float, dict[str, str]]]):
        self.lines = lines
        self.queries: list[tuple[int, int]] = []

    def start_query(self, *, logGroupName, startTime, endTime, queryString):
        self.queries.append((startTime, endTime))
        return {"queryId": str(len(self.queries) - 1)}

    def get_query_results(self, *, queryId):
        start, end = self.queries[int(queryId)]
        results = [
            [{"field": "@requestId", "value": req_id}]
            + [{"field": k, "value": v} for k, v in fields.items()]
            + [{"field": "@ptr", "value": "x"}]
            for req_id, (at, fields) in self.lines.items()
            if start <= at < end
        ]
        return {"status": "Complete", "results": results}


def test_windows_cover_the_run_a_slice_of_invocations_each():
    records = [{"invoke_start": 1000 + i, "invoke_end": 1001 + i} for i in range(SLICE_RECORDS + 1)]

    windows = query_windows(records)

    assert len(windows) == 2
    assert windows[0][0] == 940 and windows[-1][1] == 1001 + SLICE_RECORDS + 60
    assert all(a[1] == b[0] for a, b in pairwise(windows))


def test_reports_join_their_invocations():
    warm = {
        "@duration": "120.5",
        "@billedDuration": "121",
        "@memorySize": "1769000000",
        "@maxMemoryUsed": "200000000",
    }
    logs = Logs({"a": (1001, warm), "b": (1002, {**warm, "@initDuration": "80"}), "c": (1, warm)})
    records = [
        {"req_id": "a", "invoke_start": 1000, "invoke_end": 1001},
        {"req_id": "b", "invoke_start": 1000, "invoke_end": 1002},
    ]

    found = query_all_reports(logs, "fragment", {"a", "b"}, query_windows(records))
    joined = join(records, found)

    assert [j["req_id"] for j in joined] == ["a", "b"]
    assert joined[0]["memory_size_mb"] == 1769
    assert joined[0]["billed_ms"] == 121
    assert (joined[0]["init_ms"], joined[1]["init_ms"]) == (None, 80)


def test_reports_stop_when_a_rerun_finds_no_more(monkeypatch):
    monkeypatch.setattr("datafusion_lambda_benchmarks.cloudwatch.time.sleep", lambda s: None)
    warm = {"@duration": "1", "@billedDuration": "1", "@memorySize": "1", "@maxMemoryUsed": "1"}
    logs = Logs({"a": (1001, warm)})
    records = [{"invoke_start": 1000, "invoke_end": 1001}]

    found = query_all_reports(logs, "fragment", {"a", "never"}, query_windows(records))

    assert found.keys() == {"a"}
    assert len(logs.queries) == 2
