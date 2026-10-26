import icechunk
import numpy as np
import pytest
import zarr
from map_lambda import LOCAL_ENDPOINT_URL

from datafusion_lambda_benchmarks.sql import run

LOCAL = {"endpoint_url": LOCAL_ENDPOINT_URL, "concurrency": 100}
GROUP = "single/spatial"
TIMES, LATS, LONS = 12, 5, 8


@pytest.fixture
def repo(tmp_path):
    """An icechunk repo with a variable in a group, one time step a chunk."""
    path = tmp_path / "repo"
    storage = icechunk.local_filesystem_storage(str(path))
    session = icechunk.Repository.create(storage).writable_session("main")
    group = zarr.group(store=session.store).require_group(GROUP)
    for name, n in [("time", TIMES), ("lat", LATS), ("lon", LONS)]:
        group.create_array(name, shape=(n,), dtype="int64", dimension_names=[name])[:] = range(n)
    t2m = group.create_array(
        "t2m",
        shape=(TIMES, LATS, LONS),
        chunks=(1, LATS, LONS),
        dtype="float64",
        dimension_names=["time", "lat", "lon"],
    )
    t2m[:] = values()
    session.commit("t2m")
    return path


def values():
    t, y, x = np.meshgrid(np.arange(TIMES), np.arange(LATS), np.arange(LONS), indexing="ij")
    return (t * 100 + y * 10 + x).astype("float64")


@pytest.mark.lambda_integration
def test_a_scan_of_an_icechunk_group_fans_out(repo):
    result = run(
        "SELECT lat, SUM(t2m) AS total FROM era5 WHERE time >= 3 AND time < 9 "
        "GROUP BY lat ORDER BY lat",
        {"era5": {"url": repo.as_uri(), "group": GROUP}},
        chunks_per_invocation=2,
        **LOCAL,
    )

    assert result.table.to_pylist() == [
        {"lat": lat, "total": total} for lat, total in enumerate(values()[3:9].sum(axis=(0, 2)))
    ]
    assert result.chunks == 6
    assert result.invocations == 3
