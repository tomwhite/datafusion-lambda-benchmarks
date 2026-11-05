from click.testing import CliRunner
from map_lambda.deploy import spec

from datafusion_lambda_benchmarks.cli import cli
from datafusion_lambda_benchmarks.deploy import PROJECT, SPEC
from datafusion_lambda_benchmarks.runs import FUNCTION


def test_the_project_is_named_in_the_spec():
    assert PROJECT == "datafusion-lambda"


def test_the_spec_deploys_the_worker_the_client_calls():
    (fragment,) = spec.load(SPEC).functions
    assert fragment.name == FUNCTION


def test_the_spec_builds_from_this_workspace():
    deployed = spec.load(SPEC)
    assert deployed.workspace is not None
    assert (deployed.workspace / "Cargo.toml").exists()
    assert set(deployed.binaries.values()) <= {
        p.name for p in deployed.workspace.iterdir() if (p / "Cargo.toml").exists()
    }


def test_the_cli_has_map_lambda_s_deploy_commands():
    result = CliRunner().invoke(cli, ["deploy", "--help"])

    assert result.exit_code == 0, result.output
    assert "--allow-bucket BUCKET" in result.output
