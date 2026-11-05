"""The `datafusion-lambda-benchmarks` command line."""

import click
import map_lambda.deploy

from datafusion_lambda_benchmarks.deploy import SPEC


@click.group()
def cli() -> None:
    """Query ERA5 Zarr with SQL on AWS Lambda."""


for _command in map_lambda.deploy.commands(SPEC):
    cli.add_command(_command)
