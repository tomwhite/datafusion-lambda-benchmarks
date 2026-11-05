"""The lambdas' deploy spec."""

from pathlib import Path

import map_lambda.deploy

SPEC = Path(__file__).with_name("deploy.toml")

PROJECT = map_lambda.deploy.project(SPEC)
