#!/usr/bin/env bash
# Deploys the workers and map-lambda's coordinator, in AWS_REGION.
set -euo pipefail

export AWS_PAGER=""

cd "$(dirname "$0")/../rust"

cargo lambda build --release --arm64 --package fragment --package coordinator

worker_role=$(aws iam get-role --role-name datafusion-lambda-benchmarks-worker-role --query Role.Arn --output text)
coordinator_role=$(aws iam get-role --role-name map-lambda-coordinator-role --query Role.Arn --output text)

cargo lambda deploy --binary-name fragment --role "$worker_role" fragment
cargo lambda deploy --binary-name coordinator --role "$coordinator_role" coordinator

# cargo lambda deploy keeps an existing function's role, which breaks it once
# that role is renamed or recreated
for pair in "fragment $worker_role" "coordinator $coordinator_role"; do
  set -- $pair
  aws lambda wait function-updated --function-name "$1"
  aws lambda update-function-configuration --function-name "$1" --role "$2" >/dev/null
done
