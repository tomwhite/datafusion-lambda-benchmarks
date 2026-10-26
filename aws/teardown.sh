#!/usr/bin/env bash
set -uo pipefail

export AWS_PAGER=""

aws lambda delete-function --function-name fragment
aws logs delete-log-group --log-group-name /aws/lambda/fragment

aws iam delete-role-policy --role-name map-lambda-coordinator-role --policy-name invoke-datafusion-lambda-benchmarks-workers
aws iam detach-role-policy --role-name datafusion-lambda-benchmarks-worker-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name datafusion-lambda-benchmarks-worker-role
