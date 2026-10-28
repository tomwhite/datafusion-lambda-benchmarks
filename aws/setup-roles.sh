#!/usr/bin/env bash
# Once, before deploy.sh. Wait a few seconds afterwards for IAM to apply.
set -euo pipefail

export AWS_PAGER=""

WORKER_ROLE=datafusion-lambda-benchmarks-worker-role
DISPATCHER_ROLE=map-lambda-dispatcher-role

# The dispatcher role is shared with other map-lambda projects, so it may already exist
for role in $WORKER_ROLE $DISPATCHER_ROLE; do
  aws iam create-role --role-name $role --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' >/dev/null 2>&1 || true
  aws iam attach-role-policy --role-name $role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
done

# Inline policy names must be unique across projects sharing the dispatcher role
aws iam put-role-policy --role-name $DISPATCHER_ROLE \
  --policy-name invoke-datafusion-lambda-benchmarks-workers --policy-document '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:*:*:function:fragment"
  }]
}'

