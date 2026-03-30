#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="us-east-2"
S3_BUCKET="persona-vectors-prod-831959026511-us-east-2-an"
S3_PREFIX="persona-vectors/runs"
MODEL_PATH="/opt/dlami/nvme/hf-cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

cd /home/ubuntu/steering-vectors-for-multiturn-dialogue
source venv/bin/activate

python scripts/run_three_traits.py \
  --profile optimized \
  --target-model "$MODEL_PATH" \
  --generator-model gpt-5.4-nano \
  --judge-model gpt-5.4-nano \
  --s3-upload \
  --s3-bucket "$S3_BUCKET" \
  --s3-prefix "$S3_PREFIX" \
  --s3-region "$AWS_REGION" \
  --s3-strict

TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)

aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"
