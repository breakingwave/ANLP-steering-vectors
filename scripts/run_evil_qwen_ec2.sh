#!/usr/bin/env bash
# Run evil trait on Qwen2.5-7B-Instruct on EC2 A10G, upload to S3, then STOP the instance.
#
# Usage:
#   S3_BUCKET=my-bucket OPENAI_API_KEY=sk-... ./scripts/run_evil_qwen_ec2.sh
#
# Optional env vars:
#   S3_PREFIX        — S3 key prefix (default: persona-vectors/runs)
#   S3_REGION        — AWS region for S3 (default: auto-detect from instance metadata)
#   GENERATOR_MODEL  — artifact generator (default: gpt-5.4-nano)
#   JUDGE_MODEL      — judge model (default: gpt-5.4-nano)
#   SKIP_STOP        — set to "1" to skip instance stop (for debugging)
set -euo pipefail

# --- Validate required env vars ---
: "${S3_BUCKET:?Set S3_BUCKET to the target S3 bucket name}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

S3_PREFIX="${S3_PREFIX:-persona-vectors/runs}"
S3_REGION="${S3_REGION:-}"
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
GENERATOR_MODEL="${GENERATOR_MODEL:-gpt-5.4-nano}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-nano}"
SKIP_STOP="${SKIP_STOP:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== EC2 evil-trait Qwen run ==="
echo "Trait: evil"
echo "Target model: $TARGET_MODEL"
echo "Generator: $GENERATOR_MODEL"
echo "Judge: $JUDGE_MODEL"
echo "S3: s3://$S3_BUCKET/$S3_PREFIX"
echo ""

# --- Run the pipeline ---
cd "$REPO_ROOT"

python scripts/run_three_traits.py \
  --profile paper_closest \
  --traits evil \
  --target-model "$TARGET_MODEL" \
  --generator-model "$GENERATOR_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --s3-upload \
  --s3-bucket "$S3_BUCKET" \
  --s3-prefix "$S3_PREFIX" \
  ${S3_REGION:+--s3-region "$S3_REGION"} \
  --s3-strict \
  --extract-judge-mode online \
  --eval-judge-mode online

echo ""
echo "=== Pipeline completed, results uploaded to S3 ==="

# --- Stop the instance (not terminate) ---
if [ "$SKIP_STOP" = "1" ]; then
  echo "SKIP_STOP=1 — skipping instance stop"
  exit 0
fi

echo "Stopping this EC2 instance in 10 seconds... (Ctrl-C to abort)"
sleep 10

# Get instance ID from EC2 metadata (IMDSv2)
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
if [ -n "$TOKEN" ]; then
  INSTANCE_ID=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
else
  INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
fi

if [ -z "$INSTANCE_ID" ]; then
  echo "WARNING: Could not detect EC2 instance ID. Not stopping."
  exit 0
fi

REGION=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || \
         curl -s http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "us-east-1")

echo "Stopping instance $INSTANCE_ID in $REGION..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
