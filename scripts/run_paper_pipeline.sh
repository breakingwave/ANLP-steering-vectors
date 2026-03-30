#!/usr/bin/env bash
# Backward-compatible wrapper around the Python 3-trait orchestrator.
#
# Legacy usage:
#   ./scripts/run_paper_pipeline.sh [TARGET_MODEL] [GENERATOR_MODEL] [JUDGE_MODEL]
#
# Modern usage:
#   ./scripts/run_paper_pipeline.sh --profile optimized --s3-upload --s3-bucket my-bucket ...
set -euo pipefail

if [[ "${1:-}" == "" || "${1:-}" == --* ]]; then
  python scripts/run_three_traits.py "$@"
  exit 0
fi

TARGET_MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
GENERATOR_MODEL="${2:-gpt-5.4-nano}"
JUDGE_MODEL="${3:-gpt-5.4-nano}"

python scripts/run_three_traits.py \
  --profile paper_closest \
  --target-model "${TARGET_MODEL}" \
  --generator-model "${GENERATOR_MODEL}" \
  --judge-model "${JUDGE_MODEL}" \
  --extract-judge-mode online \
  --eval-judge-mode online
