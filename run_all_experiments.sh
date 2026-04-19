#!/bin/bash
# Run all persona drift experiments: evil + sycophancy x last + max aggregation
# Usage: bash run_all_experiments.sh

set -e

cd "$(dirname "$0")"

export HF_TOKEN=<token>
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

echo "========================================"
echo " Experiment 1/4: evil | last"
echo "========================================"
python3 simulate_persona_drift.py --vector-file evil.json --aggregation last --output persona_drift.png

echo "========================================"
echo " Experiment 2/4: evil | max"
echo "========================================"
python3 simulate_persona_drift.py --vector-file evil.json --aggregation max --output persona_drift.png

echo "========================================"
echo " Experiment 3/4: sycophancy | last"
echo "========================================"
python3 simulate_persona_drift.py --vector-file sycophancy.json --aggregation last --output persona_drift.png

echo "========================================"
echo " Experiment 4/4: sycophancy | max"
echo "========================================"
python3 simulate_persona_drift.py --vector-file sycophancy.json --aggregation max --output persona_drift.png

echo ""
echo "All experiments complete. Results in runs/"
ls -lh runs/
