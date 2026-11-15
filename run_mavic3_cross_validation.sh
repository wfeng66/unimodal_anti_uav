#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-yolo}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
MONTH_YEAR="$(date +%m_%Y)"
RESULTS_CSV="${RESULTS_DIR}/${MODEL,,}_mavic3_${MONTH_YEAR}.csv"
VALIDATION_SEQUENCES=(Avata M300 Mavic2 Pham4)

mkdir -p "$RESULTS_DIR"

for validation_sequence in "${VALIDATION_SEQUENCES[@]}"; do
    echo "Cross-validation fold: validation=${validation_sequence}, test=Mavic3"
    "$PYTHON" "${SCRIPT_DIR}/train.py" \
        --model "$MODEL" \
        --val "$validation_sequence" \
        --test Mavic3 \
        --results-csv "$RESULTS_CSV" \
        --epochs 10
        "$@"
done

echo "All folds complete. Results: ${RESULTS_CSV}"
