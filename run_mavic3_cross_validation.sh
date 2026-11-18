#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-yolo}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
MONTH_YEAR="$(date +%m_%Y)"
RESULTS_CSV="${RESULTS_DIR}/${MODEL,,}_mavic3_${MONTH_YEAR}.csv"
VALIDATION_SEQUENCES=(Avata M300 Mavic2 Pham4)

mkdir -p "$RESULTS_DIR"

for validation_sequence in "${VALIDATION_SEQUENCES[@]}"; do
    echo "Cross-validation fold: validation=${validation_sequence}, test=Mavic3"
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" train.py \
        --model "$MODEL" \
        --data-root /home/lab/feng_home/data/MMAUD \
        --val "$validation_sequence" \
        --test Mavic3 \
        --results-csv "$RESULTS_CSV" \
        --freeze-pretrained-epochs 5 \
        --learning-rate 5e-8 \
        --weight-decay 0.1 \
        --batch-size 16 \
        --epochs 100 \
        "$@"
done


echo "All folds complete. Results: ${RESULTS_CSV}"
