#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS_PATH=/absolute/path/to/your/predictions.jsonl
OUTPUT_DIR=""  # optional; leave empty to use auto path
SCORING_MODEL="Qwen/Qwen3-Embedding-0.6B"  # optional override
BATCH_SIZE=64
SCORING_MAX_LENGTH=512
LOG_LEVEL=INFO

cmd=(
    python score.py
    --predictions "$PREDICTIONS_PATH"
    --batch_size "$BATCH_SIZE"
    --scoring_max_length "$SCORING_MAX_LENGTH"
    --log_level "$LOG_LEVEL"
)

if [[ -n "$SCORING_MODEL" ]]; then
    cmd+=(--scoring_model "$SCORING_MODEL")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    cmd+=(--output_dir "$OUTPUT_DIR")
fi

"${cmd[@]}"
