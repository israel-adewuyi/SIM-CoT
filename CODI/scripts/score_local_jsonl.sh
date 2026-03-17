#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS_PATH=/home/user/israel/SIM-CoT/CODI/runs/qwen4B_lora-16_nlatent-3_lr-8e4/eval_epoch1_t08/eval_res.jsonl
# PREDICTIONS_PATH=/home/user/israel/SIM-CoT/CODI/runs/qwen_4B_vanilla/predictions/local-jsonl_iter_0.jsonl
# OUTPUT_DIR="/home/user/israel/SIM-CoT/CODI/runs/qwen_4B_vanilla"
OUTPUT_DIR="/home/user/israel/SIM-CoT/CODI/runs/qwen4B_lora-16_nlatent-3_lr-8e4/eval_epoch1_t08"  # optional; leave empty to use auto path
SCORING_MODEL="Qwen/Qwen3-Embedding-4B"  # optional override
BATCH_SIZE=4
SCORING_MAX_LENGTH=2048
SCORED_ROWS_PATH="scored_rows.jsonl"  # relative to OUTPUT_DIR (or auto output_dir)
LOG_LEVEL=INFO

cmd=(
    python score.py
    --predictions "$PREDICTIONS_PATH"
    --batch_size "$BATCH_SIZE"
    --scoring_max_length "$SCORING_MAX_LENGTH"
    --scored_rows_path "$SCORED_ROWS_PATH"
    --log_level "$LOG_LEVEL"
)

if [[ -n "$SCORING_MODEL" ]]; then
    cmd+=(--scoring_model "$SCORING_MODEL")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    cmd+=(--output_dir "$OUTPUT_DIR")
fi

"${cmd[@]}"
