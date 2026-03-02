#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_PATH=/home/user/israel/SIM-CoT/CODI/data/v2_eval.jsonl
OUTPUT_DIR=/home/user/israel/SIM-CoT/CODI/runs/qwen_4B_vanilla/eval

python "$CODI_DIR/test_vanilla.py" \
    --data_name "local-jsonl" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path Qwen/Qwen3-4B \
    --seed 11 \
    --model_max_length 16384 \
    --max_new_tokens 1024 \
    --bf16 \
    --batch_size 2 \
    --greedy True \
    --temperature 0.1 \
    --top_k 40 \
    --top_p 0.95 \
    --inf_num_iterations 1
