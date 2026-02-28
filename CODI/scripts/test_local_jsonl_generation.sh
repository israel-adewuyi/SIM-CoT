#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=/absolute/path/to/your/eval.jsonl
CKPT_DIR=/absolute/path/to/your/checkpoints
OUTPUT_DIR=./eval_outputs/local_jsonl

python test.py \
    --data_name "local-jsonl" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --seed 11 \
    --model_max_length 512 \
    --max_new_tokens 256 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 64 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 2048 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir "$CKPT_DIR"
