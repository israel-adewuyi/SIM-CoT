#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
EVAL_DATASET="${EVAL_DATASET:-data/context_target_v1_eval_messages.jsonl}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
MESSAGES_FIELD="${MESSAGES_FIELD:-messages}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_qwen3_messages}"
CKPT_DIR="${CKPT_DIR:-./outputs/checkpoints/train_test}"
MAX_TOKEN_NUM="${MAX_TOKEN_NUM:-2048}"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "Set CKPT_DIR to your checkpoint directory before running." >&2
  exit 1
fi

uv run python test.py \
  --data_name messages \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME}" \
  --hf_dataset_name "${EVAL_DATASET}" \
  --hf_dataset_split "${EVAL_SPLIT}" \
  --messages_field "${MESSAGES_FIELD}" \
  --seed 11 \
  --model_max_length 1024 \
  --max_token_num "${MAX_TOKEN_NUM}" \
  --bf16 \
  --lora_r 1 --lora_alpha 16 --lora_init \
  --batch_size 64 \
  --greedy True \
  --num_latent 4 \
  --use_prj True \
  --prj_dim 2560 \
  --prj_no_ln False \
  --prj_dropout 0.0 \
  --inf_latent_iterations 4 \
  --inf_num_iterations 1 \
  --remove_eos False \
  --use_lora True \
  --ckpt_dir "${CKPT_DIR}"
