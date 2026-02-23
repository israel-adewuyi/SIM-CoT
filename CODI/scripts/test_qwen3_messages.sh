#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${EXP_ID:-model_qwen4B_max_len_8192-lora_r_16}"
if [[ -z "${EXP_ID}" ]]; then
  echo "Set EXP_ID (lowercase: [A-Za-z0-9._-]) to identify this experiment." >&2
  exit 1
fi
if [[ ! "${EXP_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid EXP_ID='${EXP_ID}'. Use lowercase [A-Za-z0-9._-] only." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
EVAL_DATASET="${EVAL_DATASET:-data/context_target_v1_eval_messages.jsonl}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
MESSAGES_FIELD="${MESSAGES_FIELD:-messages}"
EVAL_TAG="${EVAL_TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${EVAL_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid EVAL_TAG='${EVAL_TAG}'. Use [A-Za-z0-9._-] only." >&2
  exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval/${EXP_ID}_32K}"
CKPT_DIR="${CKPT_DIR:-./outputs/experiments/${EXP_ID}/train/checkpoints}"
MAX_TOKEN_NUM="${MAX_TOKEN_NUM:-32768}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "Set CKPT_DIR to your checkpoint directory before running." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

uv run python test.py \
  --data_name messages \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME}" \
  --hf_dataset_name "${EVAL_DATASET}" \
  --hf_dataset_split "${EVAL_SPLIT}" \
  --messages_field "${MESSAGES_FIELD}" \
  --seed 11 \
  --model_max_length 32768 \
  --max_token_num "${MAX_TOKEN_NUM}" \
  --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
  --bf16 \
  --lora_r 16 --lora_alpha 16 --lora_init \
  --batch_size 1 \
  --greedy True \
  --num_latent 4 \
  --use_prj True \
  --prj_dim 2560 \
  --prj_no_ln False \
  --prj_dropout 0.0 \
  --inf_latent_iterations 4 \
  --inf_num_iterations 1 \
  --remove_eos True \
  --use_lora True \
  --ckpt_dir "${CKPT_DIR}"
