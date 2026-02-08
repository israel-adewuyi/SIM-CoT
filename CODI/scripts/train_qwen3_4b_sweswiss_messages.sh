#!/usr/bin/env bash
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-./outputs}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
HF_DATASET="${HF_DATASET:-SWE-Swiss/SWESwiss-SFT-Repair-4K}"
HF_SPLIT="${HF_SPLIT:-train}"
DECODER_PATH="${DECODER_PATH:-}"

mkdir -p "${SAVE_DIR}"

EXTRA_ARGS=()
if [[ -n "${DECODER_PATH}" ]]; then
  EXTRA_ARGS+=(--decoder_path "${DECODER_PATH}")
fi

python3 train.py \
  --output_dir "${SAVE_DIR}" \
  --expt_name qwen3-4b-sweswiss-messages \
  --logging_dir "${SAVE_DIR}/logs" \
  --logging_steps 10 \
  --model_name_or_path "${MODEL_NAME}" \
  --data_name sweswiss \
  --hf_dataset_name "${HF_DATASET}" \
  --hf_dataset_split "${HF_SPLIT}" \
  --messages_field messages \
  --seed 11 \
  --model_max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --max_grad_norm 1.0 \
  --use_lora True \
  --lora_r 64 --lora_alpha 16 --lora_init \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --save_safetensors False \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --do_train \
  --report_to tensorboard \
  --num_latent 4 \
  --logging_strategy "steps" \
  --use_prj True \
  --prj_dim 2560 \
  --prj_dropout 0.0 \
  --distill_loss_div_std True \
  --remove_eos False \
  --distill_loss_factor 10 \
  --ref_loss_factor 1.0 \
  --max_token_num 4096 \
  --use_decoder True \
  "${EXTRA_ARGS[@]}"
