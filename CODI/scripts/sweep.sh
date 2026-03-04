#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=/home/user/israel/SIM-CoT/CODI/data/v2_train.jsonl
OUTPUT_DIR=outputs
SEED=11

BASE_RUN_NAME=qwen4B_nlatent-3

# grid search
LRS=(1e-3 5e-3 1e-4 5e-4 1e-5 5e-5)
LORA_RS=(8 16 32)

for LR in "${LRS[@]}"; do
  for R in "${LORA_RS[@]}"; do
    RUN_NAME="${BASE_RUN_NAME}_lr-${LR}_lora-r-${R}"

    python train.py \
      --run_name "$RUN_NAME" \
      --model_name_or_path Qwen/Qwen3-4B \
      --data_name local-jsonl \
      --data_path "$DATA_PATH" \
      --seed "$SEED" \
      --output_dir "$OUTPUT_DIR" \
      --model_max_length 16384 \
      --max_token_num 16384 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --bf16 \
      --num_train_epochs 2 \
      --learning_rate "$LR" \
      --max_grad_norm 2.0 \
      --use_lora True \
      --lora_r "$R" \
      --lora_alpha 32 \
      --lora_init \
      --save_strategy "no" \
      --weight_decay 0.1 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --do_train \
      --report_to none \
      --logging_strategy "steps" \
      --logging_steps 1 \
      --logging_first_step True \
      --use_prj True \
      --prj_dim 2048 \
      --prj_dropout 0.0 \
      --distill_loss_div_std True \
      --exp_mode False \
      --exp_data_num 200 \
      --remove_eos True \
      --distill_loss_factor 20 \
      --print_ref_model_stats False \
      --use_decoder True \
      --decoder_path Qwen/Qwen3-1.7B \
      --num_latent 3
  done
done