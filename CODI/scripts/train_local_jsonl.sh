#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=/home/user/israel/SIM-CoT/CODI/data/v2_train.jsonl
EXPT_NAME=qwen_4B
RUN_NAME=lora-16_nlatent-3_lr-8e4

python train.py \
    --expt_name "$EXPT_NAME" \
    --run_name "$RUN_NAME" \
    --model_name_or_path Qwen/Qwen3-4B \
    --data_name local-jsonl \
    --data_path "$DATA_PATH" \
    --seed 11 \
    --output_dir outputs \
    --model_max_length 16384 \
    --max_token_num 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --num_train_epochs 4 \
    --learning_rate 8e-4 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 16 --lora_alpha 32 --lora_init \
    --save_strategy "epoch" \
    --save_total_limit 1 \
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
    --print_ref_model_stats True \
    --use_decoder True \
    --decoder_path Qwen/Qwen3-1.7B \
    --num_latent 3 \
