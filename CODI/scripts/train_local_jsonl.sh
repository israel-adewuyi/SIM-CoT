#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=/absolute/path/to/your/train.jsonl
EXPT_NAME=local_jsonl_train
RUN_NAME=local_jsonl_seed11

python train.py \
    --expt_name "$EXPT_NAME" \
    --run_name "$RUN_NAME" \
    --logging_steps 10 \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --data_name local-jsonl \
    --data_path "$DATA_PATH" \
    --seed 11 \
    --model_max_length 512 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --num_train_epochs 3 \
    --learning_rate 8e-4 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to tensorboard \
    --num_latent 6 \
    --logging_strategy "steps" \
    --use_prj True \
    --prj_dim 2048 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --exp_mode False \
    --exp_data_num 200 \
    --remove_eos True \
    --distill_loss_factor 20 \
    --print_ref_model_stats True
