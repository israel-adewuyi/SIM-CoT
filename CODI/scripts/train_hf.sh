#!/usr/bin/env bash
set -euo pipefail

HF_DATASET=israel-adewuyi/kwaiklear-sample-level-agent-trajectories-350K
RUN_NAME="run1_qwen4B_nlatent-3_lr-1e3_lora-r-128_4k"
CACHE_KEY="350K_4K"

python train.py \
    --run_name "$RUN_NAME" \
    --model_name_or_path Qwen/Qwen3-4B \
    --data_name hf \
    --dataset_cache_key "$CACHE_KEY" \
    --hf_dataset_name "$HF_DATASET" \
    --seed 11 \
    --output_dir outputs \
    --model_max_length 8192 \
    --max_token_num 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --bf16 \
    --num_train_epochs 2 \
    --learning_rate 1e-3 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 32 --lora_alpha 32 --lora_init \
    --save_strategy "epoch" \
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
    --num_latent 2 \
