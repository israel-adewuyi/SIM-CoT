#!/usr/bin/env bash
set -euo pipefail

SAVE_DIR="${SAVE_DIR:-./outputs}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
HF_DATASET="${HF_DATASET:-SWE-Swiss/SWESwiss-SFT-Repair-4K}"
HF_SPLIT="${HF_SPLIT:-train}"
DECODER_PATH="${DECODER_PATH:-}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
MAX_TOKEN_NUM="${MAX_TOKEN_NUM:-4096}"
NUM_LATENT="${NUM_LATENT:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
export CODI_REQUIRE_CUDA="${CODI_REQUIRE_CUDA:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${SAVE_DIR}"

EXTRA_ARGS=()
if [[ -n "${DECODER_PATH}" ]]; then
  EXTRA_ARGS+=(--decoder_path "${DECODER_PATH}")
fi

python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. Refusing to run training on CPU.")
    sys.exit(1)

print(f"[precheck] CUDA OK: {torch.cuda.device_count()} GPU(s) visible.")
for i in range(torch.cuda.device_count()):
    print(f"[precheck] GPU {i}: {torch.cuda.get_device_name(i)}")
PY

NPROC_PER_NODE="$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" train.py)
  EXTRA_ARGS+=(--ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}")
else
  LAUNCHER=(python3 train.py)
fi

"${LAUNCHER[@]}" \
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
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --bf16 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --max_grad_norm 1.0 \
  --use_lora True \
  --lora_r 64 --lora_alpha 16 --lora_init \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --do_train \
  --report_to tensorboard \
  --num_latent "${NUM_LATENT}" \
  --logging_strategy "steps" \
  --use_prj True \
  --prj_dim 2560 \
  --prj_dropout 0.0 \
  --distill_loss_div_std True \
  --remove_eos False \
  --distill_loss_factor 10 \
  --ref_loss_factor 1.0 \
  --max_token_num "${MAX_TOKEN_NUM}" \
  --use_decoder True \
  "${EXTRA_ARGS[@]}"
