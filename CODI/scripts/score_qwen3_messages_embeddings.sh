#!/usr/bin/env bash
set -euo pipefail

GENERATIONS="${GENERATIONS:-./outputs/eval_qwen3_messages/messages_generations.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-data/context_target_v1_eval_messages.jsonl}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
MESSAGES_FIELD="${MESSAGES_FIELD:-messages}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_qwen3_messages/embedding_scores}"

SCORING_MODEL_ID="${SCORING_MODEL_ID:-sentence-transformers/all-mpnet-base-v2}"
SCORING_MODEL_REVISION="${SCORING_MODEL_REVISION:-}"
SCORING_POOLING="${SCORING_POOLING:-mean}"
SCORING_NORMALIZE="${SCORING_NORMALIZE:-true}"
SCORING_MAX_LENGTH="${SCORING_MAX_LENGTH:-512}"
SCORING_DTYPE="${SCORING_DTYPE:-float16}"
SCORING_DEVICE="${SCORING_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SIMILARITY_METRIC="${SIMILARITY_METRIC:-cosine}"
THRESHOLD="${THRESHOLD:-}"
DEDUPE_POLICY="${DEDUPE_POLICY:-last}"
SCORING_VERSION="${SCORING_VERSION:-embedding_harness_v1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

read -r -a generation_paths <<< "${GENERATIONS}"
if [[ "${#generation_paths[@]}" -eq 0 ]]; then
  echo "GENERATIONS is empty. Provide at least one generation JSONL path." >&2
  exit 1
fi

cmd=(
  uv run python eval_harness/score_generations_with_embeddings.py
  --generations "${generation_paths[@]}"
  --dataset_name "${EVAL_DATASET}"
  --dataset_split "${EVAL_SPLIT}"
  --messages_field "${MESSAGES_FIELD}"
  --output_dir "${OUTPUT_DIR}"
  --scoring_model_id "${SCORING_MODEL_ID}"
  --scoring_pooling "${SCORING_POOLING}"
  --scoring_normalize "${SCORING_NORMALIZE}"
  --scoring_max_length "${SCORING_MAX_LENGTH}"
  --scoring_dtype "${SCORING_DTYPE}"
  --scoring_device "${SCORING_DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --similarity_metric "${SIMILARITY_METRIC}"
  --dedupe_policy "${DEDUPE_POLICY}"
  --scoring_version "${SCORING_VERSION}"
  --log_level "${LOG_LEVEL}"
)

if [[ -n "${SCORING_MODEL_REVISION}" ]]; then
  cmd+=(--scoring_model_revision "${SCORING_MODEL_REVISION}")
fi

if [[ -n "${THRESHOLD}" ]]; then
  cmd+=(--threshold "${THRESHOLD}")
fi

"${cmd[@]}"
