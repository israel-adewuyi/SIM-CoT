#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${EXP_ID:-model_qwen4B_max_len_4096-lora_r_2}"
if [[ -z "${EXP_ID}" ]]; then
  echo "Set EXP_ID (lowercase: [A-Za-z0-9._-]) to identify this experiment." >&2
  exit 1
fi
if [[ ! "${EXP_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid EXP_ID='${EXP_ID}'. Use lowercase [A-Za-z0-9._-] only." >&2
  exit 1
fi

EVAL_TAG="${EVAL_TAG:-$(date +%Y%m%d_%H%M%S)}"
GENERATIONS="${GENERATIONS:-}"
EVAL_DATASET="${EVAL_DATASET:-data/context_target_v1_eval_messages.jsonl}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
MESSAGES_FIELD="${MESSAGES_FIELD:-messages}"

SCORING_MODEL_ID="${SCORING_MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"
SCORING_MODEL_REVISION="${SCORING_MODEL_REVISION:-}"
SCORER_TAG="${SCORER_TAG:-qwen3_emb_0.6B}"
SCORING_POOLING="${SCORING_POOLING:-mean}"
SCORING_NORMALIZE="${SCORING_NORMALIZE:-true}"
SCORING_MAX_LENGTH="${SCORING_MAX_LENGTH:-4096}"
SCORING_DTYPE="${SCORING_DTYPE:-float16}"
SCORING_DEVICE="${SCORING_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SIMILARITY_METRIC="${SIMILARITY_METRIC:-cosine}"
THRESHOLD="${THRESHOLD:-}"
DEDUPE_POLICY="${DEDUPE_POLICY:-last}"
SCORING_VERSION="${SCORING_VERSION:-embedding_harness_v1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "${SCORER_TAG}" ]]; then
  SCORER_TAG="${SCORING_MODEL_ID//\//__}"
  if [[ -n "${SCORING_MODEL_REVISION}" ]]; then
    SCORER_TAG="${SCORER_TAG}__rev_${SCORING_MODEL_REVISION}"
  fi
fi
SCORER_TAG="$(printf '%s' "${SCORER_TAG}" | tr -cs '[:alnum:]._-' '_')"

if [[ -z "${GENERATIONS}" ]]; then
  GENERATIONS="./outputs/eval/${EXP_ID}/messages_generations.jsonl"
fi

if [[ ! "${EVAL_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid EVAL_TAG='${EVAL_TAG}'. Use [A-Za-z0-9._-] only." >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="./outputs/eval/${EXP_ID}/${SCORER_TAG}"
fi

read -r -a generation_paths <<< "${GENERATIONS}"
if [[ "${#generation_paths[@]}" -eq 0 ]]; then
  echo "GENERATIONS is empty. Provide at least one generation JSONL path." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

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
