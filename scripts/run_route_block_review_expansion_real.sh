#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/AutoPlanner}"
cd "$ROOT"

normalize_deepseek_key_value() {
  local value="${1:-}"
  value="$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

normalize_deepseek_env_key() {
  if [[ -n "${DEEPSEEK_API_KEY+x}" ]]; then
    DEEPSEEK_API_KEY="$(normalize_deepseek_key_value "$DEEPSEEK_API_KEY")"
    export DEEPSEEK_API_KEY
  fi
}

load_env_key() {
  local env_file="$1"
  if [[ -n "${DEEPSEEK_API_KEY:-}" || ! -f "$env_file" ]]; then
    return 0
  fi
  local line
  line="$(grep -E '^DEEPSEEK_API_KEY=' "$env_file" | tail -n 1 || true)"
  if [[ -n "$line" ]]; then
    DEEPSEEK_API_KEY="$(normalize_deepseek_key_value "${line#DEEPSEEK_API_KEY=}")"
    export DEEPSEEK_API_KEY
  fi
}

normalize_deepseek_env_key
load_env_key ".env.local"
load_env_key ".env"
normalize_deepseek_env_key

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set. Export it or place it in .env.local or .env before running real review." >&2
  exit 2
fi
if [[ "$DEEPSEEK_API_KEY" == "replace_with_your_deepseek_key" ]]; then
  echo "DEEPSEEK_API_KEY still contains the dotenv placeholder; replace it before running real review." >&2
  exit 2
fi

WORK_ROOT="${WORK_ROOT:-results/shared/model_strengthening_20260519_review_expansion}"
OUT_DIR="${OUT_DIR:-${WORK_ROOT}/real_review_pipeline}"
REVIEW_JSONL="${REVIEW_JSONL:-${WORK_ROOT}/route_pool_evidence_review_expansion.jsonl}"
TRANSFORM_SANITY_JSON="${TRANSFORM_SANITY_JSON-${WORK_ROOT}/route_pool_evidence_review_expansion_transform_sanity.json}"
VALUE_PACK="${VALUE_PACK:-}"
WORKERS="${WORKERS:-4}"
MODEL="${DEEPSEEK_MODEL:-}"
PREFIX="${REVIEW_PREFIX:-expansion_real}"
CACHE="${CACHE:-${OUT_DIR}/${PREFIX}_review_cache.jsonl}"
export AUTOPLANNRELLM_DEEPSEEK_TIMEOUT_S="${AUTOPLANNRELLM_DEEPSEEK_TIMEOUT_S:-300}"

mkdir -p "$OUT_DIR"

MODEL_ARGS=()
if [[ -n "$MODEL" ]]; then
  MODEL_ARGS=(--model "$MODEL")
fi
CACHE_ARGS=()
if [[ -n "$CACHE" ]]; then
  CACHE_ARGS=(--cache "$CACHE")
fi
TRANSFORM_ARGS=()
if [[ -n "$TRANSFORM_SANITY_JSON" ]]; then
  TRANSFORM_ARGS=(--transform-sanity-json "$TRANSFORM_SANITY_JSON")
fi

PYTHONPATH=. python -m cascade_planner.eval.run_route_pool_evidence_review_pipeline \
  --review-jsonl "$REVIEW_JSONL" \
  "${TRANSFORM_ARGS[@]}" \
  --prompts-jsonl "${OUT_DIR}/route_pool_evidence_review_prompts.jsonl" \
  --output-dir "$OUT_DIR" \
  --prefix "$PREFIX" \
  --workers "$WORKERS" \
  --resume \
  --no-dry-run \
  "${CACHE_ARGS[@]}" \
  "${MODEL_ARGS[@]}"

PYTHONPATH=. python -m cascade_planner.eval.build_route_block_review_label_pack \
  --input "${OUT_DIR}/${PREFIX}_labels.jsonl" \
  --output-jsonl "${OUT_DIR}/${PREFIX}_review_label_pack.jsonl" \
  --report "${OUT_DIR}/${PREFIX}_review_label_pack_report.json" \
  --dataset "${PREFIX}_review_labels"

python -m json.tool "${OUT_DIR}/${PREFIX}_promotion_gate.json" >/dev/null
python -m json.tool "${OUT_DIR}/${PREFIX}_review_label_pack_report.json" >/dev/null

if [[ -n "$VALUE_PACK" ]]; then
  PYTHONPATH=. python -m cascade_planner.eval.merge_route_block_review_labels \
    --value-pack "$VALUE_PACK" \
    --review-label-pack "${OUT_DIR}/${PREFIX}_review_label_pack.jsonl" \
    --output-jsonl "${OUT_DIR}/${PREFIX}_merged_route_block_value_pack.jsonl" \
    --report "${OUT_DIR}/${PREFIX}_merged_route_block_value_pack_report.json" \
    --dataset "${PREFIX}_merged_route_block_value"
  python -m json.tool "${OUT_DIR}/${PREFIX}_merged_route_block_value_pack_report.json" >/dev/null
fi

echo "Real review pipeline complete:"
echo "  ${OUT_DIR}/${PREFIX}_pipeline_manifest.json"
echo "  ${OUT_DIR}/${PREFIX}_review_label_pack_report.json"
if [[ -n "$VALUE_PACK" ]]; then
  echo "  ${OUT_DIR}/${PREFIX}_merged_route_block_value_pack_report.json"
fi
