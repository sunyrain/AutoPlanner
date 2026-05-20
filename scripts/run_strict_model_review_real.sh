#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/AutoPlanner}"
cd "$ROOT"

export REVIEW_JSONL="${REVIEW_JSONL:-results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.jsonl}"
export TRANSFORM_SANITY_JSON="${TRANSFORM_SANITY_JSON-}"
export OUT_DIR="${OUT_DIR:-results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline}"
export REVIEW_PREFIX="${REVIEW_PREFIX:-strict_model_real}"
export VALUE_PACK="${VALUE_PACK:-results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl}"

PYTHONPATH=. scripts/run_route_block_review_expansion_real.sh
