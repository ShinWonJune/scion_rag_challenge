#!/usr/bin/env bash
# Exp4 single cell runner. Args: <cell_tag> <llm_backend> [extra args]
# Examples:
#   scripts/run_exp4_cell.sh gpt_oss_20b vllm
#   scripts/run_exp4_cell.sh qwen3_8b vllm
#   scripts/run_exp4_cell.sh gpt_5_4 chatgpt
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source scripts/run_experiment_setup.sh > /dev/null

CELL="${1:?cell_tag required}"
LLM="${2:?llm backend required (vllm|chatgpt)}"

if [ -z "${SHRAG_BEST_ENCODER:-}" ]; then
  echo "ERROR: SHRAG_BEST_ENCODER not set" >&2
  exit 1
fi
if [ -z "${SHRAG_EXP1_FROZEN:-}" ]; then
  echo "ERROR: SHRAG_EXP1_FROZEN not set" >&2
  exit 1
fi

OUT="outputs/e2e_${CELL}"
EVAL_OUT="experiments/outputs/eval/${CELL}"

case "$CELL" in
  gpt_oss_20b)
    LLM_ARGS="--llm vllm --vllm-url ${GPT_OSS_VLLM_URL} --vllm-model ${GPT_OSS_VLLM_MODEL}"
    GEN_MODEL="${GPT_OSS_VLLM_MODEL}"
    ;;
  qwen3_8b)
    LLM_ARGS="--llm vllm --vllm-url ${QWEN3_8B_VLLM_URL} --vllm-model ${QWEN3_8B_VLLM_MODEL}"
    GEN_MODEL="${QWEN3_8B_VLLM_MODEL}"
    ;;
  gpt_5_4)
    LLM_ARGS="--llm chatgpt --llm-model gpt-5.4 --openai-reasoning-effort medium --max-answer-tokens 4000"
    GEN_MODEL="gpt-5.4"
    ;;
  *)
    echo "ERROR: unknown cell $CELL" >&2
    exit 1
    ;;
esac

echo "=== Exp4 cell=$CELL llm=$LLM_ARGS at $(date) ==="

python -m shrag.pipeline.run \
  --questions data/test.csv \
  --encoder "${SHRAG_BEST_ENCODER}" \
  ${LLM_ARGS} \
  --extractor chatgpt \
  --extractor-model gpt-4.1-mini \
  --extractor-temperature 0 \
  --frozen-queries "${SHRAG_EXP1_FROZEN}" \
  --scienceon-max-concurrency "${SHRAG_MAX_CONCURRENCY:-1}" \
  --top-k 50 \
  --max-rank 5 \
  --output "${OUT}"

echo "=== Judge cell=$CELL at $(date) ==="

python -m experiments.eval_gold_judge.run_gold_judge \
  --final-dir "${OUT}/final" \
  --judge-backend openai \
  --judge-model gpt-5.4 \
  --reasoning-effort medium \
  --max-tokens 2000 \
  --generation-model "${GEN_MODEL}" \
  --retrieved-max-docs 5 \
  --output-dir "${EVAL_OUT}"

echo "=== DONE cell=$CELL at $(date) ==="
