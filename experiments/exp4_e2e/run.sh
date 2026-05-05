#!/usr/bin/env bash
set -euo pipefail

BEST_ENCODER="${SHRAG_BEST_ENCODER:?Set SHRAG_BEST_ENCODER to the selected Exp2 encoder config.}"
FROZEN_QUERIES="${SHRAG_EXP1_FROZEN:?Set SHRAG_EXP1_FROZEN to the Exp1 frozen queries JSONL path.}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4}"
JUDGE_EFFORT="${JUDGE_EFFORT:-medium}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2000}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-4000}"

if [[ ! -f "${FROZEN_QUERIES}" ]]; then
  echo "Frozen query file not found: ${FROZEN_QUERIES}" >&2
  exit 1
fi

if [[ ! -f "${BEST_ENCODER}" ]]; then
  echo "Encoder config not found: ${BEST_ENCODER}" >&2
  exit 1
fi

run_pipeline_cell() {
  local run_tag="$1"
  local generation_model="$2"
  shift 2

  local pipeline_dir="outputs/e2e_${run_tag}"
  local judge_dir="experiments/outputs/eval/${run_tag}"

  python -m shrag.pipeline.run \
    --questions data/test.csv \
    --encoder "${BEST_ENCODER}" \
    --sources scienceon \
    --extractor chatgpt \
    --extractor-model gpt-4.1-mini \
    --extractor-temperature 0 \
    --frozen-queries "${FROZEN_QUERIES}" \
    --scienceon-max-concurrency "${SHRAG_MAX_CONCURRENCY:-2}" \
    --target-documents 50 \
    --top-k 50 \
    --max-rank 5 \
    --max-answer-tokens "${MAX_ANSWER_TOKENS}" \
    --output "${pipeline_dir}" \
    "$@"

  python -m experiments.eval_gold_judge.run_gold_judge \
    --final-dir "${pipeline_dir}/final" \
    --judge-backend openai \
    --judge-model "${JUDGE_MODEL}" \
    --reasoning-effort "${JUDGE_EFFORT}" \
    --max-tokens "${JUDGE_MAX_TOKENS}" \
    --generation-model "${generation_model}" \
    --retrieved-max-docs 5 \
    --output-dir "${judge_dir}"
}

run_pipeline_cell \
  "gpt_oss_20b" \
  "${GPT_OSS_VLLM_MODEL:-openai/gpt-oss-20b}" \
  --llm vllm \
  --vllm-url "${GPT_OSS_VLLM_URL:-http://10.38.38.40:8004/v1}" \
  --vllm-model "${GPT_OSS_VLLM_MODEL:-openai/gpt-oss-20b}"

run_pipeline_cell \
  "qwen3_8b" \
  "${QWEN3_8B_VLLM_MODEL:-Qwen/Qwen3-8B}" \
  --llm vllm \
  --vllm-url "${QWEN3_8B_VLLM_URL:-http://10.38.38.40:8005/v1}" \
  --vllm-model "${QWEN3_8B_VLLM_MODEL:-Qwen/Qwen3-8B}"

run_pipeline_cell \
  "gpt_5_4" \
  "gpt-5.4" \
  --llm chatgpt \
  --llm-model gpt-5.4 \
  --openai-reasoning-effort medium

echo "Done: outputs/e2e_<cell> and experiments/outputs/eval/<cell>"
