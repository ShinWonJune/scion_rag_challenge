#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)}"
OUT_ROOT="experiments/outputs/e2e/${RUN_ID}"
mkdir -p "${OUT_ROOT}"

python -m pipeline.run_pipeline \
  --questions data/test.csv \
  --encoder configs/query_encoder/config_gte-multilingual-base.json \
  --llm vllm \
  --extractor vllm \
  --sources scienceon \
  --vllm-url "${VLLM_URL:-http://localhost:8000/v1}" \
  --vllm-model "${VLLM_MODEL:-openai/gpt-oss-20b}" \
  --output "${OUT_ROOT}/pipeline"

python -m experiments.shared.evaluate.judge \
  --predictions "${OUT_ROOT}/pipeline/final/predictions.json" \
  --retrieval_dir "${OUT_ROOT}/pipeline/retrieval" \
  --judge_backend gemini \
  --judge_model gemini-2.5-flash \
  --output "${OUT_ROOT}/judge/judge_scores.jsonl"

echo "Done: ${OUT_ROOT}"
