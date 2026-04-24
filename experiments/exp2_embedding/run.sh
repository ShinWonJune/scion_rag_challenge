#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)}"
OUT_ROOT="experiments/outputs/embedding_benchmark/${RUN_ID}"
mkdir -p "${OUT_ROOT}"

QUERIES="${OUT_ROOT}/queries.jsonl"
python -m pipeline.step1_search \
  --questions data/test.csv \
  --extractor vllm \
  --emit-frozen-queries "${QUERIES}"

CORPUS="${OUT_ROOT}/corpus.jsonl"
python -m pipeline.step1_search \
  --questions data/test.csv \
  --sources scienceon \
  --extractor vllm \
  --frozen-queries "${QUERIES}" \
  --output-dir "${OUT_ROOT}/search"

SEARCH_DOCS=$(find "${OUT_ROOT}/search" -name search_documents.jsonl | head -n 1)
cp "${SEARCH_DOCS}" "${CORPUS}"

python -m experiments.shared.evaluate.embed_benchmark \
  --cases experiments/shared/cases/exp2_embed.yaml \
  --corpus "${CORPUS}" \
  --queries "${QUERIES}" \
  --gold data/gold/scienceon_gold.json \
  --output "${OUT_ROOT}"

echo "Done: ${OUT_ROOT}"
