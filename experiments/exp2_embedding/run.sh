#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_exp2}"
OUT_ROOT="experiments/outputs/embedding_benchmark/${RUN_ID}"
CORPUS="${SHRAG_EXP1_CORPUS:?Set SHRAG_EXP1_CORPUS to the Exp1 search_documents.jsonl path.}"
QUERIES="${SHRAG_EXP1_FROZEN:?Set SHRAG_EXP1_FROZEN to the Exp1 frozen queries JSONL path.}"
GOLD_JSON="${GOLD_JSON:-data/gold/scienceon_gold.json}"

if [[ ! -f "${CORPUS}" ]]; then
  echo "Corpus file not found: ${CORPUS}" >&2
  exit 1
fi

if [[ ! -f "${QUERIES}" ]]; then
  echo "Frozen query file not found: ${QUERIES}" >&2
  exit 1
fi

if [[ ! -f "${GOLD_JSON}" ]]; then
  echo "Gold JSON not found: ${GOLD_JSON}" >&2
  echo "Run EXPERIMENT_MANUAL.md section 1.5 first." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

python -m experiments.shared.evaluate.embed_benchmark \
  --cases experiments/shared/cases/exp2_embed.yaml \
  --corpus "${CORPUS}" \
  --queries "${QUERIES}" \
  --gold "${GOLD_JSON}" \
  --output "${OUT_ROOT}"

echo "Done: ${OUT_ROOT}"
