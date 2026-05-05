#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)}"
OUT_ROOT="experiments/outputs/concurrency_sweep/${RUN_ID}"
mkdir -p "${OUT_ROOT}"
rm -rf outputs/_shared_cache/scienceon

for MAX_C in 1 2 3 5 8; do
  python -m shrag.pipeline.steps.step1_search \
    --questions data/test.csv \
    --sources scienceon \
    --extractor chatgpt \
    --extractor-model gpt-4.1-mini \
    --extractor-temperature 0 \
    --scienceon-max-concurrency "${MAX_C}" \
    --scienceon-fixed-concurrency \
    --output-dir "${OUT_ROOT}/c${MAX_C}"
done

python -m experiments.shared.evaluate.concurrency_report \
  --input "${OUT_ROOT}" \
  --epsilon 0.01 \
  --output "${OUT_ROOT}/report.md"

echo "Done: ${OUT_ROOT}"
