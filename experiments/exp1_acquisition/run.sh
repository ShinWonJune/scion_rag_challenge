#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)}"
OUT_ROOT="experiments/outputs/acquisition/${RUN_ID}"
mkdir -p "${OUT_ROOT}"

for EXTRACTOR in gemini vllm; do
  for RETRY in off on; do
    EXTRA_ARGS=()
    if [ "${RETRY}" = "on" ]; then
      EXTRA_ARGS+=(--scienceon-max-retries 5)
    else
      EXTRA_ARGS+=(--scienceon-max-retries 0)
    fi
    python -m pipeline.step1_search \
      --questions data/test.csv \
      --sources scienceon \
      --extractor "${EXTRACTOR}" \
      --output-dir "${OUT_ROOT}/${EXTRACTOR}_retry_${RETRY}" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "Done: ${OUT_ROOT}"
