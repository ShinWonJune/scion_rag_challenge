#!/usr/bin/env bash
# Exp1 single cell runner. Args: <k> <n> <m> <phase>
# k = ScienceON docs per term per query (10 × max_pages, so k=10→pages=1, k=50→pages=5)
# n = per-language top-N search_terms (slice frozen queries)
# m = target_documents per query (post-dedup cap)
# phase = phaseA | phaseB | phaseC
#
# Outputs to experiments/outputs/exp1_kmn/<phase>/k{k}_n{n}_m{m}/
set -uo pipefail
trap 'echo "[exp1_kmn] FATAL at line $LINENO (rc=$?)"' ERR

K="${1:?k required}"
N="${2:?n required}"
M="${3:?m required}"
PHASE="${4:?phase required}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load env (.env vars + conda shrag) — required for vLLM URL + creds
if [ -f scripts/run_experiment_setup.sh ]; then
  # shellcheck disable=SC1091
  source scripts/run_experiment_setup.sh > /dev/null
fi

# k = 10 * max_pages, so max_pages = k / 10
MAX_PAGES=$((K / 10))
if [ "$MAX_PAGES" -lt 1 ]; then MAX_PAGES=1; fi

CELL_TAG="k${K}_n${N}_m${M}"
OUT_DIR="experiments/outputs/exp1_kmn/${PHASE}/${CELL_TAG}"
SLICED_QUERIES="experiments/outputs/exp1_kmn/_frozen/queries_v3_n${N}.jsonl"
FROZEN_FULL="experiments/outputs/exp1_kmn/_frozen/queries_v3_chatgpt_gold41.jsonl"

mkdir -p "$OUT_DIR" "${OUT_DIR}/search" "${OUT_DIR}/retrieval"

# Slice frozen queries to first N search_terms per language (idempotent)
if [ ! -s "$SLICED_QUERIES" ]; then
  python -m experiments.exp1_kmn.slice_frozen_queries \
    --input "$FROZEN_FULL" \
    --output "$SLICED_QUERIES" \
    --n "$N"
fi

START_TS=$(date +%s)
echo "[exp1_kmn] cell=${CELL_TAG} phase=${PHASE} k=${K} n=${N} m=${M} max_pages=${MAX_PAGES} at $(date)"

# Step 1 idempotency: skip if a non-empty search_documents.jsonl already exists
EXISTING_DOCS=$(find "${OUT_DIR}/search" -name "search_documents.jsonl" -size +0 2>/dev/null | head -1)
if [ -n "${EXISTING_DOCS}" ]; then
  echo "[exp1_kmn] step1 SKIP — using existing ${EXISTING_DOCS}"
  STEP1_END=${START_TS}
else
# Step 1: search with sliced queries (warm cache after first cell)
python -m shrag.pipeline.steps.step1_search \
  --questions experiments/outputs/exp1_kmn/questions_gold41.jsonl \
  --sources scienceon \
  --extractor vllm \
  --vllm-url "${GPT_OSS_VLLM_URL:-http://10.38.38.40:8004/v1}" \
  --vllm-model "${GPT_OSS_VLLM_MODEL:-openai/gpt-oss-20b}" \
  --extractor-temperature 0 \
  --frozen-queries "$SLICED_QUERIES" \
  --target-documents "$M" \
  --scienceon-max-pages "$MAX_PAGES" \
  --scienceon-max-concurrency "${SHRAG_MAX_CONCURRENCY:-2}" \
  --scienceon-min-interval-sec 0.5 \
  --output-dir "${OUT_DIR}/search" \
  > "${OUT_DIR}/step1.log" 2>&1
STEP1_END=$(date +%s)
echo "[exp1_kmn] step1 done: $((STEP1_END - START_TS))s"
fi  # end step1 idempotency

# Step 3: build vector DB with gte (default for Exp1)
# step1 creates a timestamp subdir under search/ — find the actual jsonl
SEARCH_DOCS=$(find "${OUT_DIR}/search" -name "search_documents.jsonl" -size +0 2>/dev/null | head -1)
if [ -z "${SEARCH_DOCS}" ]; then
  echo "[exp1_kmn] FATAL: no search_documents.jsonl under ${OUT_DIR}/search"
  exit 2
fi
echo "[exp1_kmn] using search_docs: ${SEARCH_DOCS}"
ENCODER_CFG="configs/query_encoder/config_gte-multilingual-base.json"
# Each cell needs a fresh vectordb path so cells don't overwrite each other
VECTORDB_OUT="${OUT_DIR}/vectordb.csv"

# Patch encoder config in-place per cell (jq-free)
python -c "
import json
cfg = json.load(open('${ENCODER_CFG}'))
cfg['jsonl_path'] = '$(realpath ${SEARCH_DOCS})'
cfg['output_file'] = '$(realpath -m ${VECTORDB_OUT})'
cfg['output_dir'] = '$(realpath -m ${OUT_DIR})'
out = '${OUT_DIR}/encoder_cfg.json'
json.dump(cfg, open(out, 'w'), ensure_ascii=False, indent=2)
print(out)
"
CELL_ENCODER="${OUT_DIR}/encoder_cfg.json"

# Step3 idempotency: skip if vector_db_*.csv already exists
EXISTING_VECTORDB=$(find "${OUT_DIR}" -name "vector_db_*.csv" -size +0 2>/dev/null | head -1)
if [ -n "${EXISTING_VECTORDB}" ]; then
  echo "[exp1_kmn] step3 SKIP — using existing ${EXISTING_VECTORDB}"
  STEP3_END=${STEP1_END}
else
python -m shrag.pipeline.steps.step3_build_vectordb \
  --encoder "${CELL_ENCODER}" \
  --docs "${SEARCH_DOCS}" \
  --schema configs/csv_schema/test_2.json \
  > "${OUT_DIR}/step3.log" 2>&1
STEP3_END=$(date +%s)
echo "[exp1_kmn] step3 done: $((STEP3_END - STEP1_END))s"
fi  # end step3 idempotency

# Find the actual vectordb CSV that step3 wrote (it adds a timestamp subdir + filename)
ACTUAL_VECTORDB=$(find "${OUT_DIR}" -name "vector_db_*.csv" -size +0 2>/dev/null | head -1)
if [ -z "${ACTUAL_VECTORDB}" ]; then
  echo "[exp1_kmn] FATAL: no vector_db_*.csv under ${OUT_DIR}"
  exit 3
fi

# Step 4: retrieve
python -m shrag.pipeline.steps.step4_retrieve \
  --encoder "${CELL_ENCODER}" \
  --questions experiments/outputs/exp1_kmn/questions_gold41.jsonl \
  --schema configs/csv_schema/test_2.json \
  --vectordb "${ACTUAL_VECTORDB}" \
  --top-k 50 \
  --output-root "${OUT_DIR}/retrieval" \
  > "${OUT_DIR}/step4.log" 2>&1
STEP4_END=$(date +%s)
echo "[exp1_kmn] step4 done: $((STEP4_END - STEP3_END))s"

# step4 writes per-question JSONs in a timestamp subdir → combine them into one jsonl
RETRIEVAL_DIR=$(find "${OUT_DIR}/retrieval" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
RETRIEVAL_JSONL="${OUT_DIR}/retrieval.jsonl"
if [ -n "${RETRIEVAL_DIR}" ]; then
  python -m experiments.exp1_kmn.aggregate_retrieval \
    --in "${RETRIEVAL_DIR}" \
    --out "${RETRIEVAL_JSONL}" \
    > "${OUT_DIR}/aggregate.log" 2>&1 || echo "[exp1_kmn] retrieval aggregation failed for ${CELL_TAG}"
fi

if [ -s "${RETRIEVAL_JSONL}" ]; then
  python -m experiments.shared.evaluate.retrieval_eval \
    --retrieval "${RETRIEVAL_JSONL}" \
    --gold data/gold/scienceon_gold.json \
    --output "${OUT_DIR}/metrics.json" \
    > "${OUT_DIR}/eval.log" 2>&1 || echo "[exp1_kmn] eval failed for ${CELL_TAG}"
else
  echo "[exp1_kmn] WARN: no retrieval.jsonl produced for ${CELL_TAG}" > "${OUT_DIR}/eval.log"
fi

cat <<EOF > "${OUT_DIR}/timing.json"
{
  "cell": "${CELL_TAG}",
  "phase": "${PHASE}",
  "k": ${K}, "n": ${N}, "m": ${M},
  "step1_sec": $((STEP1_END - START_TS)),
  "step3_sec": $((STEP3_END - STEP1_END)),
  "step4_sec": $((STEP4_END - STEP3_END)),
  "total_sec": $((STEP4_END - START_TS))
}
EOF

echo "[exp1_kmn] DONE ${CELL_TAG} total=$((STEP4_END - START_TS))s"
