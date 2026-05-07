#!/usr/bin/env bash
# Run Phase B (m sweep) + Phase C (cross-check) using best (k, n) derived from Phase A.
# Args: <k_star> <n_star>
# m sweep: {10, 20, 30, 50, 70}; m* derived as smallest m within X% of best Phase A Hit@5.
# Phase C: (k*±10, n*±2 step) with m=m*.
set -uo pipefail

K_STAR="${1:?k_star required}"
N_STAR="${2:?n_star required}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Phase B: vary m ----
echo "[phaseB] sweep m at k*=${K_STAR} n*=${N_STAR}"
M_VALUES=(10 20 30 50 70)
for M in "${M_VALUES[@]}"; do
  CELL_DIR="experiments/outputs/exp1_kmn/phaseB/k${K_STAR}_n${N_STAR}_m${M}"
  if [ -s "${CELL_DIR}/timing.json" ]; then
    echo "[phaseB] SKIP m=${M} (done)"
    continue
  fi
  bash experiments/exp1_kmn/run_cell.sh "${K_STAR}" "${N_STAR}" "${M}" phaseB \
    || echo "[phaseB] FAIL m=${M}"
done

# Aggregate Phase B and pick m*
python -m experiments.exp1_kmn.aggregate \
  --root experiments/outputs/exp1_kmn \
  --out experiments/outputs/exp1_kmn/_aggregate.csv

# Use python to derive m* (smallest m within 1pt Hit@5 of Phase B best)
M_STAR=$(python -c "
import csv
rows=[r for r in csv.DictReader(open('experiments/outputs/exp1_kmn/_aggregate.csv')) if r['phase']=='phaseB' and r['hit_at_5']]
if not rows:
  print(30)
else:
  rows = [(int(r['m']), float(r['hit_at_5'])) for r in rows]
  best_h5 = max(h for _, h in rows)
  rows.sort(key=lambda r: r[0])
  pick = next((m for m, h in rows if h >= best_h5 - 0.025), rows[-1][0])
  print(pick)
")
echo "[phaseB] m_star=${M_STAR}"

# ---- Phase C: ±1 step grid around (k*, n*) with m=m* ----
echo "[phaseC] cross-check at k*=${K_STAR} n*=${N_STAR} m*=${M_STAR}"
# k step = 10, n step = roughly 2; bound by [10, 50] and [1, 10]
K_LIST=()
for D in -10 0 10; do
  V=$((K_STAR + D))
  if [ "$V" -ge 10 ] && [ "$V" -le 50 ]; then K_LIST+=("$V"); fi
done
N_LIST=()
for D in -2 0 2; do
  V=$((N_STAR + D))
  if [ "$V" -ge 1 ] && [ "$V" -le 10 ]; then N_LIST+=("$V"); fi
done

for K in "${K_LIST[@]}"; do
  for N in "${N_LIST[@]}"; do
    if [ "$K" = "$K_STAR" ] && [ "$N" = "$N_STAR" ]; then continue; fi
    CELL_DIR="experiments/outputs/exp1_kmn/phaseC/k${K}_n${N}_m${M_STAR}"
    if [ -s "${CELL_DIR}/timing.json" ]; then
      echo "[phaseC] SKIP k=${K} n=${N} (done)"
      continue
    fi
    bash experiments/exp1_kmn/run_cell.sh "${K}" "${N}" "${M_STAR}" phaseC \
      || echo "[phaseC] FAIL k=${K} n=${N}"
  done
done

# Final aggregate + best_config.json
python -m experiments.exp1_kmn.aggregate \
  --root experiments/outputs/exp1_kmn \
  --out experiments/outputs/exp1_kmn/_aggregate.csv

python -c "
import csv, json
rows=[r for r in csv.DictReader(open('experiments/outputs/exp1_kmn/_aggregate.csv')) if r['hit_at_5']]
rows.sort(key=lambda r: (-float(r['hit_at_5']), -float(r['mrr'] or 0), float(r['total_sec'] or 1e9)))
best = rows[0]
out = {
  'k': int(best['k']), 'n': int(best['n']), 'm': int(best['m']),
  'phase': best['phase'], 'cell': best['cell'],
  'hit_at_5': float(best['hit_at_5']),
  'mrr': float(best['mrr'] or 0),
  'mean_gold_rank': float(best['mean_gold_rank'] or 0),
  'total_sec': float(best['total_sec'] or 0),
  'encoder': 'gte',
  'frozen_queries': 'experiments/outputs/exp1_kmn/_frozen/queries_v3_chatgpt_gold41.jsonl',
  'corpus_path': f'experiments/outputs/exp1_kmn/{best[\"phase\"]}/{best[\"cell\"]}/search/search_documents.jsonl',
}
json.dump(out, open('experiments/outputs/exp1_kmn/best_config.json','w'), indent=2, ensure_ascii=False)
print('best_config.json:', out)
"

echo "[phaseBC] DONE"
