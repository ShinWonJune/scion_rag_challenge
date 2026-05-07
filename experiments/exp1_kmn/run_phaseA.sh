#!/usr/bin/env bash
# Run Phase A: 5x5 grid of (k, n) with m=70 fixed.
# Cold cell (k=50, n=10, m=70) is run first to warm cache; rest are warm.
# Caller can run cold cell manually first; this driver also handles it (idempotent).
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

KS=(50 40 30 20 10)
NS=(10 8 5 3 1)
M=70

START_ALL=$(date +%s)
TOTAL=$((${#KS[@]} * ${#NS[@]}))
DONE=0
FAILED=()

for K in "${KS[@]}"; do
  for N in "${NS[@]}"; do
    DONE=$((DONE + 1))
    CELL_DIR="experiments/outputs/exp1_kmn/phaseA/k${K}_n${N}_m${M}"
    if [ -s "${CELL_DIR}/timing.json" ]; then
      echo "[phaseA] [${DONE}/${TOTAL}] SKIP k=${K} n=${N} m=${M} (already done)"
      continue
    fi
    echo "[phaseA] [${DONE}/${TOTAL}] START k=${K} n=${N} m=${M} at $(date)"
    if bash experiments/exp1_kmn/run_cell.sh "${K}" "${N}" "${M}" phaseA; then
      echo "[phaseA] [${DONE}/${TOTAL}] DONE k=${K} n=${N} m=${M}"
    else
      echo "[phaseA] [${DONE}/${TOTAL}] FAIL k=${K} n=${N} m=${M}"
      FAILED+=("k${K}_n${N}_m${M}")
    fi
  done
done

END_ALL=$(date +%s)
ELAPSED=$((END_ALL - START_ALL))
echo "[phaseA] total ${TOTAL} cells in $((ELAPSED / 60))m $((ELAPSED % 60))s"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[phaseA] FAILED: ${FAILED[*]}"
  exit 1
fi
echo "[phaseA] ALL DONE"
