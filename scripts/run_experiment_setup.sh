#!/usr/bin/env bash
# SHRAG experiment runner setup helper.
# Sources .env, overrides Qwen model id (manual §12 patch), activates shrag conda env.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env (export every key)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Manual §0 + §12: Qwen base model (not Instruct). Override stale .env value.
export QWEN3_8B_VLLM_URL="${QWEN3_8B_VLLM_URL:-http://10.38.38.40:8005/v1}"
export QWEN3_8B_VLLM_MODEL="Qwen/Qwen3-8B"

# Encoding
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

# Conda activation
if [ -f /home/wonjune/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /home/wonjune/miniconda3/etc/profile.d/conda.sh
  conda activate shrag
fi

echo "[setup] project=$PROJECT_ROOT"
echo "[setup] python=$(which python)"
echo "[setup] OPENAI_API_KEY=$([ -n "${OPENAI_API_KEY:-}" ] && echo SET || echo MISSING)"
echo "[setup] GPT_OSS_VLLM_URL=${GPT_OSS_VLLM_URL:-}"
echo "[setup] QWEN3_8B_VLLM_URL=${QWEN3_8B_VLLM_URL:-}"
echo "[setup] QWEN3_8B_VLLM_MODEL=${QWEN3_8B_VLLM_MODEL:-}"
