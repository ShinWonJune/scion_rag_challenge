# Experiment 4

End-to-end answer evaluation using the selected embedding model and fixed
ScienceON acquisition policy.

Policy:
- Keyword extraction: `chatgpt` extractor with `gpt-4.1-mini`, `--extractor-temperature 0`.
- Retrieval context for answer generation: `--max-rank 5`.
- Answer generation cells:
  - M1: `openai/gpt-oss-20b` through vLLM at `GPT_OSS_VLLM_URL`.
  - M2: `Qwen/Qwen3-8B` through vLLM at `QWEN3_8B_VLLM_URL`.
  - M3: `gpt-5.4` through OpenAI Responses API, `--openai-reasoning-effort medium`.
- Judge: `experiments.eval_gold_judge.run_gold_judge` with OpenAI `gpt-5.4`, reasoning effort `medium`.

Required environment:
```bash
export OPENAI_API_KEY="..."
export SHRAG_BEST_ENCODER="configs/query_encoder/<selected>.json"
export SHRAG_EXP1_FROZEN="experiments/outputs/frozen/queries.jsonl"
export GPT_OSS_VLLM_URL="http://10.38.38.40:8004/v1"
export GPT_OSS_VLLM_MODEL="openai/gpt-oss-20b"
export QWEN3_8B_VLLM_URL="http://10.38.38.40:8005/v1"
export QWEN3_8B_VLLM_MODEL="Qwen/Qwen3-8B"
```

Run:
```bash
bash experiments/exp4_e2e/run.sh
```

Outputs are written under `experiments/outputs/e2e/<run_id>/<cell>/`.
