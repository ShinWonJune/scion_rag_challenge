# shrag_lc — LangChain reimplementation of SHRAG

A from-scratch LangChain build of the SHRAG (Search like Human with RAG)
pipeline that lives in `shrag/`. Same five stages, expressed with LangChain
idioms (LCEL, `Document`, `VectorStore`, `BaseDocumentCompressor`).

```
question
  ─▶ keyword extraction (KO/EN, LCEL chain)        search/keywords.py
  ─▶ search-term build (OR-rotation)               search/terms.py
  ─▶ platform search → Documents                   search/clients.py · acquire.py
  ─▶ 3*title+abstract embedding (gte-multilingual) embeddings.py
  ─▶ FAISS dense retrieval (inner product = cosine) vectorstore.py
  ─▶ cross-encoder rerank (dragonkue, top-3)       rerank.py
  ─▶ grounded answer (vLLM gpt-oss-20b)            prompts.py · generate.py
```

## Defaults (from `experiments/BEST_CONFIG.md`)

| Stage | Default |
|---|---|
| Acquisition | target m=50, OR-only terms |
| Embedding | `Alibaba-NLP/gte-multilingual-base`, fp16, batch=32, max_len=512, `3*title+abstract` |
| Dense retrieval | FAISS, top-5 (MAX_INNER_PRODUCT) |
| Rerank | `dragonkue/bge-reranker-v2-m3-ko`, candidates=5 → top-3 |
| LLM | vLLM `openai/gpt-oss-20b` (OpenAI-compatible) |

All overridable via `PipelineConfig` (`config.py`) or CLI flags.

## Install

Dependencies live in the Windows venv (`.venv/Scripts/python.exe`), not WSL.

```bash
.venv/Scripts/python.exe -m pip install -r requirements-langchain.txt
```

Set endpoints/keys in `.env` (see `.env.example`): `VLLM_BASE_URL`,
`VLLM_MODEL`, and per-source credentials. Wikipedia needs no key; PubMed works
keyless (lower rate limit); ScienceON needs `configs/credentials/...json`.

## Run

```bash
# Wikipedia source needs no credentials — good for a quick check.
.venv/Scripts/python.exe -m shrag_lc.cli run \
  --questions data/test.csv --source wikipedia --limit 3

# Faithful default config (needs ScienceON credentials + vLLM up):
.venv/Scripts/python.exe -m shrag_lc.cli run \
  --questions data/test.csv --source scienceon \
  --encoder configs/query_encoder/config_gte-multilingual-base.json
```

Outputs land in `outputs_lc/run_<timestamp>/` (`predictions.json`, `manifest.json`).

## Evaluate retrieval

```bash
.venv/Scripts/python.exe -m shrag_lc.eval.run_eval \
  --retrieval retrieved.jsonl --gold data/test.csv
```
`retrieved.jsonl` lines: `{"id": "<qid>", "titles": ["...", ...]}` (ranked).
Gold titles come from the `retrieved_article_name_*` columns of `data/test.csv`.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/shrag_lc -q
```
Offline: search-term + IR-metric parity against the original modules, plus a
FAISS/rerank/generate smoke test using fake embeddings and `FakeListChatModel`.

## Module map

| File | Responsibility |
|---|---|
| `config.py` | `Settings` (.env) + `PipelineConfig` (hyperparams, encoder-JSON loader) |
| `llm.py` | chat-model factory (vLLM / OpenAI / Gemini) |
| `prompts.py` | grounding answer prompt + KO/EN keyword prompts |
| `embeddings.py` | 3T+A text builder + `HuggingFaceEmbeddings` factory |
| `vectorstore.py` | FAISS build / save / load |
| `rerank.py` | cross-encoder `BaseDocumentCompressor` |
| `search/keywords.py` | LCEL keyword extraction + parsing |
| `search/terms.py` | OR-rotation search-term builder |
| `search/clients.py` | ScienceON / PubMed / Wikipedia → common schema |
| `search/acquire.py` | term-major acquisition until target |
| `generate.py` | context JSON + answer chain |
| `pipeline.py` | end-to-end orchestration |
| `cli.py` | command-line entrypoint |
| `eval/` | IR metrics + gold-title retrieval eval |
