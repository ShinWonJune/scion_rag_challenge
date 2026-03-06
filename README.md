# scion_rag_challenge

Scientific literature RAG pipeline for the ScienceON challenge.

## Architecture Principle

- `src/`: reusable business logic (search, extraction, retrieval, evaluation)
- `pipeline/`: CLI entrypoints and orchestration only
- dependency direction: `pipeline -> src` only

## Canonical Paths

- configs: `configs/`
- run outputs: `outputs/`
- datasets: `data/`

Do not use `pipeline/configs` or `pipeline/outputs` for new runs.

## Quick Start

```bash
pip install -r requirements.txt
```

Set required API keys in environment variables (or local credential files under `configs/credentials/`).

You can start from `.env.example` and load it into your shell.

```bash
cp .env.example .env
```

PowerShell example:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $name, $value = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
```

```bash
python pipeline/run_pipeline.py \
  --questions data/rag_test_data/questions.jsonl \
  --encoder configs/query_encoder/config_bge_m3.json \
  --llm gemini \
  --output outputs/run_$(date +%Y%m%d)
```

With decomposition:

```bash
python pipeline/run_pipeline.py \
  --questions data/rag_test_data/questions.jsonl \
  --encoder configs/query_encoder/config_bge_m3.json \
  --llm gemini \
  --decompose \
  --output outputs/run_$(date +%Y%m%d)
```

## Step Entrypoints

- `pipeline/step1_search.py`
- `pipeline/step2_decompose.py`
- `pipeline/step3_build_vectordb.py`
- `pipeline/step4_retrieve.py`
- `pipeline/step5_generate.py`
