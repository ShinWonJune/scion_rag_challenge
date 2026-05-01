# Gold-Abstract LLM Judge Evaluation

Evaluate pipeline answers against manually judged ScienceON FULL gold abstracts.

This experiment does not change search or collection. It filters evaluation to the
41 qids in `experiments/gold_scienceon/artifacts/scienceon_gold_full_only.jsonl`.

## Method

The evaluator uses gold evidence directly, rather than comparing the pipeline
answer to another generated reference answer. For each FULL gold qid it sends:

- question
- gold ScienceON document title
- gold ScienceON document abstract
- retrieved ScienceON context used by answer generation
- pipeline answer

The judge must use only the gold title/abstract as answer-quality ground truth
and must separately check whether the answer is grounded in the retrieved
context. It scores:

- `answer_relevance` from 0 to 2
- `evidence_coverage` from 0 to 3
- `faithfulness_to_gold` from 0 to 3
- `faithfulness_to_retrieved` from 0 to 3
- `specificity` from 0 to 2
- `overall` from 0 to 10
- `grounding_flag`: `grounded`, `partially_grounded`, `ungrounded`, or an error flag
- `escape_used`: whether the answer abstained because retrieved documents were insufficient

The script does not blindly trust the judge's `label`. It recomputes
`server_computed_label` from the rubric and records `label_mismatch` when the
judge label disagrees:

- `abstain`: answer used the configured insufficient-evidence escape hatch
- `correct`: `overall >= 8` and `faithfulness_to_gold >= 2` and `evidence_coverage >= 2`
- `incorrect`: `overall <= 3` or `faithfulness_to_gold == 0` or `answer_relevance == 0`
- otherwise `partial`

Use a judge model from a different family than the answer generation model when
possible. If `--generation-model` equals `--judge-model`, the script emits a
warning because self-preference bias is possible.

## Retrieval Metrics Caveat

The retrieval metrics reported by this script are computed from the retrieval
payload embedded in each final answer file. For the current fullrun, Step4 used
`--top-k 5` but Step5 used `--max-rank 1`, so final answer files contain only
the rank-1 context used for answer generation. Therefore `hit_at_3` and
`hit_at_5` in this script should be interpreted as "gold was present in the
answer-generation context", not as full Step4 top-k retrieval performance.

To analyze acquisition/ranking bottlenecks separately, evaluate Step1 search
outputs and Step4 retrieval directories directly.

## Inputs

- Pipeline final output directory, containing one `<qid>.json` per question.
- Gold file: `experiments/gold_scienceon/artifacts/scienceon_gold_full_only.jsonl`.

## Outputs

- `judge_results.jsonl`: one LLM judge result per FULL qid.
- `summary.json`: aggregate answer-quality and retrieval metrics.
- `run_manifest.json`: model, prompt version, gold hash, CLI args, timestamps, git SHA.
- `judge_review.md`: human-readable review with low scores and errors.

## Judge Backends

Use `vllm` for externally hosted open-weight judges such as GPT-OSS 20B and
Qwen3-8B. vLLM is preferred over Ollama for this experiment because it exposes an
OpenAI-compatible server API, supports continuous batching, has better
multi-request throughput, and fits the existing runner without a separate
adapter. Ollama is reasonable for local smoke tests, but it is less suitable for
repeatable multi-model evaluation runs.

Use `openai` for ChatGPT/OpenAI API judges. The runner uses the OpenAI Responses
API and reads the API key from `--api-key` or `OPENAI_API_KEY`. GPT-5.4 is
available in the API as `gpt-5.4`; use a dated snapshot if exact reproducibility
is more important than tracking the alias.

Recommended judge set:

- `openai/gpt-oss-20b` through `--judge-backend vllm`
- `Qwen/Qwen3-8B` or the exact model id exposed by your vLLM server through `--judge-backend vllm`
- `gpt-5.4` through `--judge-backend openai`

## Recommended Commands

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m experiments.eval_gold_judge.run_gold_judge `
  --final-dir outputs\e2e_vllm20b_testcsv_scienceon_fullrun\final `
  --gold experiments\gold_scienceon\artifacts\scienceon_gold_full_only.jsonl `
  --judge-backend vllm `
  --judge-model openai/gpt-oss-20b `
  --generation-model openai/gpt-oss-20b `
  --vllm-url http://10.38.38.40:8004/v1 `
  --output-dir experiments\eval_gold_judge\artifacts\e2e_vllm20b_testcsv_scienceon_fullrun
```

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m experiments.eval_gold_judge.run_gold_judge `
  --final-dir outputs\e2e_v3_max5\final `
  --gold experiments\gold_scienceon\artifacts\scienceon_gold_full_only.jsonl `
  --judge-backend vllm `
  --judge-model Qwen/Qwen3-8B `
  --generation-model openai/gpt-oss-20b `
  --vllm-url http://<qwen-vllm-host>:8000/v1 `
  --output-dir experiments\eval_gold_judge\artifacts\e2e_v3_max5_judge_qwen3_8b
```

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:OPENAI_API_KEY='<your-key>'
.\.venv\Scripts\python.exe -m experiments.eval_gold_judge.run_gold_judge `
  --final-dir outputs\e2e_v3_max5\final `
  --gold experiments\gold_scienceon\artifacts\scienceon_gold_full_only.jsonl `
  --judge-backend openai `
  --judge-model gpt-5.4 `
  --reasoning-effort low `
  --generation-model openai/gpt-oss-20b `
  --output-dir experiments\eval_gold_judge\artifacts\e2e_v3_max5_judge_gpt54
```

Use `--dry-run` to validate parsing and filtering without calling an LLM.

## Expected Final Answer Schema

Each final answer JSON should include:

- `id`: question id. Filename stem is used as fallback.
- `result` or `answer`: pipeline answer text.
- `retrival` or `retrieval`: embedded retrieval payload. The current pipeline
  uses the misspelled key `retrival`, so the evaluator supports both.
