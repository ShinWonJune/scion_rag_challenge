# ScienceON Gold Artifacts

Final evaluation should use only manually judged FULL gold rows.

Primary files:
- `scienceon_gold_full_only.jsonl`: 41 FULL gold questions for final evaluation.
- `scienceon_gold_full_only.csv`: compact review table for the 41 FULL rows.
- `scienceon_gold_full_qids.txt`: qid allowlist for filtering pipeline outputs.
- `scienceon_gold_excluded_partial.jsonl`: 9 excluded PARTIAL rows, kept for audit.
- `scienceon_gold_llm_judge.jsonl`: manual LLM-as-judge review of gold abstract sufficiency.
- `scienceon_gold_docs.jsonl`: all 50 selected ScienceON gold candidates.

Do not use the heuristic `score` as an answer-quality metric. It is only a candidate-selection signal.
For final answer evaluation, use `scienceon_gold_full_only.jsonl` as evidence and judge pipeline answers against the gold title/abstract.
