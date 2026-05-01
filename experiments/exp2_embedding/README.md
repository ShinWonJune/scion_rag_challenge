# Experiment 2

Fixed-corpus embedding and retrieval benchmark across multiple encoder models.

Policy:
- Keyword extraction: `chatgpt` extractor with `gpt-4.1-mini`, `--extractor-temperature 0`.
- Corpus source: ScienceON only.
- Evaluation: `experiments.shared.evaluate.embed_benchmark` on FULL ScienceON gold.

Run:
```bash
bash experiments/exp2_embedding/run.sh
```

Outputs are written under `experiments/outputs/embedding_benchmark/<run_id>/`.
