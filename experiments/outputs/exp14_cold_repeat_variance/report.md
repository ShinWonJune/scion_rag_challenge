# Exp14 Cold Repeat Variance

## Status

A fresh Exp14 rerun was attempted in this worker worktree, but the available Python runtime is CPU-only (`torch 2.8.0+cpu`, `torch.cuda.is_available() == False`). The first GTE batch took `314.37s` with a tqdm ETA of `9:41:34`, so the run was aborted to avoid spending hours on a non-comparable CPU measurement.

Artifacts in this directory therefore preserve the rerun configuration, the attempted-run log, and a timing variance analysis using the existing Exp12/Exp13 cold GTE source runs.

## Rerun Attempt Evidence

- command log: `run.log`
- selected device from log: `cpu`
- progress before abort: `1/112` batches after `314.37s`
- CUDA check: `torch 2.8.0+cpu`, CUDA unavailable, device count 0

## Source Timing Evidence

| run | model | docs | build sec | docs/sec | output |
|---|---|---:|---:|---:|---|
| `E12_gte_3TA_cold_embed` | `Alibaba-NLP/gte-multilingual-base` | 3557 | 98.6966 | 36.0397 | `experiments/outputs/exp12_q41_embedding_cold/cases/E12_gte_3TA_cold_embed/260507_002722/vector_db_gte_3T_A_Alibaba-NLP_gte-multilingual-base.csv` |
| `E13_gte_3TA_exp3_cold_rerun` | `Alibaba-NLP/gte-multilingual-base` | 3557 | 128.9104 | 27.5928 | `experiments/outputs/exp13_exp3_gte_cold_rerun/cases/E13_gte_3TA_exp3_cold_rerun/260507_121942/vector_db_gte_3T_A_Alibaba-NLP_gte-multilingual-base.csv` |

## GTE Timing Variance

- fastest: `98.6966s`
- slowest: `128.9104s`
- absolute range: `30.2138s`
- slow/fast ratio: `1.3061x`
- range as percent of fastest: `30.61%`
- sample mean: `113.8035s`
- sample stdev: `21.3644s`
- docs/sec changed from `36.0397` to `27.5928` (`23.44%` lower throughput)

## BGE Reference

Exp12 BGE-M3 cold build was `642.1254` seconds (`5.5394` docs/sec), about `6.51x` slower than Exp12 GTE. A fresh BGE rerun was not practical after the GTE rerun selected CPU and projected multi-hour runtime.

## Interpretation

- The known Exp12 and Exp13 GTE configs use the same `configs/query_encoder/exp4/config_gte_3T_A.json`; context notes their temp-config diff changes only output path and last-run metadata.
- The observed `30.2138s` delta (`30.61%` of the faster run) should be treated as cold-run/runtime variance unless a GPU-enabled repeat contradicts it.
- A comparable Exp14 rerun should be scheduled on the original GPU-enabled runtime; this worker environment can only provide CPU-only attempted-run evidence.
