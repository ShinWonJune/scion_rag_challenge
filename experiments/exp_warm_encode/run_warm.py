"""Warm-state encoding speed benchmark.

Per case: load model → warmup pass (full corpus, discarded) → N warm measurement runs.
Subprocess-per-case keeps CUDA state isolated across configs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_corpus(corpus_path: str, embedding_mode: str = "3T+A") -> list[str]:
    import json as _json
    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            title = d.get("title", "") or ""
            abstract = d.get("abstract", "") or ""
            if embedding_mode == "3T+A":
                text = f"{title} {title} {title} {abstract}"
            else:
                text = f"{title} {abstract}"
            docs.append(text)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--embedding-dim", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--torch-dtype", default=None, help="float16 / bfloat16 / float32")
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--attn-impl", default=None)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-warm-runs", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    result = {
        "case_id": args.case_id,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "torch_dtype": args.torch_dtype,
        "max_seq_length": args.max_seq_length,
        "attn_implementation": args.attn_impl,
        "n_warm_runs": args.n_warm_runs,
        "status": "started",
        "started_at": time.time(),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        result["device"] = device

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        load_start = time.perf_counter()
        kwargs = {"trust_remote_code": True, "truncate_dim": int(args.embedding_dim), "device": device}
        model_kwargs = {}
        if args.torch_dtype:
            dt_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dt = dt_map.get(args.torch_dtype)
            if dt is not None:
                model_kwargs["torch_dtype"] = dt
        if args.attn_impl:
            model_kwargs["attn_implementation"] = args.attn_impl
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        model = SentenceTransformer(args.model_name, **kwargs)
        if args.max_seq_length:
            model.max_seq_length = int(args.max_seq_length)
        result["effective_max_seq_length"] = int(getattr(model, "max_seq_length", 0) or 0)
        result["load_sec"] = round(time.perf_counter() - load_start, 4)

        texts = load_corpus(args.corpus)
        result["doc_count"] = len(texts)

        # Warmup pass
        w_start = time.perf_counter()
        _ = model.encode(
            texts,
            batch_size=int(args.batch_size),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        result["warmup_sec"] = round(time.perf_counter() - w_start, 4)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        run_secs = []
        for _ in range(int(args.n_warm_runs)):
            r_start = time.perf_counter()
            _ = model.encode(
                texts,
                batch_size=int(args.batch_size),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            run_secs.append(round(time.perf_counter() - r_start, 4))

        result["warm_run_secs"] = run_secs
        result["warm_min_sec"] = min(run_secs)
        result["warm_median_sec"] = sorted(run_secs)[len(run_secs) // 2]
        result["warm_mean_sec"] = round(sum(run_secs) / len(run_secs), 4)
        result["docs_per_sec_warm_min"] = round(len(texts) / min(run_secs), 4)

        if device == "cuda":
            result["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

        result["status"] = "completed"
        result["finished_at"] = time.time()
    except Exception as e:
        import traceback
        result["status"] = "error"
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback_tail"] = traceback.format_exc()[-2000:]
        result["finished_at"] = time.time()
    finally:
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
