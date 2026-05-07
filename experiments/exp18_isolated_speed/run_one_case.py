"""Single-case isolated embedding-speed runner.

Designed to be invoked as a subprocess: parent process can monitor RAM/VRAM and kill cleanly
if memory thresholds are exceeded. Each invocation does ONE cold model load → encode →
save → exit, so CUDA context, cuDNN cache, and PyTorch state are not shared across cases.

Output: writes result.json in --output-dir with timing + memory + status.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Allow tf32 for fair speed comparison; SentenceTransformers handles dtype internally.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def percentiles(values, qs=(50, 90, 95, 99)):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    out = {"min": s[0], "max": s[-1]}
    for q in qs:
        idx = max(0, min(n - 1, int(round(q / 100 * (n - 1)))))
        out[f"p{q}"] = s[idx]
    return out


def load_corpus(corpus_path: str, embedding_mode: str) -> list[str]:
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
            elif embedding_mode in ("3*title+abstract",):
                text = f"{title} {title} {title} {abstract}"
            elif embedding_mode == "T+A":
                text = f"{title} {abstract}"
            elif embedding_mode == "T":
                text = title
            elif embedding_mode == "A":
                text = abstract
            else:
                raise ValueError(f"unknown embedding_mode: {embedding_mode}")
            docs.append(text)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--embedding-dim", type=int, required=True)
    ap.add_argument("--embedding-mode", default="3T+A")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--torch-dtype", default=None, help="float16 / bfloat16 / null")
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--attn-impl", default=None)
    ap.add_argument("--length-bucket", action="store_true")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    # Initialize result early so partial failures still emit something useful
    result = {
        "case_id": args.case_id,
        "model_name": args.model_name,
        "embedding_dim": args.embedding_dim,
        "embedding_mode": args.embedding_mode,
        "batch_size": args.batch_size,
        "torch_dtype": args.torch_dtype,
        "max_seq_length": args.max_seq_length,
        "attn_implementation": args.attn_impl,
        "length_bucket": bool(args.length_bucket),
        "status": "started",
        "started_at": time.time(),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    try:
        # Lazy imports so that argparse failures don't pay torch startup cost.
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        result["device"] = device
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()

        # Reset peak memory counter
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---- Load model (cold) ----
        load_start = time.perf_counter()
        kwargs = {"trust_remote_code": True, "truncate_dim": int(args.embedding_dim), "device": device}
        model_kwargs = {}
        if args.torch_dtype:
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dt = dtype_map.get(args.torch_dtype)
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

        # ---- Build texts ----
        texts = load_corpus(args.corpus, args.embedding_mode)
        result["doc_count"] = len(texts)

        # ---- Token-length distribution (single-shot, before encoding) ----
        tokenizer = model.tokenizer
        lengths = []
        for t in texts:
            enc = tokenizer(t, add_special_tokens=True, truncation=False, return_attention_mask=False)
            lengths.append(len(enc["input_ids"]))
        result["token_lengths"] = percentiles(lengths)

        # Optional: length bucketing
        encode_texts = texts
        order = None
        if args.length_bucket:
            order = sorted(range(len(texts)), key=lambda i: lengths[i])
            encode_texts = [texts[i] for i in order]

        # ---- Encode ----
        enc_start = time.perf_counter()
        embeddings = model.encode(
            encode_texts,
            batch_size=int(args.batch_size),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        encode_sec = time.perf_counter() - enc_start

        # restore order
        if order is not None:
            import numpy as np
            restored = [None] * len(texts)
            for new_i, orig_i in enumerate(order):
                restored[orig_i] = embeddings[new_i]
            embeddings = np.stack(restored, axis=0)

        result["encode_sec"] = round(encode_sec, 4)
        result["actual_dim"] = int(embeddings.shape[1])
        result["docs_per_sec"] = round(len(texts) / encode_sec, 4) if encode_sec > 0 else None

        # ---- Save ----
        import numpy as np
        save_start = time.perf_counter()
        np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
        result["save_sec"] = round(time.perf_counter() - save_start, 4)

        # Memory peak
        if device == "cuda":
            result["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

        result["total_sec"] = round(result["load_sec"] + result["encode_sec"] + result["save_sec"], 4)
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
