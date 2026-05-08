"""Single embedding model evaluation in isolated subprocess.

Loads model with current best runtime (fp16, max_seq=512, sdpa) → encodes corpus
→ encodes queries → cosine top-50 → metrics. Saves result.json + retrieval.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_corpus(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def _load_queries(path: str) -> list[dict]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = str(d.get("id") or d.get("question_id") or "")
            text = d.get("question") or d.get("query") or d.get("text") or ""
            qs.append({"id": qid, "text": text})
    return qs


def _load_gold(path: str) -> dict[str, set[str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): {str(v).strip() for v in vs} for k, vs in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--output", required=True, help="Output dir for result.json + retrieval.jsonl")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--torch-dtype", default="float16")
    ap.add_argument("--prefix-passage", default="")
    ap.add_argument("--prefix-query", default="")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "torch_dtype": args.torch_dtype,
        "prefix_passage": args.prefix_passage,
        "prefix_query": args.prefix_query,
        "started_at": time.time(),
        "status": "started",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    try:
        # ---- import (cold) ----
        t0 = time.perf_counter()
        import torch
        import numpy as np
        from sentence_transformers import SentenceTransformer
        result["import_sec"] = round(time.perf_counter() - t0, 4)
        result["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        result["torch_version"] = torch.__version__

        if result["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---- model load (cold) ----
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dt = dtype_map.get(args.torch_dtype)
        model_kwargs = {}
        if dt is not None:
            model_kwargs["torch_dtype"] = dt

        t0 = time.perf_counter()
        try:
            model = SentenceTransformer(
                args.model,
                trust_remote_code=True,
                device=result["device"],
                model_kwargs={**model_kwargs, "attn_implementation": "sdpa"},
            )
        except Exception:
            # fallback without attn override
            model = SentenceTransformer(
                args.model,
                trust_remote_code=True,
                device=result["device"],
                model_kwargs=model_kwargs,
            )
        if args.max_seq_length:
            model.max_seq_length = int(args.max_seq_length)
        result["effective_max_seq_length"] = int(getattr(model, "max_seq_length", 0) or 0)
        result["load_sec"] = round(time.perf_counter() - t0, 4)

        # ---- build texts (3T+A) ----
        docs = _load_corpus(args.corpus)
        doc_texts = []
        doc_ids = []
        for d in docs:
            title = d.get("title", "") or ""
            abstract = d.get("abstract", "") or ""
            text = f"{title} {title} {title} {abstract}"
            if args.prefix_passage:
                text = args.prefix_passage + text
            doc_texts.append(text)
            doc_ids.append(str(d.get("doc_id") or d.get("CN") or d.get("cn") or ""))
        result["doc_count"] = len(doc_texts)

        queries = _load_queries(args.queries)
        q_texts = [(args.prefix_query + q["text"]) if args.prefix_query else q["text"] for q in queries]
        result["query_count"] = len(queries)

        # ---- token length distribution (sample doc tokenization) ----
        try:
            tokenizer = model.tokenizer
            sample_lengths = []
            for t in doc_texts[: min(500, len(doc_texts))]:
                enc = tokenizer(t, add_special_tokens=True, truncation=False, return_attention_mask=False)
                sample_lengths.append(len(enc["input_ids"]))
            sample_lengths.sort()
            n = len(sample_lengths)
            result["token_lengths_sample"] = {
                "min": sample_lengths[0],
                "max": sample_lengths[-1],
                "p50": sample_lengths[n // 2],
                "p90": sample_lengths[min(n - 1, int(0.90 * (n - 1)))],
                "p99": sample_lengths[min(n - 1, int(0.99 * (n - 1)))],
            }
        except Exception:
            pass

        # ---- encode corpus ----
        t0 = time.perf_counter()
        doc_embs = model.encode(
            doc_texts,
            batch_size=int(args.batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if result["device"] == "cuda":
            torch.cuda.synchronize()
        result["encode_corpus_sec"] = round(time.perf_counter() - t0, 4)
        result["actual_dim"] = int(doc_embs.shape[1])
        result["docs_per_sec"] = round(len(doc_texts) / result["encode_corpus_sec"], 4) if result["encode_corpus_sec"] > 0 else None

        # ---- encode queries ----
        t0 = time.perf_counter()
        q_embs = model.encode(
            q_texts,
            batch_size=int(args.batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if result["device"] == "cuda":
            torch.cuda.synchronize()
        result["encode_query_sec"] = round(time.perf_counter() - t0, 4)

        # ---- retrieve top-K ----
        import numpy as np
        t0 = time.perf_counter()
        sims = q_embs @ doc_embs.T  # (n_q, n_doc)
        top_idx = np.argsort(-sims, axis=1)[:, : args.top_k]
        result["retrieve_sec"] = round(time.perf_counter() - t0, 4)

        # ---- build retrieval.jsonl ----
        retrieval = []
        for qi, q in enumerate(queries):
            hits = []
            for rank, di in enumerate(top_idx[qi]):
                hits.append({"rank": rank + 1, "score": float(sims[qi, di]), "doc_id": doc_ids[di]})
            retrieval.append({"question_id": q["id"], "hits": hits})
        retrieval.sort(key=lambda r: int(r["question_id"]) if r["question_id"].isdigit() else r["question_id"])
        retr_path = out_dir / "retrieval.jsonl"
        with retr_path.open("w", encoding="utf-8") as f:
            for r in retrieval:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # ---- compute metrics ----
        gold = _load_gold(args.gold)
        ks = [1, 3, 5, 10, 20]
        hits_by_q = {r["question_id"]: [h["doc_id"] for h in r["hits"]] for r in retrieval}
        common = [q for q in hits_by_q if q in gold]
        for k in ks:
            hits_list = [1.0 if any(d in gold[q] for d in hits_by_q[q][:k]) else 0.0 for q in common]
            result[f"hit_at_{k}"] = round(sum(hits_list) / len(hits_list), 6) if common else None
        # MRR@10
        mrr_list = []
        for q in common:
            r = 0.0
            for rank, d in enumerate(hits_by_q[q][:10], start=1):
                if d in gold[q]:
                    r = 1.0 / rank
                    break
            mrr_list.append(r)
        result["mrr_at_10"] = round(sum(mrr_list) / len(mrr_list), 6) if common else None
        # mean_gold_rank
        ranks = []
        for q in common:
            for rank, d in enumerate(hits_by_q[q], start=1):
                if d in gold[q]:
                    ranks.append(rank)
                    break
        result["mean_gold_rank"] = round(sum(ranks) / len(ranks), 4) if ranks else None
        result["gold_found_rate"] = round(len(ranks) / len(common), 6) if common else None
        result["n_queries_evaluated"] = len(common)

        # ---- VRAM peak ----
        if result["device"] == "cuda":
            result["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)

        result["status"] = "completed"
        result["finished_at"] = time.time()
    except Exception as e:
        import traceback
        result["status"] = "error"
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        result["traceback"] = traceback.format_exc()[-2000:]
        result["finished_at"] = time.time()
    finally:
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
