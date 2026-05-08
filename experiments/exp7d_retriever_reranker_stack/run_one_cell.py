"""Exp7d: retriever × reranker stack 측정 (cold/warm stage-by-stage).

Stages:
  1. retriever load (cold)
  2. encode corpus run1 (cold JIT) + run2 (warm)
  3. encode queries run1 (warm-after-corpus) + run2 (fully warm)
  4. dense retrieve top-K (warm)
  5. reranker load (cold)
  6. reranker warmup pass
  7. rerank run1 (warm) + run2 (fully warm)

→ Hit@k / MRR / mgr from rerank_run2 scores.
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


def _sync(device: str, torch):
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True)
    ap.add_argument("--reranker", required=True)
    ap.add_argument("--candidates", type=int, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--retriever-batch", type=int, default=32)
    ap.add_argument("--reranker-batch", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--torch-dtype", default="float16")
    ap.add_argument("--prefix-passage", default="")
    ap.add_argument("--prefix-query", default="")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "retriever": args.retriever,
        "reranker": args.reranker,
        "candidates": args.candidates,
        "started_at": time.time(),
        "status": "started",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    try:
        # ---- import (cold) ----
        t0 = time.perf_counter()
        import torch, numpy as np
        from sentence_transformers import SentenceTransformer, CrossEncoder
        result["import_sec"] = round(time.perf_counter() - t0, 4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result["device"] = device
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dt = dtype_map.get(args.torch_dtype)

        # ---- 1. retriever load (cold) ----
        t0 = time.perf_counter()
        try:
            retriever = SentenceTransformer(
                args.retriever, trust_remote_code=True, device=device,
                model_kwargs={"torch_dtype": dt, "attn_implementation": "sdpa"},
            )
        except Exception:
            retriever = SentenceTransformer(args.retriever, trust_remote_code=True, device=device,
                                            model_kwargs={"torch_dtype": dt})
        if args.max_seq_length:
            retriever.max_seq_length = int(args.max_seq_length)
        result["retriever_load_cold_sec"] = round(time.perf_counter() - t0, 4)

        # ---- build texts ----
        docs = _load_corpus(args.corpus)
        doc_ids = [str(d.get("doc_id") or d.get("CN") or d.get("cn") or "") for d in docs]
        doc_texts_raw = []
        for d in docs:
            t = d.get("title", "") or ""
            a = d.get("abstract", "") or ""
            doc_texts_raw.append(f"{t} {t} {t} {a}")
        doc_texts = [(args.prefix_passage + x) if args.prefix_passage else x for x in doc_texts_raw]
        result["doc_count"] = len(doc_texts)

        queries = _load_queries(args.queries)
        q_texts_raw = [q["text"] for q in queries]
        q_texts = [(args.prefix_query + x) if args.prefix_query else x for x in q_texts_raw]
        result["query_count"] = len(queries)

        # ---- 2. encode corpus run1 (cold JIT) ----
        t0 = time.perf_counter()
        doc_embs = retriever.encode(doc_texts, batch_size=args.retriever_batch,
                                    convert_to_numpy=True, normalize_embeddings=True,
                                    show_progress_bar=False)
        _sync(device, torch)
        result["encode_corpus_run1_sec"] = round(time.perf_counter() - t0, 4)
        result["actual_dim"] = int(doc_embs.shape[1])

        # ---- 2. encode corpus run2 (warm) ----
        t0 = time.perf_counter()
        _ = retriever.encode(doc_texts, batch_size=args.retriever_batch,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
        _sync(device, torch)
        result["encode_corpus_run2_sec"] = round(time.perf_counter() - t0, 4)

        # ---- 3. encode queries run1 (warm-after-corpus) ----
        t0 = time.perf_counter()
        q_embs = retriever.encode(q_texts, batch_size=args.retriever_batch,
                                  convert_to_numpy=True, normalize_embeddings=True,
                                  show_progress_bar=False)
        _sync(device, torch)
        result["encode_query_run1_sec"] = round(time.perf_counter() - t0, 4)

        # ---- 3. encode queries run2 (fully warm) ----
        t0 = time.perf_counter()
        _ = retriever.encode(q_texts, batch_size=args.retriever_batch,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
        _sync(device, torch)
        result["encode_query_run2_sec"] = round(time.perf_counter() - t0, 4)

        # ---- 4. dense top-K (warm) ----
        t0 = time.perf_counter()
        sims = q_embs @ doc_embs.T
        top_idx = np.argsort(-sims, axis=1)[:, : args.candidates]
        result["retrieve_sec"] = round(time.perf_counter() - t0, 4)

        # ---- 5. reranker load (cold) ----
        t0 = time.perf_counter()
        try:
            reranker = CrossEncoder(args.reranker, device=device, trust_remote_code=True)
        except TypeError:
            reranker = CrossEncoder(args.reranker, device=device)
        result["reranker_load_cold_sec"] = round(time.perf_counter() - t0, 4)

        # ---- build rerank pairs ----
        pairs = []
        meta = []  # (qid, doc_id) per pair
        for qi, q in enumerate(queries):
            qid = q["id"]
            qtext = q_texts_raw[qi]
            for di in top_idx[qi]:
                doc_text = f"{docs[di].get('title','') or ''}\n{docs[di].get('abstract','') or ''}".strip()
                pairs.append([qtext, doc_text])
                meta.append((qid, doc_ids[di]))
        result["n_pairs"] = len(pairs)

        # ---- 6. reranker warmup ----
        t0 = time.perf_counter()
        _ = reranker.predict(pairs[: args.reranker_batch], batch_size=args.reranker_batch, show_progress_bar=False)
        _sync(device, torch)
        result["rerank_warmup_sec"] = round(time.perf_counter() - t0, 4)

        # ---- 7. rerank run1 (warm) ----
        t0 = time.perf_counter()
        scores1 = reranker.predict(pairs, batch_size=args.reranker_batch, show_progress_bar=False)
        _sync(device, torch)
        result["rerank_run1_sec"] = round(time.perf_counter() - t0, 4)

        # ---- 7. rerank run2 (fully warm) ----
        t0 = time.perf_counter()
        scores2 = reranker.predict(pairs, batch_size=args.reranker_batch, show_progress_bar=False)
        _sync(device, torch)
        result["rerank_run2_sec"] = round(time.perf_counter() - t0, 4)

        # ---- e2e summary ----
        # Cold-path = retriever_load + encode_corpus_run1 + encode_query_run1 + retrieve + reranker_load + warmup + rerank_run1
        # Warm-path (per query) = encode_query_run2/n_q + retrieve/n_q + rerank_run2/n_q
        n_q = len(queries)
        result["e2e_cold_total_sec"] = round(
            result["retriever_load_cold_sec"]
            + result["encode_corpus_run1_sec"]
            + result["encode_query_run1_sec"]
            + result["retrieve_sec"]
            + result["reranker_load_cold_sec"]
            + result["rerank_warmup_sec"]
            + result["rerank_run1_sec"], 4)
        result["e2e_warm_per_query_sec"] = round(
            (result["encode_query_run2_sec"] + result["retrieve_sec"] + result["rerank_run2_sec"]) / n_q, 6)
        # Production cold-load (1회) = retriever_load + reranker_load + corpus_encode_run1 + warmup
        result["one_time_cold_load_sec"] = round(
            result["retriever_load_cold_sec"] + result["reranker_load_cold_sec"]
            + result["encode_corpus_run1_sec"] + result["rerank_warmup_sec"], 4)

        # ---- compute retrieval metrics from scores2 ----
        from collections import defaultdict
        per_q = defaultdict(list)
        for (qid, doc_id), s in zip(meta, scores2):
            per_q[qid].append((float(s), doc_id))
        retrieval = []
        for qid, items in per_q.items():
            items.sort(key=lambda x: x[0], reverse=True)
            top5 = [{"rank": i + 1, "score": s, "doc_id": d} for i, (s, d) in enumerate(items[:5])]
            retrieval.append({"question_id": qid, "hits": top5})
        retrieval.sort(key=lambda r: int(r["question_id"]) if r["question_id"].isdigit() else r["question_id"])
        (out_dir / "retrieval.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in retrieval) + "\n", encoding="utf-8")

        # Eval
        gold = _load_gold(args.gold)
        ks = [1, 3, 5, 10]
        hits_by_q = {r["question_id"]: [h["doc_id"] for h in r["hits"]] for r in retrieval}
        common = [q for q in hits_by_q if q in gold]
        for k in ks:
            v = [1.0 if any(d in gold[q] for d in hits_by_q[q][:k]) else 0.0 for q in common]
            result[f"hit_at_{k}"] = round(sum(v) / len(v), 6) if common else None
        mrrs = []
        for q in common:
            r = 0.0
            for rank, d in enumerate(hits_by_q[q][:10], start=1):
                if d in gold[q]:
                    r = 1.0 / rank; break
            mrrs.append(r)
        result["mrr_at_10"] = round(sum(mrrs) / len(mrrs), 6) if common else None
        ranks = []
        for q in common:
            for rank, d in enumerate(hits_by_q[q], start=1):
                if d in gold[q]:
                    ranks.append(rank); break
        result["mean_gold_rank"] = round(sum(ranks) / len(ranks), 4) if ranks else None
        result["n_queries_evaluated"] = len(common)

        if device == "cuda":
            result["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)

        result["status"] = "completed"
        result["finished_at"] = time.time()
    except Exception as e:
        import traceback
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()[-2000:]
        result["finished_at"] = time.time()
    finally:
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
