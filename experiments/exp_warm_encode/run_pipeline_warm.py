"""End-to-end warm pipeline benchmark: encoder + dense retrieve + cross-encoder rerank.

Per config:
  1. Load encoder + reranker (excluded from timing)
  2. Warmup pass: full pipeline once (discarded)
  3. N warm runs of full pipeline, report median per stage
  4. Quality metrics (Hit@1/3/5/10, MRR@10) computed on final top-K output

Subprocess-per-config keeps CUDA state isolated.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def make_text(doc: dict, mode: str = "3T+A") -> str:
    title = doc.get("title", "") or ""
    abstract = doc.get("abstract", "") or ""
    if mode == "3T+A":
        return f"{title} {title} {title} {abstract}"
    return f"{title} {abstract}"


def compute_metrics(predictions: dict, gold: dict, ks=(1, 3, 5, 10)) -> dict:
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    gold_ranks = []
    n_q = 0
    for qid, pred_list in predictions.items():
        gold_ids = set(gold.get(str(qid), []) or gold.get(qid, []))
        if not gold_ids:
            continue
        n_q += 1
        for k in ks:
            if any(d in gold_ids for d in pred_list[:k]):
                hits[k] += 1
        rank = None
        for i, d in enumerate(pred_list[:10]):
            if d in gold_ids:
                rank = i + 1
                break
        if rank is not None:
            rr_sum += 1.0 / rank
            gold_ranks.append(rank)
    return {
        **{f"hit@{k}": round(hits[k] / n_q, 6) if n_q else 0 for k in ks},
        "mrr@10": round(rr_sum / n_q, 6) if n_q else 0,
        "mean_gold_rank": round(sum(gold_ranks) / len(gold_ranks), 4) if gold_ranks else None,
        "gold_found_rate": round(len(gold_ranks) / n_q, 6) if n_q else 0,
        "n_questions": n_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-model", required=True)
    ap.add_argument("--encoder-dtype", required=True)
    ap.add_argument("--encoder-max-seq", type=int, required=True)
    ap.add_argument("--encoder-batch", type=int, required=True)
    ap.add_argument("--encoder-dim", type=int, default=None)
    ap.add_argument("--encoder-attn-impl", default="sdpa")
    ap.add_argument("--reranker-model", required=True)
    ap.add_argument("--rerank-batch", type=int, default=32)
    ap.add_argument("--retrieve-top-k", type=int, required=True)
    ap.add_argument("--rerank-candidates", type=int, required=True)
    ap.add_argument("--output-top-k", type=int, default=5)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--embedding-mode", default="3T+A")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-warm-runs", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    result = {
        "case_id": args.case_id,
        "encoder": {
            "model": args.encoder_model,
            "dtype": args.encoder_dtype,
            "max_seq": args.encoder_max_seq,
            "batch": args.encoder_batch,
            "attn_impl": args.encoder_attn_impl,
            "truncate_dim": args.encoder_dim,
        },
        "reranker": {"model": args.reranker_model, "batch": args.rerank_batch},
        "retrieve_top_k": args.retrieve_top_k,
        "rerank_candidates": args.rerank_candidates,
        "output_top_k": args.output_top_k,
        "embedding_mode": args.embedding_mode,
        "n_warm_runs": args.n_warm_runs,
        "status": "started",
        "started_at": time.time(),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    try:
        import torch
        from sentence_transformers import SentenceTransformer, CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        result["device"] = device

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---- Load data ----
        corpus_data = load_jsonl(args.corpus)
        doc_ids = [d["doc_id"] for d in corpus_data]
        doc_texts = [make_text(d, args.embedding_mode) for d in corpus_data]
        result["n_corpus_docs"] = len(corpus_data)

        queries_data = load_jsonl(args.queries)
        qids = [str(q.get("id") or q.get("qid")) for q in queries_data]
        query_texts = [q.get("question") or q.get("query") or "" for q in queries_data]
        result["n_queries"] = len(queries_data)

        with open(args.gold, encoding="utf-8") as f:
            gold_raw = json.load(f)
        gold = {str(k): (v if isinstance(v, list) else [v]) for k, v in gold_raw.items()}

        # ---- Load encoder ----
        dt_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        model_kwargs = {"torch_dtype": dt_map[args.encoder_dtype]}
        if args.encoder_attn_impl:
            model_kwargs["attn_implementation"] = args.encoder_attn_impl
        enc_kwargs = {"trust_remote_code": True, "device": device, "model_kwargs": model_kwargs}
        if args.encoder_dim:
            enc_kwargs["truncate_dim"] = args.encoder_dim
        encoder = SentenceTransformer(args.encoder_model, **enc_kwargs)
        encoder.max_seq_length = args.encoder_max_seq

        # ---- Load reranker ----
        reranker = CrossEncoder(args.reranker_model, device=device, trust_remote_code=True)

        # ---- One pipeline pass ----
        def one_pass():
            stage = {}
            t = time.perf_counter()
            doc_emb = encoder.encode(
                doc_texts,
                batch_size=args.encoder_batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            stage["corpus_embed_sec"] = time.perf_counter() - t

            t = time.perf_counter()
            q_emb = encoder.encode(
                query_texts,
                batch_size=args.encoder_batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            stage["query_embed_sec"] = time.perf_counter() - t

            t = time.perf_counter()
            sims = q_emb @ doc_emb.T
            topk_idx = np.argsort(-sims, axis=1)[:, : args.retrieve_top_k]
            stage["retrieve_sec"] = time.perf_counter() - t

            t = time.perf_counter()
            predictions: dict[str, list[str]] = {}
            for qi, qid in enumerate(qids):
                cand_idx = topk_idx[qi][: args.rerank_candidates]
                pairs = [(query_texts[qi], doc_texts[idx]) for idx in cand_idx]
                if pairs:
                    scores = reranker.predict(pairs, batch_size=args.rerank_batch, show_progress_bar=False)
                    order = np.argsort(-np.asarray(scores))
                    reranked = [doc_ids[cand_idx[o]] for o in order]
                else:
                    reranked = []
                tail = [doc_ids[idx] for idx in topk_idx[qi][args.rerank_candidates :]]
                predictions[qid] = (reranked + tail)[: args.output_top_k]
            if device == "cuda":
                torch.cuda.synchronize()
            stage["rerank_sec"] = time.perf_counter() - t

            stage["total_sec"] = sum(stage.values())
            stage["per_query_sec"] = (
                stage["query_embed_sec"] + stage["retrieve_sec"] + stage["rerank_sec"]
            ) / len(qids)
            return stage, predictions

        # ---- Warmup ----
        wt = time.perf_counter()
        _, _ = one_pass()
        result["warmup_total_sec"] = round(time.perf_counter() - wt, 4)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ---- Warm runs ----
        runs = []
        final_predictions = None
        for _ in range(args.n_warm_runs):
            stage, preds = one_pass()
            runs.append({k: round(v, 4) for k, v in stage.items()})
            final_predictions = preds

        def median(key):
            vals = sorted(r[key] for r in runs)
            return vals[len(vals) // 2]

        result["warm_runs"] = runs
        result["warm_median"] = {
            "corpus_embed_sec": round(median("corpus_embed_sec"), 4),
            "query_embed_sec": round(median("query_embed_sec"), 4),
            "retrieve_sec": round(median("retrieve_sec"), 4),
            "rerank_sec": round(median("rerank_sec"), 4),
            "total_sec": round(median("total_sec"), 4),
            "per_query_sec": round(median("per_query_sec"), 4),
        }

        result["quality"] = compute_metrics(final_predictions, gold)

        if device == "cuda":
            result["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

        result["status"] = "completed"
        result["finished_at"] = time.time()
    except Exception as e:
        import traceback

        result["status"] = "error"
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback_tail"] = traceback.format_exc()[-3000:]
        result["finished_at"] = time.time()
    finally:
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
