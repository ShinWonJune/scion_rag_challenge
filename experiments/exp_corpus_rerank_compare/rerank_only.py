"""Top-K dense retrieval 결과에 reranker만 적용해 Hit 지표 재산출.

이미 있는 phaseB/k20_n10_m50의 retrieval.jsonl(dense top-50) 위에서
상위 K 후보만 bge-reranker-v2-m3-ko 로 재정렬한다.

비교 대상: exp7d_retriever_reranker_stack 의 E7d_gte_drk_c5/c10 (k=30 corpus).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def _load_corpus_by_id(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            did = str(d.get("doc_id") or d.get("CN") or "")
            if did: out[did] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True, help="dense top-K 결과 jsonl")
    ap.add_argument("--corpus", required=True, help="search_documents.jsonl")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--reranker", default="dragonkue/bge-reranker-v2-m3-ko")
    ap.add_argument("--candidates", type=int, required=True)
    ap.add_argument("--reranker-batch", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    result = {"reranker": args.reranker, "candidates": args.candidates,
              "retrieval_src": args.retrieval, "corpus": args.corpus,
              "started_at": time.time(), "status": "started"}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    try:
        t0 = time.perf_counter()
        import torch
        from sentence_transformers import CrossEncoder
        result["import_sec"] = round(time.perf_counter() - t0, 4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result["device"] = device
        if device == "cuda": torch.cuda.reset_peak_memory_stats()

        # load assets
        corpus = _load_corpus_by_id(args.corpus)
        result["corpus_size"] = len(corpus)

        queries = {}
        with open(args.queries, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                qid = str(d.get("id") or d.get("question_id") or "")
                queries[qid] = d.get("question") or d.get("query") or d.get("text") or ""

        retrieval = {}
        with open(args.retrieval, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                qid = str(row["question_id"])
                hits = row.get("hits", [])
                doc_ids = [str(h.get("doc_id") if isinstance(h, dict) else h) for h in hits]
                retrieval[qid] = doc_ids

        gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        gold = {str(k): {str(v).strip() for v in vs} for k, vs in gold.items()}

        # cold load
        t0 = time.perf_counter()
        try:
            reranker = CrossEncoder(args.reranker, device=device, trust_remote_code=True)
        except TypeError:
            reranker = CrossEncoder(args.reranker, device=device)
        result["reranker_load_cold_sec"] = round(time.perf_counter() - t0, 4)

        # build pairs (top-candidates per query)
        pairs, meta = [], []
        for qid, doc_ids in retrieval.items():
            qtext = queries.get(qid, "")
            for did in doc_ids[:args.candidates]:
                d = corpus.get(did, {})
                doc_text = f"{d.get('title','') or ''}\n{d.get('abstract','') or ''}".strip()
                pairs.append([qtext, doc_text]); meta.append((qid, did))
        result["n_pairs"] = len(pairs)

        # warmup
        t0 = time.perf_counter()
        _ = reranker.predict(pairs[:args.reranker_batch], batch_size=args.reranker_batch, show_progress_bar=False)
        if device == "cuda": torch.cuda.synchronize()
        result["rerank_warmup_sec"] = round(time.perf_counter() - t0, 4)

        # run1 (warm)
        t0 = time.perf_counter()
        _scores1 = reranker.predict(pairs, batch_size=args.reranker_batch, show_progress_bar=False)
        if device == "cuda": torch.cuda.synchronize()
        result["rerank_run1_sec"] = round(time.perf_counter() - t0, 4)

        # run2 (fully warm)
        t0 = time.perf_counter()
        scores2 = reranker.predict(pairs, batch_size=args.reranker_batch, show_progress_bar=False)
        if device == "cuda": torch.cuda.synchronize()
        result["rerank_run2_sec"] = round(time.perf_counter() - t0, 4)

        # metrics
        from collections import defaultdict
        per_q = defaultdict(list)
        for (qid, did), s in zip(meta, scores2):
            per_q[qid].append((float(s), did))
        hits_by_q = {}
        for qid, items in per_q.items():
            items.sort(key=lambda x: x[0], reverse=True)
            hits_by_q[qid] = [d for _, d in items]

        common = [q for q in hits_by_q if q in gold and gold[q]]
        for k in [1, 3, 5, 10]:
            if k > args.candidates: continue
            v = [1.0 if any(d in gold[q] for d in hits_by_q[q][:k]) else 0.0 for q in common]
            result[f"hit_at_{k}"] = round(sum(v)/len(v), 6) if common else None
        mrrs = []
        for q in common:
            r = 0.0
            for rank, d in enumerate(hits_by_q[q][:10], start=1):
                if d in gold[q]: r = 1.0/rank; break
            mrrs.append(r)
        result["mrr_at_10"] = round(sum(mrrs)/len(mrrs), 6) if common else None
        ranks = []
        for q in common:
            for rank, d in enumerate(hits_by_q[q], start=1):
                if d in gold[q]: ranks.append(rank); break
        result["mean_gold_rank"] = round(sum(ranks)/len(ranks), 4) if ranks else None
        result["n_queries_evaluated"] = len(common)

        if device == "cuda":
            result["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated()/(1024**2), 2)

        result["status"] = "completed"
        result["finished_at"] = time.time()
    except Exception as e:
        import traceback
        result["status"] = "error"; result["error"] = str(e)
        result["traceback"] = traceback.format_exc()[-2000:]
        result["finished_at"] = time.time()
    finally:
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
