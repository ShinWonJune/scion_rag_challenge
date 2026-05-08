"""Single-model isolated reranker benchmark.

각 model 별 subprocess: import → load(cold timing) → warmup pass → 측정 pass × 2.
모델 메모리는 1회 로드 후 재사용 → run1=warm-after-warmup, run2=warm-stable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_pairs(candidates_jsonl: Path) -> list[tuple[str, str, str, str]]:
    """Returns list of (question_id, query, doc_id, doc_text) tuples — 41Q × 5 cands = 205 pairs."""
    pairs = []
    with candidates_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r['question_id'])
            for h in r.get('hits', []):
                title = h.get('title', '') or ''
                abstract = h.get('abstract', '') or ''
                doc_text = f"{title}\n{abstract}".strip()
                pairs.append((qid, h.get('query', ''), h.get('doc_id', ''), doc_text))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True, help="JSONL with {question_id, query, hits[{title,abstract,doc_id}]}")
    ap.add_argument("--queries", required=True, help="Original queries.jsonl for query text")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = {
        "model": args.model,
        "started_at": time.time(),
        "status": "started",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    try:
        # ---- Stage 1: import (process startup overhead) ----
        t0 = time.perf_counter()
        import torch
        from sentence_transformers import CrossEncoder
        out["import_sec"] = round(time.perf_counter() - t0, 4)

        out["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if out["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---- Stage 2: model load (cold) ----
        t0 = time.perf_counter()
        try:
            model = CrossEncoder(args.model, device=out["device"], trust_remote_code=True)
        except TypeError:
            model = CrossEncoder(args.model, device=out["device"])
        out["load_sec"] = round(time.perf_counter() - t0, 4)

        # ---- Build pairs ----
        # candidates jsonl has hits with title/abstract per question. Need query text from queries.jsonl
        queries_by_id = {}
        with open(args.queries, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    q = json.loads(line)
                    queries_by_id[str(q.get("id") or q.get("question_id"))] = q.get("question") or q.get("query", "")

        pairs_meta = []  # (qid, query, doc_id, doc_text)
        with open(args.candidates, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = str(r["question_id"])
                qtext = queries_by_id.get(qid, "")
                for h in r.get("hits", []):
                    title = h.get("title", "") or ""
                    abstract = h.get("abstract", "") or ""
                    doc_text = f"{title}\n{abstract}".strip()
                    pairs_meta.append((qid, qtext, h.get("doc_id", ""), doc_text))

        out["n_pairs"] = len(pairs_meta)
        out["n_queries"] = len({m[0] for m in pairs_meta})
        text_pairs = [[m[1], m[3]] for m in pairs_meta]

        # ---- Stage 3: warmup pass — 1 small batch to compile CUDA kernels / cuDNN tuner ----
        t0 = time.perf_counter()
        warmup_batch = text_pairs[: min(args.batch_size, len(text_pairs))]
        _ = model.predict(warmup_batch, batch_size=args.batch_size, show_progress_bar=False)
        if out["device"] == "cuda":
            torch.cuda.synchronize()
        out["warmup_sec"] = round(time.perf_counter() - t0, 4)

        # ---- Stage 4: measured pass run 1 (after warmup) ----
        t0 = time.perf_counter()
        scores1 = model.predict(text_pairs, batch_size=args.batch_size, show_progress_bar=False)
        if out["device"] == "cuda":
            torch.cuda.synchronize()
        out["rerank_run1_sec"] = round(time.perf_counter() - t0, 4)

        # ---- Stage 5: measured pass run 2 (fully warm) ----
        t0 = time.perf_counter()
        scores2 = model.predict(text_pairs, batch_size=args.batch_size, show_progress_bar=False)
        if out["device"] == "cuda":
            torch.cuda.synchronize()
        out["rerank_run2_sec"] = round(time.perf_counter() - t0, 4)

        # ---- Stage 6: per-query latency (run 2 used as canonical warm) ----
        out["sec_per_query_warm"] = round(out["rerank_run2_sec"] / out["n_queries"], 6)
        out["sec_per_pair_warm"] = round(out["rerank_run2_sec"] / out["n_pairs"], 6)

        # ---- Stage 7: derive metrics from scores2 (as final answer) ----
        # Re-score and produce reranked top-K per query — use existing gold to compute Hit@k
        from collections import defaultdict
        per_q = defaultdict(list)
        for (qid, _, doc_id, _), s in zip(pairs_meta, scores2):
            per_q[qid].append((float(s), doc_id))
        # Memory & status
        if out["device"] == "cuda":
            out["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)

        # Save reranked top-5 for downstream gold eval
        reranked = []
        for qid, items in per_q.items():
            items.sort(key=lambda x: x[0], reverse=True)
            top5 = [{"rank": i + 1, "score": s, "doc_id": d} for i, (s, d) in enumerate(items[:5])]
            reranked.append({"question_id": qid, "hits": top5})
        reranked.sort(key=lambda r: int(r["question_id"]) if r["question_id"].isdigit() else r["question_id"])

        retr_path = Path(args.output).parent / "reranked.jsonl"
        with retr_path.open("w", encoding="utf-8") as f:
            for r in reranked:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out["reranked_jsonl"] = str(retr_path)

        out["status"] = "completed"
        out["finished_at"] = time.time()
    except Exception as e:
        import traceback
        out["status"] = "error"
        out["error"] = str(e)
        out["traceback"] = traceback.format_exc()[-2000:]
        out["finished_at"] = time.time()
    finally:
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
