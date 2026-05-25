"""Head-to-head 비교: (k=20, rerank c=10) vs (k=30, rerank c=5) — 수집부터 reranking까지

Stage 1 (subprocess): step1_search 실측 (ScienceON 호출, 캐시 layer 포함)
Stage 2 (in-process): 같은 Python 프로세스에서 retriever + reranker 1회 로드,
        각 config 별로 corpus run1(cold JIT)/run2(warm), query encode,
        dense retrieve, rerank warmup → run1(warm) → run2(fully warm) 측정.

두 config 모두 같은 warm-up 프로토콜·동일 GPU 상태에서 측정.
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from collections import defaultdict

RETRIEVER = "Alibaba-NLP/gte-multilingual-base"
RERANKER  = "dragonkue/bge-reranker-v2-m3-ko"

CONFIGS = {
    "A_k20_c10": {
        "label": "k=20, n=10, m=50 + rerank top-10",
        "max_pages": 2,           # k = 10 × max_pages
        "candidates": 10,
        "frozen": "experiments/outputs/exp1_kmn/_frozen/queries_v3_n10.jsonl",
    },
    "B_k30_c5": {
        "label": "k=30, n=10, m=50 + rerank top-5",
        "max_pages": 3,
        "candidates": 5,
        "frozen": "experiments/outputs/exp1_kmn/_frozen/queries_v3_n10.jsonl",
    },
}
QUERIES_PATH = "experiments/outputs/exp1_kmn/questions_gold41.jsonl"
GOLD_PATH = "data/gold/scienceon_gold.json"


def _load_corpus(p):
    docs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: docs.append(json.loads(line))
    return docs


def _load_queries(p):
    qs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            qid = str(d.get("id") or d.get("question_id") or "")
            text = d.get("question") or d.get("query") or d.get("text") or ""
            qs.append({"id": qid, "text": text})
    return qs


def _hit(hits, gold, k):
    return 1.0 if any(d in gold for d in hits[:k]) else 0.0


def _mrr(hits, gold, k):
    for rank, d in enumerate(hits[:k], 1):
        if d in gold: return 1.0/rank
    return 0.0


def run_step1(cfg_key, cfg, out_root: Path) -> dict:
    """Run step1_search live (ScienceON + cache) and measure wall clock + corpus path."""
    out_dir = out_root / cfg_key / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "shrag.pipeline.steps.step1_search",
        "--questions", QUERIES_PATH,
        "--sources", "scienceon",
        "--extractor", "vllm",
        "--vllm-url", os.environ.get("GPT_OSS_VLLM_URL", ""),
        "--vllm-model", os.environ.get("GPT_OSS_VLLM_MODEL", "openai/gpt-oss-20b"),
        "--extractor-temperature", "0",
        "--frozen-queries", cfg["frozen"],
        "--target-documents", "50",
        "--scienceon-max-pages", str(cfg["max_pages"]),
        "--scienceon-max-concurrency", "2",
        "--scienceon-min-interval-sec", "0.5",
        "--output-dir", str(out_dir),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"step1 failed for {cfg_key}: {r.stderr[-1500:]}")
    # locate produced search_documents.jsonl (step1 nests under a timestamp dir)
    found = sorted(out_dir.rglob("search_documents.jsonl"))
    if not found:
        raise RuntimeError(f"step1 produced no search_documents.jsonl under {out_dir}")
    return {"step1_wall_sec": round(wall, 4), "corpus_path": str(found[-1])}


def measure_config(cfg_key, cfg, corpus_path, retriever, reranker, queries, gold,
                   torch, np, device, retriever_batch=32, reranker_batch=16):
    out = {"label": cfg["label"], "candidates": cfg["candidates"]}

    # corpus prep
    docs = _load_corpus(corpus_path)
    doc_ids = [str(d.get("doc_id") or d.get("CN") or "") for d in docs]
    doc_texts = [f"{d.get('title','') or ''} {d.get('title','') or ''} {d.get('title','') or ''} {d.get('abstract','') or ''}" for d in docs]
    out["corpus_size"] = len(docs)

    q_texts = [q["text"] for q in queries]

    def _sync():
        if device == "cuda": torch.cuda.synchronize()

    # encode corpus run1 (cold JIT for this corpus)
    t0 = time.perf_counter()
    doc_embs = retriever.encode(doc_texts, batch_size=retriever_batch,
                                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    _sync()
    out["encode_corpus_run1_sec"] = round(time.perf_counter() - t0, 4)

    # encode corpus run2 (warm)
    t0 = time.perf_counter()
    _ = retriever.encode(doc_texts, batch_size=retriever_batch,
                         convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    _sync()
    out["encode_corpus_run2_sec"] = round(time.perf_counter() - t0, 4)

    # encode queries
    t0 = time.perf_counter()
    q_embs = retriever.encode(q_texts, batch_size=retriever_batch,
                              convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    _sync()
    out["encode_query_run1_sec"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    _ = retriever.encode(q_texts, batch_size=retriever_batch,
                         convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    _sync()
    out["encode_query_run2_sec"] = round(time.perf_counter() - t0, 4)

    # dense retrieve top-K
    t0 = time.perf_counter()
    sims = q_embs @ doc_embs.T
    top_idx = np.argsort(-sims, axis=1)[:, : cfg["candidates"]]
    out["retrieve_sec"] = round(time.perf_counter() - t0, 4)

    # build pairs
    pairs, meta = [], []
    for qi, q in enumerate(queries):
        for di in top_idx[qi]:
            t = docs[di].get("title", "") or ""
            a = docs[di].get("abstract", "") or ""
            pairs.append([q["text"], f"{t}\n{a}".strip()])
            meta.append((q["id"], doc_ids[di]))
    out["n_pairs"] = len(pairs)

    # rerank warmup (per config to be fair)
    t0 = time.perf_counter()
    _ = reranker.predict(pairs[:reranker_batch], batch_size=reranker_batch, show_progress_bar=False)
    _sync()
    out["rerank_warmup_sec"] = round(time.perf_counter() - t0, 4)

    # rerank run1 (warm)
    t0 = time.perf_counter()
    _ = reranker.predict(pairs, batch_size=reranker_batch, show_progress_bar=False)
    _sync()
    out["rerank_run1_sec"] = round(time.perf_counter() - t0, 4)

    # rerank run2 (fully warm)
    t0 = time.perf_counter()
    scores2 = reranker.predict(pairs, batch_size=reranker_batch, show_progress_bar=False)
    _sync()
    out["rerank_run2_sec"] = round(time.perf_counter() - t0, 4)

    # Hit / MRR from rerank scores
    per_q = defaultdict(list)
    for (qid, did), s in zip(meta, scores2):
        per_q[qid].append((float(s), did))
    hits_by_q = {}
    for qid, items in per_q.items():
        items.sort(key=lambda x: x[0], reverse=True)
        hits_by_q[qid] = [d for _, d in items]

    common = [q for q in hits_by_q if q in gold and gold[q]]
    for k in [1, 3, 5, 10]:
        if k > cfg["candidates"]: continue
        v = [_hit(hits_by_q[q], gold[q], k) for q in common]
        out[f"hit_at_{k}"] = round(sum(v)/len(v), 6) if common else None
    mrrs = [_mrr(hits_by_q[q], gold[q], 10) for q in common]
    out["mrr_at_10"] = round(sum(mrrs)/len(mrrs), 6) if common else None
    ranks = []
    for q in common:
        for rank, d in enumerate(hits_by_q[q], 1):
            if d in gold[q]: ranks.append(rank); break
    out["mean_gold_rank"] = round(sum(ranks)/len(ranks), 4) if ranks else None
    out["n_queries_evaluated"] = len(common)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="experiments/outputs/exp_corpus_rerank_compare")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--torch-dtype", default="float16")
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"started_at": time.time()}

    t0 = time.perf_counter()
    import torch, numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder
    summary["import_sec"] = round(time.perf_counter() - t0, 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary["device"] = device
    if device == "cuda": torch.cuda.reset_peak_memory_stats()

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(args.torch_dtype)

    # one-time loads
    t0 = time.perf_counter()
    try:
        retriever = SentenceTransformer(RETRIEVER, trust_remote_code=True, device=device,
                                        model_kwargs={"torch_dtype": dt, "attn_implementation": "sdpa"})
    except Exception:
        retriever = SentenceTransformer(RETRIEVER, trust_remote_code=True, device=device,
                                        model_kwargs={"torch_dtype": dt})
    if args.max_seq_length: retriever.max_seq_length = int(args.max_seq_length)
    summary["retriever_load_cold_sec"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    try:
        reranker = CrossEncoder(RERANKER, device=device, trust_remote_code=True)
    except TypeError:
        reranker = CrossEncoder(RERANKER, device=device)
    summary["reranker_load_cold_sec"] = round(time.perf_counter() - t0, 4)

    # data
    queries = _load_queries(QUERIES_PATH)
    summary["query_count"] = len(queries)
    gold = json.loads(Path(GOLD_PATH).read_text(encoding="utf-8"))
    gold = {str(k): {str(v).strip() for v in vs} for k, vs in gold.items()}

    # Stage 1: step1_search subprocess for each config (warm ScienceON cache)
    step1_info = {}
    for key, cfg in CONFIGS.items():
        print(f"[{key}] step1_search ...", flush=True)
        info = run_step1(key, cfg, out_dir)
        step1_info[key] = info
        print(f"  step1_wall={info['step1_wall_sec']}s corpus={info['corpus_path']}")

    # Stage 2: encode + retrieve + rerank per config (models already loaded)
    results = {}
    for key, cfg in CONFIGS.items():
        print(f"[{key}] {cfg['label']} ...", flush=True)
        results[key] = measure_config(key, cfg, step1_info[key]["corpus_path"],
                                      retriever, reranker, queries, gold,
                                      torch, np, device)
        results[key]["step1_wall_sec"] = step1_info[key]["step1_wall_sec"]
        print(f"  done. corpus={results[key]['corpus_size']} hit@1={results[key].get('hit_at_1')} "
              f"hit@5={results[key].get('hit_at_5')} run2={results[key]['rerank_run2_sec']}s")

    # compose totals
    rload = summary["retriever_load_cold_sec"]
    rrload = summary["reranker_load_cold_sec"]
    for key, r in results.items():
        # 수집부터 rerank까지 (cold load 1회 포함, run1 사용):
        r["pipeline_cold_with_step1_sec"] = round(
            r["step1_wall_sec"] + rload + r["encode_corpus_run1_sec"]
            + r["encode_query_run1_sec"] + r["retrieve_sec"]
            + rrload + r["rerank_warmup_sec"] + r["rerank_run1_sec"], 4)
        # warm per batch (models+corpus 이미 메모리, step1 cache hit):
        r["pipeline_warm_per_batch_sec"] = round(
            r["step1_wall_sec"] + r["encode_query_run2_sec"]
            + r["retrieve_sec"] + r["rerank_run2_sec"], 4)
        r["warm_per_query_sec"] = round(r["pipeline_warm_per_batch_sec"] / max(1, summary["query_count"]), 6)

    if device == "cuda":
        summary["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated()/(1024**2), 2)

    summary["results"] = results
    summary["finished_at"] = time.time()
    summary["wall_sec"] = round(summary["finished_at"] - summary["started_at"], 2)

    (out_dir / "compare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_dir / 'compare_summary.json'}")
    print(f"Total wall: {summary['wall_sec']}s")


if __name__ == "__main__":
    main()
