"""Warm benchmark for shrag_lc embedding -> retrieve -> rerank.

This intentionally skips live acquisition and answer generation. It mirrors
``experiments/exp_warm_encode/run_pipeline_warm.py`` but routes through the
LangChain components used by ``shrag_lc`` so their overhead is measurable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from shrag_lc.config import PipelineConfig
from shrag_lc.embeddings import build_embedding_text, build_embeddings
from shrag_lc.rerank import build_reranker

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _load_gold(path: str | Path) -> dict[str, set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): {str(x) for x in (v if isinstance(v, list) else [v])} for k, v in payload.items()}


def _flatten_doc(item: dict[str, Any]) -> dict[str, Any]:
    if "metadata" in item:
        metadata = item.get("metadata") or {}
        return {
            "doc_id": metadata.get("doc_id") or item.get("doc_id") or "",
            "title": metadata.get("title") or item.get("title") or "",
            "abstract": metadata.get("abstract") or item.get("abstract") or "",
            "source": metadata.get("source") or item.get("source") or "",
            "url": metadata.get("url") or item.get("url") or "",
        }
    return item


def _make_documents(corpus_data: list[dict[str, Any]], embedding_mode: str) -> list[Document]:
    documents: list[Document] = []
    for raw in corpus_data:
        doc = _flatten_doc(raw)
        doc_id = str(doc.get("doc_id") or doc.get("CN") or doc.get("cn") or "")
        documents.append(
            Document(
                page_content=build_embedding_text(doc, embedding_mode),
                metadata={
                    "doc_id": doc_id,
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "source": doc.get("source", ""),
                    "url": doc.get("url", ""),
                },
            )
        )
    return documents


def _compute_metrics(predictions: dict[str, list[str]], gold: dict[str, set[str]]) -> dict[str, Any]:
    ks = (1, 3, 5, 10)
    qids = [qid for qid in predictions if qid in gold]
    out: dict[str, Any] = {"n_queries": len(qids)}
    reciprocal: list[float] = []
    ranks: list[int] = []
    for qid in qids:
        pred = predictions[qid]
        gold_ids = gold[qid]
        rank = None
        for idx, doc_id in enumerate(pred[:10], start=1):
            if doc_id in gold_ids:
                rank = idx
                break
        reciprocal.append((1.0 / rank) if rank else 0.0)
        if rank:
            ranks.append(rank)
        for k in ks:
            out.setdefault(f"hit@{k}", 0)
            if rank and rank <= k:
                out[f"hit@{k}"] += 1
    n = len(qids) or 1
    for k in ks:
        out[f"hit@{k}"] = round(out[f"hit@{k}"] / n, 6)
    out["mrr@10"] = round(sum(reciprocal) / n, 6)
    out["mean_gold_rank"] = round(sum(ranks) / len(ranks), 4) if ranks else None
    out["gold_found_rate"] = round(len(ranks) / n, 6)
    return out


def _sync_cuda(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


def _build_store_from_vectors(
    documents: list[Document],
    embeddings,
    doc_vectors: list[list[float]],
) -> FAISS:
    return FAISS.from_embeddings(
        text_embeddings=zip((doc.page_content for doc in documents), doc_vectors),
        embedding=embeddings,
        metadatas=[doc.metadata for doc in documents],
        ids=[str(doc.metadata.get("doc_id") or idx) for idx, doc in enumerate(documents)],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-model", required=True)
    ap.add_argument("--encoder-dtype", default="float16")
    ap.add_argument("--encoder-max-seq", type=int, required=True)
    ap.add_argument("--encoder-batch", type=int, required=True)
    ap.add_argument("--encoder-dim", type=int, default=768)
    ap.add_argument("--reranker-model", required=True)
    ap.add_argument("--rerank-batch", type=int, default=16)
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    result: dict[str, Any] = {
        "case_id": args.case_id,
        "status": "started",
        "started_at": time.time(),
        "encoder": {
            "model": args.encoder_model,
            "dtype": args.encoder_dtype,
            "max_seq": args.encoder_max_seq,
            "batch": args.encoder_batch,
            "dim": args.encoder_dim,
        },
        "reranker": {"model": args.reranker_model, "batch": args.rerank_batch},
        "retrieve_top_k": args.retrieve_top_k,
        "rerank_candidates": args.rerank_candidates,
        "output_top_k": args.output_top_k,
        "embedding_mode": args.embedding_mode,
        "n_warm_runs": args.n_warm_runs,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        result["device"] = device
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        corpus_data = _load_jsonl(args.corpus)
        documents = _make_documents(corpus_data, args.embedding_mode)
        doc_texts = [doc.page_content for doc in documents]
        result["n_corpus_docs"] = len(documents)

        query_data = _load_jsonl(args.queries)
        qids = [str(q.get("id") or q.get("qid")) for q in query_data]
        query_texts = [str(q.get("question") or q.get("query") or "") for q in query_data]
        result["n_queries"] = len(query_data)
        gold = _load_gold(args.gold)

        cfg = PipelineConfig(
            embedding_model=args.encoder_model,
            embedding_dim=args.encoder_dim,
            embedding_mode=args.embedding_mode,
            embedding_batch_size=args.encoder_batch,
            embedding_max_seq_length=args.encoder_max_seq,
            use_fp16=args.encoder_dtype == "float16",
            device="auto",
            dense_top_k=args.retrieve_top_k,
            use_reranker=True,
            reranker_model=args.reranker_model,
            rerank_top_n=args.output_top_k,
        )

        t = time.perf_counter()
        embeddings = build_embeddings(cfg)
        _sync_cuda(device)
        result["encoder_load_sec"] = round(time.perf_counter() - t, 4)

        t = time.perf_counter()
        reranker = build_reranker(cfg)
        _sync_cuda(device)
        result["reranker_load_sec"] = round(time.perf_counter() - t, 4)

        def one_pass() -> tuple[dict[str, float], dict[str, list[str]], dict[str, list[str]]]:
            stage: dict[str, float] = {}
            t0 = time.perf_counter()
            doc_vectors = embeddings.embed_documents(doc_texts)
            _sync_cuda(device)
            stage["corpus_embed_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            store = _build_store_from_vectors(documents, embeddings, doc_vectors)
            stage["index_build_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            query_vectors = embeddings.embed_documents(query_texts)
            _sync_cuda(device)
            stage["query_embed_batch_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            dense_docs_by_qid: dict[str, list[Document]] = {}
            dense_predictions: dict[str, list[str]] = {}
            for qid, q_vec in zip(qids, query_vectors):
                hits = store.similarity_search_by_vector(q_vec, k=args.retrieve_top_k)
                dense_docs_by_qid[qid] = hits
                dense_predictions[qid] = [str(doc.metadata.get("doc_id") or "") for doc in hits]
            stage["retrieve_by_vector_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            reranked_predictions: dict[str, list[str]] = {}
            for qid, query in zip(qids, query_texts):
                candidates = dense_docs_by_qid[qid][: args.rerank_candidates]
                reranked = list(reranker.compress_documents(candidates, query))
                reranked_ids = [str(doc.metadata.get("doc_id") or "") for doc in reranked]
                tail_ids = dense_predictions[qid][args.rerank_candidates :]
                reranked_predictions[qid] = (reranked_ids + tail_ids)[: args.output_top_k]
            _sync_cuda(device)
            stage["rerank_sec"] = time.perf_counter() - t0

            stage["total_sec"] = sum(stage.values())
            stage["per_query_sec"] = (
                stage["query_embed_batch_sec"] + stage["retrieve_by_vector_sec"] + stage["rerank_sec"]
            ) / len(qids)
            return stage, dense_predictions, reranked_predictions

        t = time.perf_counter()
        _, _, _ = one_pass()
        result["warmup_total_sec"] = round(time.perf_counter() - t, 4)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        runs: list[dict[str, float]] = []
        final_dense: dict[str, list[str]] = {}
        final_reranked: dict[str, list[str]] = {}
        for _ in range(args.n_warm_runs):
            stage, dense, reranked = one_pass()
            runs.append({key: round(value, 4) for key, value in stage.items()})
            final_dense = dense
            final_reranked = reranked

        def median(key: str) -> float:
            values = sorted(run[key] for run in runs)
            return round(values[len(values) // 2], 4)

        result["warm_runs"] = runs
        result["warm_median"] = {
            "corpus_embed_sec": median("corpus_embed_sec"),
            "index_build_sec": median("index_build_sec"),
            "query_embed_batch_sec": median("query_embed_batch_sec"),
            "retrieve_by_vector_sec": median("retrieve_by_vector_sec"),
            "rerank_sec": median("rerank_sec"),
            "total_sec": median("total_sec"),
            "per_query_sec": median("per_query_sec"),
        }
        result["quality"] = {
            "dense": _compute_metrics(final_dense, gold),
            "reranked": _compute_metrics(final_reranked, gold),
        }
        if device == "cuda":
            result["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
        result["status"] = "completed"
        result["finished_at"] = time.time()
    except Exception as e:  # noqa: BLE001
        import traceback

        result["status"] = "error"
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback_tail"] = traceback.format_exc()[-3000:]
        result["finished_at"] = time.time()
    finally:
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
