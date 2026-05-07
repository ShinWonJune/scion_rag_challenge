from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from shrag.retrieval import data_loader
from shrag.retrieval.retrievers import get_retriever


def _benchmark(
    embeddings: np.ndarray,
    query_vecs: np.ndarray,
    device: str,
    top_ks: list[int],
    repeats: int,
    temp_memory_gb: float,
) -> dict:
    os.environ["SHRAG_FAISS_DEVICE"] = device
    os.environ["SHRAG_FAISS_GPU_TEMP_MEMORY_GB"] = str(temp_memory_gb)

    started = time.perf_counter()
    retriever = get_retriever(embeddings)
    build_sec = time.perf_counter() - started

    rows = []
    for top_k in top_ks:
        retriever.search(query_vecs[:1], top_k=top_k)
        started = time.perf_counter()
        for _ in range(repeats):
            scores, indices = retriever.search(query_vecs, top_k=top_k)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "top_k": top_k,
                "repeats": repeats,
                "total_sec": round(elapsed, 6),
                "sec_per_repeat": round(elapsed / repeats, 6),
                "sec_per_query": round(elapsed / repeats / len(query_vecs), 8),
                "first_query_indices": [int(x) for x in indices[0][: min(top_k, 10)]],
                "first_query_scores": [float(x) for x in scores[0][: min(top_k, 10)]],
            }
        )

    return {
        "device": device,
        "build_sec": round(build_sec, 6),
        "n_docs": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "n_queries": int(query_vecs.shape[0]),
        "temp_memory_gb": temp_memory_gb if device == "gpu" else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FAISS CPU/GPU search speed.")
    parser.add_argument("--vectordb", required=True)
    parser.add_argument("--schema", default="configs/csv_schema/test_2.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[5, 7, 10, 20, 50])
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--n-queries", type=int, default=41)
    parser.add_argument("--gpu-temp-memory-gb", type=float, default=8.0)
    args = parser.parse_args()

    vectordb = data_loader.load_vectordb_from_csv(args.vectordb, args.schema)
    n_queries = min(args.n_queries, vectordb.embeddings.shape[0])
    query_vecs = np.ascontiguousarray(vectordb.embeddings[:n_queries].astype(np.float32))

    results = {
        "cpu": _benchmark(
            vectordb.embeddings,
            query_vecs,
            "cpu",
            args.top_ks,
            args.repeats,
            args.gpu_temp_memory_gb,
        ),
        "gpu": _benchmark(
            vectordb.embeddings,
            query_vecs,
            "gpu",
            args.top_ks,
            args.repeats,
            args.gpu_temp_memory_gb,
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
