"""Reranker harness: loads dense retrieval CSV/JSONL, applies cross-encoder, outputs ranked CSV."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "dragonkue/bge-reranker-v2-m3-ko",
]


def load_retrieval_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_retrieval_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        prev_qid = None
        current_hits: list[dict[str, Any]] = []
        for row in reader:
            qid = row.get("query_id", row.get("question_id", ""))
            if qid != prev_qid:
                if prev_qid is not None:
                    rows.append({"question_id": prev_qid, "hits": current_hits})
                current_hits = []
                prev_qid = qid
            current_hits.append(row)
        if prev_qid is not None:
            rows.append({"question_id": prev_qid, "hits": current_hits})
    return rows


def load_retrieval(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        return load_retrieval_csv(path)
    return load_retrieval_jsonl(path)


def rerank_results(
    retrieval_rows: list[dict[str, Any]],
    model_name: str,
    top_n: int,
    batch_size: int = 16,
    device: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)
    output_rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for row in retrieval_rows:
        qid = str(row.get("question_id", row.get("query_id", "")))
        query = str(row.get("query", row.get("question", "")))
        hits = row.get("hits", [])[:top_n]

        t0 = time.perf_counter()
        pairs = [
            [query, f"{h.get('title', '')}\n{h.get('abstract', '')}".strip()]
            for h in hits
        ]
        scores = model.predict(pairs, batch_size=batch_size)
        latency = time.perf_counter() - t0

        scored = sorted(
            zip(hits, scores), key=lambda x: float(x[1]), reverse=True
        )
        for new_rank, (hit, score) in enumerate(scored, start=1):
            output_rows.append(
                {
                    "query_id": qid,
                    "doc_id": hit.get("doc_id", hit.get("id", "")),
                    "dense_rank": hit.get("rank", ""),
                    "dense_score": hit.get("score", ""),
                    "rerank_score": float(score),
                    "new_rank": new_rank,
                    "rerank_latency_sec": round(latency, 4),
                }
            )

    total_latency = time.perf_counter() - total_start
    return output_rows, total_latency


def write_rerank_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "doc_id", "dense_rank", "dense_score", "rerank_score", "new_rank", "rerank_latency_sec"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-encoder reranker harness")
    parser.add_argument("--input", required=True, help="Dense retrieval JSONL or CSV path")
    parser.add_argument("--output", required=True, help="Output reranked CSV path")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3", choices=RERANKER_MODELS + ["any"])
    parser.add_argument("--top_n", type=int, default=50, help="Candidates to rerank per query")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    retrieval = load_retrieval(Path(args.input))
    rows, total_latency = rerank_results(
        retrieval, args.model, args.top_n, args.batch_size, args.device
    )
    write_rerank_csv(rows, Path(args.output))
    print(f"Reranked {len(retrieval)} queries in {total_latency:.2f}s -> {args.output}")


if __name__ == "__main__":
    main()
