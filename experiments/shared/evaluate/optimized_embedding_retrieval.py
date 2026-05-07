from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

from experiments.shared.evaluate.metrics_ir import summarize_ir_metrics
from experiments.shared.evaluate.retrieval_eval import evaluate as retrieval_evaluate
from shrag.retrieval import utils


DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _dtype(name: str | None) -> Any | None:
    if not name:
        return None
    value = str(name).lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_gold(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): {str(item).strip() for item in value} for key, value in payload.items()}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def _topk(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = query_embeddings @ doc_embeddings.T
    k = min(top_k, doc_embeddings.shape[0])
    indices = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(sims, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(scores, order, axis=1), np.take_along_axis(indices, order, axis=1)


def _load_model(case: dict[str, Any], device: str) -> SentenceTransformer:
    model_kwargs: dict[str, Any] = {}
    dtype_value = _dtype(case.get("torch_dtype"))
    if dtype_value is not None:
        model_kwargs["torch_dtype"] = dtype_value
    if case.get("attn_implementation"):
        model_kwargs["attn_implementation"] = str(case["attn_implementation"])

    model = SentenceTransformer(
        case["model_name"],
        trust_remote_code=True,
        local_files_only=True,
        truncate_dim=int(case["embedding_dim"]),
        device=device,
        model_kwargs=model_kwargs or None,
    )
    if case.get("max_seq_length"):
        model.max_seq_length = int(case["max_seq_length"])
    return model


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exp16. Optimized Embedding Retrieval Metrics",
        "",
        "Evaluates retrieval quality for Exp15 optimized document embeddings.",
        "",
        f"- corpus: `{summary['corpus']}`",
        f"- queries: `{summary['queries']}`",
        f"- gold: `{summary['gold']}`",
        f"- docs: `{summary['doc_count']}`",
        f"- queries: `{summary['query_count']}`",
        f"- device: `{summary['device']}`",
        "",
        "| case | model | max_len | batch | Hit@1 | Hit@3 | Hit@5 | Hit@7 | Hit@10 | MRR@10 | mean_gold_rank | query_encode_sec | retrieve_sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cases"]:
        lines.append(
            f"| {row['case_id']}"
            f" | `{row['model_name']}`"
            f" | {row.get('effective_max_seq_length', '-')}"
            f" | {row.get('batch_size', '-')}"
            f" | {row.get('hit_at_1', '-')}"
            f" | {row.get('hit_at_3', '-')}"
            f" | {row.get('hit_at_5', '-')}"
            f" | {row.get('hit_at_7', '-')}"
            f" | {row.get('hit_at_10', '-')}"
            f" | {row.get('mrr_at_10', '-')}"
            f" | {row.get('mean_gold_rank', '-')}"
            f" | {row.get('query_encode_sec', '-')}"
            f" | {row.get('retrieve_sec', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Document embeddings are loaded from Exp15 `.npy` outputs and L2-normalized before search.",
            "- Queries are encoded with the same model dtype/max length/truncate dimension used by each case.",
            "- Search uses NumPy cosine similarity over normalized vectors to avoid FAISS device differences in metric evaluation.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    corpus = _load_jsonl(Path(args.corpus))
    queries = _load_jsonl(Path(args.queries))
    gold_by_qid = _load_gold(Path(args.gold))
    doc_ids = [str(row.get("doc_id") or row.get("CN") or row.get("cn") or "") for row in corpus]
    query_texts = [str(row.get("query") or row.get("question") or "") for row in queries]
    query_ids = [str(row.get("question_id") or row.get("id") or row.get("qid") or "") for row in queries]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary: dict[str, Any] = {
        "experiment": "exp16_optimized_embedding_retrieval",
        "corpus": args.corpus,
        "queries": args.queries,
        "gold": args.gold,
        "doc_count": len(corpus),
        "query_count": len(queries),
        "device": device,
        "torch_version": torch.__version__,
        "cases": [],
    }

    for case in _load_cases(Path(args.cases)):
        case_dir = output / "cases" / str(case["case_id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "case_id": str(case["case_id"]),
            "model_name": str(case["model_name"]),
            "embedding_dim": int(case["embedding_dim"]),
            "batch_size": int(case.get("batch_size", 32)),
            "torch_dtype": case.get("torch_dtype"),
            "max_seq_length": case.get("max_seq_length"),
            "top_k": int(case.get("top_k", 50)),
            "doc_embeddings": str(case["doc_embeddings"]),
            "status": "started",
        }
        started = time.perf_counter()
        try:
            doc_embeddings = np.load(case["doc_embeddings"]).astype(np.float32, copy=False)
            if doc_embeddings.shape[0] != len(doc_ids):
                raise ValueError(f"doc count mismatch: embeddings={doc_embeddings.shape[0]} corpus={len(doc_ids)}")
            if doc_embeddings.shape[1] != row["embedding_dim"]:
                raise ValueError(f"dim mismatch: embeddings={doc_embeddings.shape[1]} expected={row['embedding_dim']}")
            doc_embeddings = utils.l2_normalize(doc_embeddings)

            model = _load_model(case, device)
            row["effective_max_seq_length"] = int(getattr(model, "max_seq_length", 0) or 0)
            prefixed = [DEFAULT_QUERY_INSTRUCTION + text for text in query_texts]
            query_started = time.perf_counter()
            query_embeddings = model.encode(
                prefixed,
                batch_size=row["batch_size"],
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            row["query_encode_sec"] = round(time.perf_counter() - query_started, 4)
            query_embeddings = utils.l2_normalize(query_embeddings)

            retrieve_started = time.perf_counter()
            scores, indices = _topk(doc_embeddings, query_embeddings, row["top_k"])
            row["retrieve_sec"] = round(time.perf_counter() - retrieve_started, 4)

            retrieval_by_qid: dict[str, list[dict[str, Any]]] = {}
            ranked_doc_ids: dict[str, list[str]] = {}
            for q_idx, qid in enumerate(query_ids):
                hits: list[dict[str, Any]] = []
                ids: list[str] = []
                for rank, (score, doc_idx) in enumerate(zip(scores[q_idx], indices[q_idx]), start=1):
                    doc = corpus[int(doc_idx)]
                    doc_id = doc_ids[int(doc_idx)]
                    ids.append(doc_id)
                    hits.append({"rank": rank, "score": float(score), "doc_id": doc_id, **doc})
                retrieval_by_qid[qid] = hits
                ranked_doc_ids[qid] = ids

            metrics = summarize_ir_metrics(ranked_doc_ids, gold_by_qid)
            hit_metrics = retrieval_evaluate(ranked_doc_ids, gold_by_qid, ks=[1, 3, 5, 7, 10, 20])
            row.update(metrics)
            for k in [1, 3, 5, 7, 10, 20]:
                row[f"hit_at_{k}"] = hit_metrics.get(f"hit_at_{k}", 0.0)
            row["mean_gold_rank"] = hit_metrics.get("mean_gold_rank")
            row["gold_found_rate"] = hit_metrics.get("gold_found_rate", 0.0)
            row["runtime_sec"] = round(time.perf_counter() - started, 4)
            row["status"] = "completed"
            (case_dir / "retrieval.jsonl").write_text(
                "\n".join(
                    json.dumps({"question_id": qid, "hits": hits}, ensure_ascii=False)
                    for qid, hits in retrieval_by_qid.items()
                ),
                encoding="utf-8",
            )
            (case_dir / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            row["runtime_sec"] = round(time.perf_counter() - started, 4)
        finally:
            summary["cases"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_report(summary, output / "report.md")
            try:
                del model  # type: ignore[name-defined]
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics for optimized embeddings.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
