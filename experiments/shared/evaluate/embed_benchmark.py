from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.shared.evaluate.metrics_ir import summarize_ir_metrics
from experiments.shared.query_transform.factory import create_transform
from experiments.shared.rerank.cross_encoder import CrossEncoderReranker
from experiments.shared.retrievers.bm25 import BM25Retriever
from src.build_vectordb_search import build_vectordb_search
from src.retrieval_system import data_loader, query_encoder
from src.retrieval_system.retrievers import get_retriever


@dataclass
class BenchmarkCase:
    case_id: str
    encoder_config: Path | None
    retriever: str = "dense"
    query_transform: str = "raw"
    rerank_model: str | None = None
    top_k: int = 50
    query_instruction: str | None = None
    bm25_tokenizer: str = "whitespace"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_gold(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): {str(item).strip() for item in value} for key, value in payload.items()}


def _case_from_dict(item: dict[str, Any]) -> BenchmarkCase:
    encoder_config = item.get("encoder_config")
    return BenchmarkCase(
        case_id=item["case_id"],
        encoder_config=Path(encoder_config) if encoder_config else None,
        retriever=item.get("retriever", "dense"),
        query_transform=item.get("query_transform", "raw"),
        rerank_model=item.get("rerank_model"),
        top_k=int(item.get("top_k", 50)),
        query_instruction=item.get("query_instruction"),
        bm25_tokenizer=item.get("bm25_tokenizer", "whitespace"),
    )


def _prepare_temp_config(original_config: Path, case_dir: Path) -> Path:
    payload = json.loads(original_config.read_text(encoding="utf-8"))
    payload["output_dir"] = str(case_dir)
    payload["output_file"] = str(case_dir / "vectordb.csv")
    temp_config = case_dir / "encoder_config.json"
    temp_config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return temp_config


def _join_keywords(query_row: dict[str, Any]) -> str:
    search_terms = query_row.get("search_terms") or []
    if search_terms:
        return " ".join(str(term).replace("|", " ") for term in search_terms)
    keywords = query_row.get("keywords") or {}
    flat = list(keywords.get("english", [])) + list(keywords.get("korean", []))
    return " ".join(flat) if flat else str(query_row.get("query") or "")


def _dense_search_for_case(
    case: BenchmarkCase,
    corpus_jsonl: Path,
    queries: list[dict[str, Any]],
    case_dir: Path,
    hyde_kwargs: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if case.encoder_config is None:
        raise ValueError(f"Dense case requires encoder_config: {case.case_id}")
    temp_config = _prepare_temp_config(case.encoder_config, case_dir)
    build_vectordb_search(
        config_path=str(temp_config),
        data_schema="configs/csv_schema/test_2.json",
        docs_jsonl_path=str(corpus_jsonl),
        auto_data_load=False,
        gpu_id=None,
    )
    config = json.loads(temp_config.read_text(encoding="utf-8"))
    vectordb = data_loader.load_vectordb_from_csv(config["output_file"], "configs/csv_schema/test_2.json")
    encoder = query_encoder.QueryEncoder(model_name=config["model_name"], device="auto")
    retriever = get_retriever(vectordb.embeddings)

    transform = create_transform(case.query_transform, **hyde_kwargs) if case.query_transform.startswith("hyde") else None
    reranker = CrossEncoderReranker(case.rerank_model) if case.rerank_model else None
    retrieval_by_qid: dict[str, list[dict[str, Any]]] = {}

    for row in queries:
        qid = str(row.get("question_id") or row.get("id"))
        base_query = str(row.get("query") or row.get("question") or "")
        if case.query_transform == "keywords":
            query_texts = [_join_keywords(row)]
        elif case.query_transform.startswith("hyde"):
            query_texts = transform.transform(base_query)
        else:
            query_texts = [base_query]

        if case.query_transform.startswith("hyde_union_"):
            merged: dict[str, dict[str, Any]] = {}
            for query_text in query_texts:
                query_vec = encoder.encode_queries([query_text], instruction=case.query_instruction)
                scores, indices = retriever.search(query_vec, top_k=case.top_k)
                for score, idx in zip(scores[0], indices[0]):
                    doc_id = vectordb.doc_ids[int(idx)]
                    current = merged.get(doc_id)
                    candidate = {
                        "doc_id": doc_id,
                        "score": float(score),
                        **vectordb.metadata[int(idx)],
                    }
                    if current is None or candidate["score"] > current["score"]:
                        merged[doc_id] = candidate
            hits = sorted(merged.values(), key=lambda item: item["score"], reverse=True)[: case.top_k]
        else:
            query_vecs = encoder.encode_queries(query_texts, instruction=case.query_instruction)
            if case.query_transform.startswith("hyde_mean_") and len(query_texts) > 1:
                query_vecs = query_vecs.mean(axis=0, keepdims=True)
            scores, indices = retriever.search(query_vecs, top_k=case.top_k)
            hits = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                hits.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "doc_id": vectordb.doc_ids[int(idx)],
                        **vectordb.metadata[int(idx)],
                    }
                )

        if reranker is not None:
            hits = reranker.rerank(base_query, hits, top_k=min(5, len(hits)))
        retrieval_by_qid[qid] = hits
    return retrieval_by_qid


def _bm25_search_for_case(case: BenchmarkCase, corpus_jsonl: Path, queries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    corpus_docs = []
    with corpus_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus_docs.append(json.loads(line))
    retriever = BM25Retriever(tokenizer=case.bm25_tokenizer)
    corpus_texts = [
        f"{doc.get('title', '')} {doc.get('abstract', '')}".strip()
        for doc in corpus_docs
    ]
    retriever.build(corpus_texts)
    retrieval_by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in queries:
        qid = str(row.get("question_id") or row.get("id"))
        query_text = _join_keywords(row) if case.query_transform == "keywords" else str(row.get("query") or row.get("question") or "")
        scores, indices = retriever.search([query_text], top_k=case.top_k)
        hits = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            doc = corpus_docs[int(idx)]
            hits.append({"rank": rank, "score": float(score), **doc})
        retrieval_by_qid[qid] = hits
    return retrieval_by_qid


def _write_case_outputs(
    case: BenchmarkCase,
    case_dir: Path,
    retrieval_by_qid: dict[str, list[dict[str, Any]]],
    gold_by_qid: dict[str, set[str]],
    runtime_sec: float,
) -> dict[str, Any]:
    ranked_doc_ids = {
        qid: [hit.get("doc_id", "") for hit in hits]
        for qid, hits in retrieval_by_qid.items()
    }
    metrics = summarize_ir_metrics(ranked_doc_ids, gold_by_qid)
    metrics["case_id"] = case.case_id
    metrics["runtime_sec"] = round(runtime_sec, 4)
    (case_dir / "retrieval.jsonl").write_text(
        "\n".join(
            json.dumps({"question_id": qid, "hits": hits}, ensure_ascii=False)
            for qid, hits in retrieval_by_qid.items()
        ),
        encoding="utf-8",
    )
    (case_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def run_benchmark(
    cases: list[BenchmarkCase],
    corpus_jsonl: Path,
    queries_jsonl: Path,
    gold_path: Path,
    output_root: Path,
    hyde_kwargs: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus_jsonl, output_root / "corpus.jsonl")
    shutil.copy2(queries_jsonl, output_root / "queries.jsonl")
    shutil.copy2(gold_path, output_root / "gold.json")
    queries = _load_queries(queries_jsonl)
    gold_by_qid = _load_gold(gold_path)
    reports = []
    for case in cases:
        case_dir = output_root / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if case.retriever == "bm25":
            retrieval_by_qid = _bm25_search_for_case(case, corpus_jsonl, queries)
        else:
            retrieval_by_qid = _dense_search_for_case(case, corpus_jsonl, queries, case_dir, hyde_kwargs)
        metrics = _write_case_outputs(case, case_dir, retrieval_by_qid, gold_by_qid, time.perf_counter() - started)
        reports.append(metrics)
    lines = ["# Benchmark Report", "", "| case | R@5 | R@10 | MRR@10 | nDCG@10 | runtime sec |", "|---|---:|---:|---:|---:|---:|"]
    for report in reports:
        lines.append(
            f"| {report['case_id']} | {report.get('recall_at_5', 0)} | {report.get('recall_at_10', 0)} | "
            f"{report.get('mrr_at_10', 0)} | {report.get('ndcg_at_10', 0)} | {report.get('runtime_sec', 0)} |"
        )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-corpus retrieval benchmark harness")
    parser.add_argument("--cases", required=True, help="YAML case definition path")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL path")
    parser.add_argument("--queries", required=True, help="Frozen queries JSONL path")
    parser.add_argument("--gold", required=True, help="Gold JSON path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--hyde-backend", choices=["vllm", "gemini"], default="vllm")
    parser.add_argument("--hyde-model", default="openai/gpt-oss-20b")
    parser.add_argument("--hyde-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--hyde-api-key", default=None)
    parser.add_argument("--hyde-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = _load_yaml(Path(args.cases))
    cases = [_case_from_dict(item) for item in payload.get("cases", [])]
    run_benchmark(
        cases=cases,
        corpus_jsonl=Path(args.corpus),
        queries_jsonl=Path(args.queries),
        gold_path=Path(args.gold),
        output_root=Path(args.output),
        hyde_kwargs={
            "backend": args.hyde_backend,
            "model": args.hyde_model,
            "base_url": args.hyde_base_url,
            "api_key": args.hyde_api_key,
            "seed": args.hyde_seed,
        },
    )


if __name__ == "__main__":
    main()
