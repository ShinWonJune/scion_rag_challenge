from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.shared.evaluate.metrics_ir import summarize_ir_metrics
from experiments.shared.evaluate.retrieval_eval import evaluate as retrieval_evaluate
from experiments.shared.rerank.cross_encoder import CrossEncoderReranker
from experiments.shared.retrievers.bm25 import BM25Retriever
from shrag.pipeline._impl.build_vectordb import build_vectordb_search
from shrag.retrieval import data_loader, query_encoder
from shrag.retrieval.retrievers import get_retriever


@dataclass
class BenchmarkCase:
    case_id: str
    encoder_config: Path | None
    retriever: str = "dense"
    rerank_model: str | None = None
    top_k: int = 50
    rerank_output_top_k: int = 5
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
        rerank_model=item.get("rerank_model"),
        top_k=int(item.get("top_k", 50)),
        rerank_output_top_k=int(item.get("rerank_output_top_k", 5)),
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


def _dense_search_for_case(
    case: BenchmarkCase,
    corpus_jsonl: Path,
    queries: list[dict[str, Any]],
    case_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], float, float]:
    if case.encoder_config is None:
        raise ValueError(f"Dense case requires encoder_config: {case.case_id}")
    temp_config = _prepare_temp_config(case.encoder_config, case_dir)
    step3_started = time.perf_counter()
    build_vectordb_search(
        config_path=str(temp_config),
        data_schema="configs/csv_schema/test_2.json",
        docs_jsonl_path=str(corpus_jsonl),
        auto_data_load=False,
        gpu_id=None,
    )
    step3_sec = time.perf_counter() - step3_started

    step4_started = time.perf_counter()
    config = json.loads(temp_config.read_text(encoding="utf-8"))
    vectordb = data_loader.load_vectordb_from_csv(config["output_file"], "configs/csv_schema/test_2.json")
    encoder = query_encoder.QueryEncoder(model_name=config["model_name"], device="auto")
    retriever = get_retriever(vectordb.embeddings)

    reranker = CrossEncoderReranker(case.rerank_model) if case.rerank_model else None
    retrieval_by_qid: dict[str, list[dict[str, Any]]] = {}

    for row in queries:
        qid = str(row.get("question_id") or row.get("id"))
        base_query = str(row.get("query") or row.get("question") or "")
        query_vec = encoder.encode_queries([base_query], instruction=case.query_instruction)
        scores, indices = retriever.search(query_vec, top_k=case.top_k)
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
            hits = reranker.rerank(base_query, hits, top_k=min(case.rerank_output_top_k, len(hits)))
        retrieval_by_qid[qid] = hits
    step4_sec = time.perf_counter() - step4_started
    return retrieval_by_qid, step3_sec, step4_sec


def _bm25_search_for_case(
    case: BenchmarkCase, corpus_jsonl: Path, queries: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
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
        query_text = str(row.get("query") or row.get("question") or "")
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
    step3_sec: float | None = None,
    step4_sec: float | None = None,
) -> dict[str, Any]:
    ranked_doc_ids = {
        qid: [hit.get("doc_id", "") for hit in hits]
        for qid, hits in retrieval_by_qid.items()
    }
    metrics = summarize_ir_metrics(ranked_doc_ids, gold_by_qid)
    # Add Hit@k and mean_gold_rank via retrieval_evaluate
    hit_metrics = retrieval_evaluate(ranked_doc_ids, gold_by_qid, ks=[1, 3, 5, 7, 10, 20])
    for k in [1, 3, 5, 7, 10, 20]:
        metrics[f"hit_at_{k}"] = hit_metrics.get(f"hit_at_{k}", 0.0)
    metrics["mean_gold_rank"] = hit_metrics.get("mean_gold_rank")
    metrics["gold_found_rate"] = hit_metrics.get("gold_found_rate", 0.0)
    metrics["case_id"] = case.case_id
    metrics["runtime_sec"] = round(runtime_sec, 4)
    if step3_sec is not None:
        metrics["step3_build_vectordb_sec"] = round(step3_sec, 4)
    if step4_sec is not None:
        metrics["step4_retrieve_sec"] = round(step4_sec, 4)
    (case_dir / "retrieval.jsonl").write_text(
        "\n".join(
            json.dumps({"question_id": qid, "hits": hits}, ensure_ascii=False)
            for qid, hits in retrieval_by_qid.items()
        ),
        encoding="utf-8",
    )
    (case_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def run_benchmark(
    cases: list[BenchmarkCase],
    corpus_jsonl: Path,
    queries_jsonl: Path,
    gold_path: Path,
    output_root: Path,
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
        step3_sec = step4_sec = None
        if case.retriever == "bm25":
            retrieval_by_qid = _bm25_search_for_case(case, corpus_jsonl, queries)
        else:
            retrieval_by_qid, step3_sec, step4_sec = _dense_search_for_case(
                case, corpus_jsonl, queries, case_dir
            )
        metrics = _write_case_outputs(
            case, case_dir, retrieval_by_qid, gold_by_qid,
            time.perf_counter() - started, step3_sec, step4_sec,
        )
        reports.append(metrics)
    lines = [
        "# Benchmark Report",
        "",
        "| case | Hit@5 | Hit@7 | Hit@10 | R@5 | R@7 | R@10 | MRR@10 | nDCG@10 | mean_gold_rank | step3_sec | step4_sec | runtime_sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['case_id']}"
            f" | {report.get('hit_at_5', 0)}"
            f" | {report.get('hit_at_7', 0)}"
            f" | {report.get('hit_at_10', 0)}"
            f" | {report.get('recall_at_5', 0)}"
            f" | {report.get('recall_at_7', 0)}"
            f" | {report.get('recall_at_10', 0)}"
            f" | {report.get('mrr_at_10', 0)}"
            f" | {report.get('ndcg_at_10', 0)}"
            f" | {report.get('mean_gold_rank', '-')}"
            f" | {report.get('step3_build_vectordb_sec', '-')}"
            f" | {report.get('step4_retrieve_sec', '-')}"
            f" | {report.get('runtime_sec', 0)} |"
        )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-corpus retrieval benchmark harness")
    parser.add_argument("--cases", required=True, help="YAML case definition path")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL path")
    parser.add_argument("--queries", required=True, help="Frozen queries JSONL path")
    parser.add_argument("--gold", required=True, help="Gold JSON path")
    parser.add_argument("--output", required=True, help="Output directory")
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
    )


if __name__ == "__main__":
    main()
