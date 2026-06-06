from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from shrag_lc.config import PipelineConfig, Settings
from shrag_lc.embeddings import doc_to_document
from shrag_lc.generate import AnswerGenerator
from shrag_lc.llm import build_generation_model
from shrag_lc.pipeline import SHRAGPipeline
from shrag_lc.rerank import CrossEncoderRerankCompressor
from shrag_lc.search.acquire import _request_stats, _search_until_target, _stats_delta
from shrag_lc.search.clients import _dedup
from shrag_lc.search.terms import build_search_terms_by_lang
from shrag_lc.vectorstore import build_faiss


def _load_questions(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question") or item.get("query") or item.get("text") or "")
            if not question:
                continue
            rows.append({"id": str(item.get("id") or item.get("question_id") or idx), "question": question})
    return rows


def _sync_cuda(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


def _device_name() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _timed_acquire(pipeline: SHRAGPipeline, query: str) -> tuple[list, dict[str, Any]]:
    acquirer = pipeline.acquirer
    cfg = pipeline.cfg

    started_total = time.perf_counter()
    started = time.perf_counter()
    try:
        keywords = acquirer.extractor.extract(query)
        keyword_status = "success"
        keyword_error = None
    except Exception as exc:  # noqa: BLE001
        keywords = {"korean": [], "english": [query]}
        keyword_status = "fallback"
        keyword_error = str(exc)
    keyword_extract_sec = time.perf_counter() - started

    started = time.perf_counter()
    terms_by_lang = build_search_terms_by_lang(keywords, cfg.number_of_operators)
    ko_terms = terms_by_lang.get("korean") or []
    en_terms = terms_by_lang.get("english") or []
    stats_before = _request_stats(acquirer.client)
    ko_result = _search_until_target(acquirer.client, ko_terms, cfg.target_documents, "korean")
    en_result = _search_until_target(acquirer.client, en_terms, cfg.target_documents, "english")
    stats_after = _request_stats(acquirer.client)
    platform_search_sec = time.perf_counter() - started

    merged = _dedup(ko_result.documents + en_result.documents, key="doc_id")
    documents = [doc_to_document(doc, cfg.embedding_mode) for doc in merged]
    request_stats_delta = _stats_delta(stats_before, stats_after)
    meta = {
        "keyword_status": keyword_status,
        "keyword_error": keyword_error,
        "keywords": keywords,
        "search_terms": ko_terms + en_terms,
        "document_count": len(documents),
        "document_count_by_lang": {"korean": len(ko_result.documents), "english": len(en_result.documents)},
        "failed_terms": ko_result.failed_terms + en_result.failed_terms,
        "keyword_extract_sec": keyword_extract_sec,
        "platform_search_sec": platform_search_sec,
        "search_total_sec": time.perf_counter() - started_total,
        "request_stats": stats_after,
        "request_stats_delta": request_stats_delta,
    }
    return documents, meta


def _timed_answer(pipeline: SHRAGPipeline, query: str, documents: list, device: str) -> dict[str, Any]:
    if not documents:
        return {
            "index_build_sec": 0.0,
            "retrieve_sec": 0.0,
            "rerank_sec": 0.0,
            "generation_sec": 0.0,
            "context_count": 0,
            "answer_chars": 0,
            "used_context": [],
            "status": "no_documents",
        }

    started = time.perf_counter()
    store = build_faiss(documents, pipeline.embeddings)
    _sync_cuda(device)
    index_build_sec = time.perf_counter() - started

    started = time.perf_counter()
    retriever = store.as_retriever(search_kwargs={"k": pipeline.cfg.dense_top_k})
    retrieved = retriever.invoke(query)
    _sync_cuda(device)
    retrieve_sec = time.perf_counter() - started

    started = time.perf_counter()
    if pipeline.reranker is not None:
        reranked = list(pipeline.reranker.compress_documents(retrieved, query))
    else:
        reranked = retrieved[: pipeline.cfg.rerank_top_n]
    _sync_cuda(device)
    rerank_sec = time.perf_counter() - started

    started = time.perf_counter()
    generated = pipeline.generator.generate(query, reranked)
    _sync_cuda(device)
    generation_sec = time.perf_counter() - started

    return {
        "index_build_sec": index_build_sec,
        "retrieve_sec": retrieve_sec,
        "rerank_sec": rerank_sec,
        "generation_sec": generation_sec,
        "context_count": len(reranked),
        "answer_chars": len(generated.get("answer") or ""),
        "used_context": generated.get("used_context", []),
        "status": "success",
    }


def _run_query(pipeline: SHRAGPipeline, query_item: dict[str, str], *, phase: str, device: str) -> dict[str, Any]:
    query = query_item["question"]
    started = time.perf_counter()
    documents, search_meta = _timed_acquire(pipeline, query)
    answer_meta = _timed_answer(pipeline, query, documents, device)
    total_sec = time.perf_counter() - started
    record = {
        "phase": phase,
        "id": query_item["id"],
        "question": query,
        "total_sec": round(total_sec, 4),
        "keyword_extract_sec": round(search_meta["keyword_extract_sec"], 4),
        "platform_search_sec": round(search_meta["platform_search_sec"], 4),
        "search_total_sec": round(search_meta["search_total_sec"], 4),
        "index_build_sec": round(answer_meta["index_build_sec"], 4),
        "retrieve_sec": round(answer_meta["retrieve_sec"], 4),
        "rerank_sec": round(answer_meta["rerank_sec"], 4),
        "generation_sec": round(answer_meta["generation_sec"], 4),
        "document_count": search_meta["document_count"],
        "document_count_by_lang": search_meta["document_count_by_lang"],
        "context_count": answer_meta["context_count"],
        "answer_chars": answer_meta["answer_chars"],
        "used_context": answer_meta["used_context"],
        "request_stats_delta": search_meta["request_stats_delta"],
        "rate_limit_count": int(search_meta["request_stats_delta"].get("rate_limit_count", 0) or 0),
        "api_error_count": int(search_meta["request_stats_delta"].get("api_error_count", 0) or 0),
        "status": answer_meta["status"],
    }
    return record


def _make_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        source="scienceon",
        target_documents=50,
        keyword_lang="all",
        number_of_operators=0,
        scienceon_max_pages=3,
        scienceon_max_concurrency=args.scienceon_max_concurrency,
        scienceon_min_interval_sec=args.scienceon_min_interval_sec,
        scienceon_fixed_concurrency=args.scienceon_fixed_concurrency,
        scienceon_max_retries=args.scienceon_max_retries,
        scienceon_retry_base_sleep_sec=args.scienceon_retry_base_sleep_sec,
        scienceon_retry_max_sleep_sec=args.scienceon_retry_max_sleep_sec,
        cache_root=args.cache_root,
        disable_cache=args.no_cache,
        embedding_model="Alibaba-NLP/gte-multilingual-base",
        embedding_dim=768,
        embedding_mode="3*title+abstract",
        embedding_batch_size=32,
        embedding_max_seq_length=512,
        use_fp16=True,
        device="auto",
        dense_top_k=5,
        use_reranker=True,
        reranker_model="dragonkue/bge-reranker-v2-m3-ko",
        rerank_top_n=3,
        llm_backend="vllm",
        llm_model=args.vllm_model,
        max_answer_tokens=args.max_answer_tokens,
        temperature=0.0,
        extractor_backend="vllm",
        extractor_model=args.vllm_model,
    )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "total_sec",
        "keyword_extract_sec",
        "platform_search_sec",
        "search_total_sec",
        "index_build_sec",
        "retrieve_sec",
        "rerank_sec",
        "generation_sec",
        "document_count",
        "rate_limit_count",
    ]
    out: dict[str, Any] = {"n": len(records)}
    for field in fields:
        vals = [row[field] for row in records if row.get(field) is not None]
        if vals:
            out[f"mean_{field}"] = round(statistics.mean(vals), 4)
            out[f"median_{field}"] = round(statistics.median(vals), 4)
            out[f"max_{field}"] = round(max(vals), 4)
    return out


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    measured = report["measured_records"]
    summary = report["summary"]
    lines = [
        "# Warm E2E Search-to-Answer Latency",
        "",
        "This experiment warms the full SHRAG path once with a separate warmup query, then measures five random queries from search through answer generation.",
        "",
        "## Configuration",
        "",
        f"- Query file: `{report['queries_path']}`",
        f"- Random seed: `{report['seed']}`",
        f"- Warmup qid: `{report['warmup_record']['id']}`",
        f"- Measured qids: `{', '.join(row['id'] for row in measured)}`",
        f"- vLLM URL: `{report['vllm_url']}`",
        f"- vLLM model: `{report['vllm_model']}`",
        f"- Search: `ScienceON k=30, n=10, m=50`, cache disabled: `{report['no_cache']}`",
        f"- Encoder: `Alibaba-NLP/gte-multilingual-base`, fp16, batch=32, max_seq_length=512, `3*title+abstract`",
        f"- Retrieval/rerank: dense top-5, `dragonkue/bge-reranker-v2-m3-ko`, top-3 context",
        f"- Throttle sweep policy: fixed concurrency/interval candidates, triggered only after observed 429",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean total sec/query | {summary.get('mean_total_sec', 0)} |",
        f"| mean keyword extract sec | {summary.get('mean_keyword_extract_sec', 0)} |",
        f"| mean platform search sec | {summary.get('mean_platform_search_sec', 0)} |",
        f"| mean index build sec | {summary.get('mean_index_build_sec', 0)} |",
        f"| mean retrieve sec | {summary.get('mean_retrieve_sec', 0)} |",
        f"| mean rerank sec | {summary.get('mean_rerank_sec', 0)} |",
        f"| mean generation sec | {summary.get('mean_generation_sec', 0)} |",
        f"| total 429 count | {sum(row['rate_limit_count'] for row in measured)} |",
        "",
        "## Per Query",
        "",
        "| qid | total | keyword | search | index build | retrieve | rerank | generate | docs | 429 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in measured:
        lines.append(
            f"| {row['id']}"
            f" | {row['total_sec']}"
            f" | {row['keyword_extract_sec']}"
            f" | {row['platform_search_sec']}"
            f" | {row['index_build_sec']}"
            f" | {row['retrieve_sec']}"
            f" | {row['rerank_sec']}"
            f" | {row['generation_sec']}"
            f" | {row['document_count']}"
            f" | {row['rate_limit_count']} |"
        )
    if report.get("throttle_sweep"):
        lines.extend(["", "## 429 Throttle Sweep", ""])
        lines.extend([
            "| concurrency | min interval sec | fixed | total 429 | mean search sec | result |",
            "|---:|---:|---|---:|---:|---|",
        ])
        for row in report["throttle_sweep"]["candidates"]:
            lines.append(
                f"| {row['max_concurrency']}"
                f" | {row['min_interval_sec']}"
                f" | true"
                f" | {row['rate_limit_count']}"
                f" | {row.get('mean_platform_search_sec', 0)}"
                f" | {row['status']} |"
            )
        best = report["throttle_sweep"].get("fastest_no_429")
        if best:
            lines.extend([
                "",
                f"Fastest no-429 candidate: `concurrency={best['max_concurrency']}`, `min_interval_sec={best['min_interval_sec']}`.",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _new_pipeline(args: argparse.Namespace, *, max_concurrency: int | None = None, min_interval: float | None = None, fixed: bool | None = None) -> SHRAGPipeline:
    cfg = _make_config(args)
    if max_concurrency is not None:
        cfg.scienceon_max_concurrency = max_concurrency
    if min_interval is not None:
        cfg.scienceon_min_interval_sec = min_interval
    if fixed is not None:
        cfg.scienceon_fixed_concurrency = fixed
    settings = Settings(
        vllm_base_url=args.vllm_url,
        vllm_model=args.vllm_model,
        scienceon_credentials_path=args.scienceon_credentials,
    )
    return SHRAGPipeline(cfg, settings)


def _run_throttle_sweep(args: argparse.Namespace, queries: list[dict[str, str]], device: str) -> dict[str, Any]:
    candidates = [
        (2, 0.5),
        (1, 0.5),
        (1, 1.0),
        (1, 2.0),
        (1, 3.0),
        (1, 5.0),
    ]
    rows: list[dict[str, Any]] = []
    for max_concurrency, interval in candidates:
        pipeline = _new_pipeline(args, max_concurrency=max_concurrency, min_interval=interval, fixed=True)
        records = []
        for item in queries:
            documents, meta = _timed_acquire(pipeline, item["question"])
            records.append(
                {
                    "id": item["id"],
                    "platform_search_sec": round(meta["platform_search_sec"], 4),
                    "document_count": len(documents),
                    "rate_limit_count": int(meta["request_stats_delta"].get("rate_limit_count", 0) or 0),
                }
            )
        rate_limit_count = sum(row["rate_limit_count"] for row in records)
        row = {
            "max_concurrency": max_concurrency,
            "min_interval_sec": interval,
            "rate_limit_count": rate_limit_count,
            "mean_platform_search_sec": round(statistics.mean(r["platform_search_sec"] for r in records), 4),
            "records": records,
            "status": "no_429" if rate_limit_count == 0 else "429_observed",
        }
        rows.append(row)
        if rate_limit_count == 0:
            break
    fastest = next((row for row in rows if row["rate_limit_count"] == 0), None)
    return {"candidates": rows, "fastest_no_429": fastest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_questions = _load_questions(Path(args.queries))
    if len(all_questions) < args.sample_size + 1:
        raise ValueError("Need at least sample_size + 1 questions for separate warmup and measured samples.")
    rng = random.Random(args.seed)
    selected = rng.sample(all_questions, args.sample_size + 1)
    warmup_item = selected[0]
    measured_items = selected[1:]
    device = _device_name()

    load_started = time.perf_counter()
    pipeline = _new_pipeline(args)
    component_load_sec = time.perf_counter() - load_started

    warmup_record = _run_query(pipeline, warmup_item, phase="warmup", device=device)
    measured_records = [
        _run_query(pipeline, item, phase="measured", device=device)
        for item in measured_items
    ]
    summary = _summarize(measured_records)
    observed_429 = any(row["rate_limit_count"] for row in measured_records + [warmup_record])
    throttle_sweep = None
    if observed_429 and args.throttle_sweep_on_429:
        throttle_sweep = _run_throttle_sweep(args, measured_items, device)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "queries_path": str(args.queries),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "device": device,
        "component_load_sec_excluded": round(component_load_sec, 4),
        "vllm_url": args.vllm_url,
        "vllm_model": args.vllm_model,
        "no_cache": args.no_cache,
        "warmup_record": warmup_record,
        "measured_records": measured_records,
        "summary": summary,
        "observed_429": observed_429,
        "throttle_sweep": throttle_sweep,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm search-to-answer E2E timing for best-config SHRAG.")
    parser.add_argument("--queries", default="experiments/outputs/exp1_kmn/questions_gold41.jsonl")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--scienceon-credentials", default="configs/credentials/scienceon_api_credentials.json")
    parser.add_argument("--cache-root", default="outputs/_shared_cache")
    parser.add_argument("--no-cache", action="store_true", default=True)
    parser.add_argument("--scienceon-max-concurrency", type=int, default=2)
    parser.add_argument("--scienceon-min-interval-sec", type=float, default=0.5)
    parser.add_argument("--scienceon-fixed-concurrency", action="store_true")
    parser.add_argument("--scienceon-max-retries", type=int, default=5)
    parser.add_argument("--scienceon-retry-base-sleep-sec", type=float, default=2.0)
    parser.add_argument("--scienceon-retry-max-sleep-sec", type=float, default=60.0)
    parser.add_argument("--max-answer-tokens", type=int, default=4000)
    parser.add_argument("--throttle-sweep-on-429", action="store_true", default=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()
    settings = Settings()
    args.vllm_url = args.vllm_url or settings.vllm_base_url
    args.vllm_model = args.vllm_model or settings.vllm_model
    return args


def main() -> None:
    args = _parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = Path(args.report_md)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report, report_md)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
