from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import statistics
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.shared.evaluate.warm_e2e_search_to_answer import (
    _load_questions,
    _new_pipeline,
)
from shrag_lc.embeddings import doc_to_document
from shrag_lc.search.acquire import _request_stats, _search_until_target, _stats_delta
from shrag_lc.search.clients import _dedup
from shrag_lc.search.terms import build_search_terms_by_lang


def _load_qids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(row["id"]) for row in payload["measured_records"]]


def _resolve_questions(all_questions: list[dict[str, str]], qids: list[str]) -> list[dict[str, str]]:
    by_id = {row["id"]: row for row in all_questions}
    missing = [qid for qid in qids if qid not in by_id]
    if missing:
        raise ValueError(f"Missing qids in query file: {missing}")
    return [by_id[qid] for qid in qids]


def _extract_terms(args: argparse.Namespace, questions: list[dict[str, str]]) -> list[dict[str, Any]]:
    pipeline = _new_pipeline(args)
    rows: list[dict[str, Any]] = []
    for item in questions:
        started = time.perf_counter()
        try:
            keywords = pipeline.acquirer.extractor.extract(item["question"])
            status = "success"
            error = None
        except Exception as exc:  # noqa: BLE001
            keywords = {"korean": [], "english": [item["question"]]}
            status = "fallback"
            error = str(exc)
        terms_by_lang = build_search_terms_by_lang(keywords, pipeline.cfg.number_of_operators)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "keyword_status": status,
                "keyword_error": error,
                "keyword_extract_sec": round(time.perf_counter() - started, 4),
                "keywords": keywords,
                "ko_terms": terms_by_lang.get("korean") or [],
                "en_terms": terms_by_lang.get("english") or [],
            }
        )
    return rows


def _timed_search_with_terms(pipeline: Any, term_row: dict[str, Any]) -> dict[str, Any]:
    client = pipeline.acquirer.client
    target = pipeline.cfg.target_documents
    stats_before = _request_stats(client)
    started = time.perf_counter()
    ko_result = _search_until_target(client, term_row["ko_terms"], target, "korean")
    en_result = _search_until_target(client, term_row["en_terms"], target, "english")
    elapsed = time.perf_counter() - started
    stats_after = _request_stats(client)

    merged = _dedup(ko_result.documents + en_result.documents, key="doc_id")
    documents = [doc_to_document(doc, pipeline.cfg.embedding_mode) for doc in merged]
    delta = _stats_delta(stats_before, stats_after)
    return {
        "id": term_row["id"],
        "platform_search_sec": round(elapsed, 4),
        "document_count": len(documents),
        "document_count_by_lang": {"korean": len(ko_result.documents), "english": len(en_result.documents)},
        "request_count": int(delta.get("request_count", 0) or 0),
        "rate_limit_count": int(delta.get("rate_limit_count", 0) or 0),
        "api_error_count": int(delta.get("api_error_count", 0) or 0),
        "failed_terms": ko_result.failed_terms + en_result.failed_terms,
    }


def _parse_candidates(raw: str) -> list[tuple[int, float]]:
    candidates: list[tuple[int, float]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid candidate {token!r}; expected concurrency:interval")
        candidates.append((int(parts[0]), float(parts[1])))
    return candidates


def _run_candidate(args: argparse.Namespace, terms: list[dict[str, Any]], max_concurrency: int, interval: float) -> dict[str, Any]:
    pipeline = _new_pipeline(args, max_concurrency=max_concurrency, min_interval=interval, fixed=True)
    rows = []
    stopped_early = False
    for term_row in terms:
        row = _timed_search_with_terms(pipeline, term_row)
        rows.append(row)
        if args.stop_candidate_on_429 and row["rate_limit_count"] > 0:
            stopped_early = True
            break
    total_429 = sum(row["rate_limit_count"] for row in rows)
    total_requests = sum(row["request_count"] for row in rows)
    return {
        "max_concurrency": max_concurrency,
        "min_interval_sec": interval,
        "fixed_concurrency": True,
        "total_requests": total_requests,
        "total_429": total_429,
        "total_api_errors": sum(row["api_error_count"] for row in rows),
        "mean_platform_search_sec": round(statistics.mean(row["platform_search_sec"] for row in rows), 4),
        "median_platform_search_sec": round(statistics.median(row["platform_search_sec"] for row in rows), 4),
        "max_platform_search_sec": round(max(row["platform_search_sec"] for row in rows), 4),
        "mean_document_count": round(statistics.mean(row["document_count"] for row in rows), 4),
        "status": "no_429" if total_429 == 0 else "429_observed",
        "stopped_early": stopped_early,
        "completed_query_count": len(rows),
        "records": rows,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# ScienceON 429 Throttle Sweep",
        "",
        "This run fixes the query set and extracted search terms, then varies only the ScienceON HTTP throttle.",
        "",
        "## Inputs",
        "",
        f"- Warm E2E baseline result: `{report['baseline_result_path']}`",
        f"- Query file: `{report['queries_path']}`",
        f"- Qids: `{', '.join(report['qids'])}`",
        f"- Cache disabled: `{report['no_cache']}`",
        f"- Sweep retries: `{report['scienceon_max_retries']}`",
        "",
        "## Candidate Results",
        "",
        "| concurrency | min interval sec | completed q | total requests | total 429 | mean search sec/q | mean docs/q | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["candidates"]:
        lines.append(
            f"| {row['max_concurrency']}"
            f" | {row['min_interval_sec']}"
            f" | {row['completed_query_count']}"
            f" | {row['total_requests']}"
            f" | {row['total_429']}"
            f" | {row['mean_platform_search_sec']}"
            f" | {row['mean_document_count']}"
            f" | {row['status']} |"
        )
    best = report.get("fastest_no_429")
    if best:
        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                f"Fastest no-429 candidate: `max_concurrency={best['max_concurrency']}`, `min_interval_sec={best['min_interval_sec']}` with mean platform search `{best['mean_platform_search_sec']}` sec/query.",
            ]
        )
    else:
        lines.extend(["", "## Conclusion", "", "No tested candidate avoided 429."])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    qids = _load_qids(Path(args.baseline_result))
    all_questions = _load_questions(Path(args.queries))
    questions = _resolve_questions(all_questions, qids)
    terms = _extract_terms(args, questions)
    candidates = [
        _run_candidate(args, terms, max_concurrency=max_concurrency, interval=interval)
        for max_concurrency, interval in _parse_candidates(args.candidates)
    ]
    no_429 = [row for row in candidates if row["total_429"] == 0]
    fastest_no_429 = min(no_429, key=lambda row: row["mean_platform_search_sec"]) if no_429 else None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_result_path": args.baseline_result,
        "queries_path": args.queries,
        "qids": qids,
        "no_cache": args.no_cache,
        "scienceon_max_retries": args.scienceon_max_retries,
        "vllm_url": args.vllm_url,
        "vllm_model": args.vllm_model,
        "terms": terms,
        "candidates": candidates,
        "fastest_no_429": fastest_no_429,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep fixed ScienceON throttle candidates on the warm E2E measured qids.")
    parser.add_argument("--baseline-result", default="experiments/outputs/warm_e2e_search_to_answer_20260606/result.json")
    parser.add_argument("--queries", default="experiments/outputs/exp1_kmn/questions_gold41.jsonl")
    parser.add_argument("--candidates", default="2:0.0,2:0.1,2:0.25,2:0.5")
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--scienceon-credentials", default="configs/credentials/scienceon_api_credentials.json")
    parser.add_argument("--cache-root", default="outputs/_shared_cache")
    parser.add_argument("--no-cache", action="store_true", default=True)
    parser.add_argument("--scienceon-max-concurrency", type=int, default=2)
    parser.add_argument("--scienceon-min-interval-sec", type=float, default=0.5)
    parser.add_argument("--scienceon-fixed-concurrency", action="store_true")
    parser.add_argument("--scienceon-max-retries", type=int, default=0)
    parser.add_argument("--scienceon-retry-base-sleep-sec", type=float, default=2.0)
    parser.add_argument("--scienceon-retry-max-sleep-sec", type=float, default=60.0)
    parser.add_argument("--max-answer-tokens", type=int, default=4000)
    parser.add_argument("--stop-candidate-on-429", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()
    from shrag_lc.config import Settings

    settings = Settings()
    args.vllm_url = args.vllm_url or settings.vllm_base_url
    args.vllm_model = args.vllm_model or settings.vllm_model
    return args


def main() -> None:
    args = _parse_args()
    if args.quiet:
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = run(args)
    else:
        report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = Path(args.report_md)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report, report_md)
    print(json.dumps({"fastest_no_429": report["fastest_no_429"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
