from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.dedup import remove_duplicates
from src.search.factories.search_client_factory import create_search_client
from src.search_pipeline.core.extractor_factory import create_keyword_extractor


def _sources_to_list(sources: str) -> list[str]:
    return [s.strip().lower() for s in sources.split(",") if s.strip()]


def _to_wiki_lang(keyword_lang: str) -> str:
    return "ko" if keyword_lang == "korean" else "en"


def _load_questions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    rows: list[dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                question = item.get("question") or item.get("query") or item.get("text")
                if question:
                    rows.append({"question_id": str(item.get("id", idx)), "query": str(question)})
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        lower_map = {name.lower(): name for name in reader.fieldnames}
        q_col = None
        for candidate in ["question", "query", "text"]:
            if candidate in lower_map:
                q_col = lower_map[candidate]
                break
        id_col = lower_map.get("id") or lower_map.get("question_id")
        if not q_col:
            raise ValueError(f"Question column not found in CSV: {path}")
        for idx, item in enumerate(reader):
            question = (item.get(q_col) or "").strip()
            if question:
                qid = (item.get(id_col) or str(idx)).strip() if id_col else str(idx)
                rows.append({"question_id": qid, "query": question})
    return rows


def _load_frozen_queries(path: Path) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Frozen queries file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item.get("question_id") or item.get("id") or "").strip()
            if qid:
                payload[qid] = item
    return payload


def _extract_query_terms(keyword_extractor: Any, query: str) -> tuple[dict[str, Any], list[str]]:
    keywords = keyword_extractor.extract_keywords(query)
    search_terms = keyword_extractor.generate_search_terms(keywords)
    if not search_terms:
        search_terms = [query]
    return keywords, search_terms


def emit_frozen_queries(questions_path: str, output_path: str, keyword_extractor: Any) -> str:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in _load_questions(Path(questions_path)):
            try:
                keywords, search_terms = _extract_query_terms(keyword_extractor, row["query"])
            except Exception as e:
                logging.error("Keyword extraction failed for frozen query '%s': %s", row["query"], e)
                keywords = {"english": [row["query"]], "korean": []}
                search_terms = [row["query"]]
            payload = {
                "question_id": row["question_id"],
                "query": row["query"],
                "keywords": keywords,
                "search_terms": search_terms,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(out_path)


def run_step1(
    questions_path: str,
    output_dir: str,
    keyword_extractor: Any,
    search_clients: dict[str, Any],
    target_documents: int = 50,
    use_timestamp_subdir: bool = True,
    frozen_queries_path: str | None = None,
) -> tuple[str, str]:
    started_at = datetime.now()
    out_dir = Path(output_dir)
    if use_timestamp_subdir:
        out_dir = out_dir / datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(search_clients) != 1:
        raise ValueError(
            "Step1 supports exactly one source per run. "
            f"Received: {', '.join(search_clients.keys()) or '<none>'}"
        )

    results: list[dict[str, Any]] = []
    all_docs: list[dict[str, Any]] = []
    source_name, client = next(iter(search_clients.items()))
    frozen_queries = _load_frozen_queries(Path(frozen_queries_path)) if frozen_queries_path else {}

    for row in _load_questions(Path(questions_path)):
        query = row["query"]
        frozen = frozen_queries.get(row["question_id"])
        if frozen:
            keywords = frozen.get("keywords") or {"english": [], "korean": []}
            search_terms = frozen.get("search_terms") or [query]
        else:
            try:
                keywords, search_terms = _extract_query_terms(keyword_extractor, query)
            except Exception as e:
                logging.error("Keyword extraction failed for query '%s': %s", query, e)
                keywords = {"english": [query], "korean": []}
                search_terms = [query]

        docs: list[dict[str, Any]] = []
        try:
            docs = client.search(search_terms, max_results=target_documents)
        except Exception as e:
            logging.error("%s search failed for query '%s': %s", source_name, query, e)
        docs = remove_duplicates(docs, key="doc_id", fallback_keys=("title",))[:target_documents]

        results.append(
            {
                "question_id": row["question_id"],
                "query": query,
                "source": source_name,
                "keywords": keywords,
                "search_terms": search_terms,
                "documents": docs,
                "document_count": len(docs),
            }
        )
        all_docs.extend(docs)

    all_docs = remove_duplicates(all_docs, key="doc_id", fallback_keys=("title",))
    meta_output = out_dir / "search_meta_results.json"
    docs_output = out_dir / "search_documents.jsonl"
    client_stats = client.get_request_stats() if hasattr(client, "get_request_stats") else {}

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": round((datetime.now() - started_at).total_seconds(), 4),
        "total_queries": len(results),
        "request_stats": client_stats,
        "results": results,
    }
    with meta_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with docs_output.open("w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return str(meta_output), str(docs_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1: question search orchestration")
    parser.add_argument("--questions", required=True, help="Input questions file (.csv or .jsonl)")
    parser.add_argument("--output-dir", default="outputs/search", help="Search output directory")
    parser.add_argument("--extractor", default="gemini", help="Keyword extractor backend")
    parser.add_argument("--sources", default="scienceon", help="Single source: scienceon|pubmed|wikipedia")
    parser.add_argument("--keyword-lang", choices=["all", "korean", "english"], default="all")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM endpoint URL")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-20b", help="vLLM model name")
    parser.add_argument("--target-documents", type=int, default=50, help="Target documents per query")
    parser.add_argument("--scienceon-credentials", default="configs/credentials/scienceon_api_credentials.json")
    parser.add_argument("--scienceon-max-pages", type=int, default=5, help="Max ScienceON pages per query")
    parser.add_argument("--scienceon-max-concurrency", type=int, default=2, help="Max concurrent ScienceON requests")
    parser.add_argument("--scienceon-min-interval-sec", type=float, default=0.5, help="Minimum interval between ScienceON requests")
    parser.add_argument("--scienceon-fixed-concurrency", action="store_true", help="Disable AIMD adaptation during ScienceON calls")
    parser.add_argument("--scienceon-max-retries", type=int, default=5, help="Max retries for ScienceON 429")
    parser.add_argument("--scienceon-retry-base-sleep-sec", type=float, default=2.0, help="Base retry sleep seconds")
    parser.add_argument("--scienceon-retry-max-sleep-sec", type=float, default=60.0, help="Max retry sleep seconds")
    parser.add_argument("--cache-root", default="outputs/_shared_cache", help="Shared request cache root")
    parser.add_argument("--no-cache", action="store_true", help="Disable shared request cache")
    parser.add_argument("--frozen-queries", default=None, help="JSONL file with precomputed keywords/search terms")
    parser.add_argument("--emit-frozen-queries", default=None, help="Only emit frozen queries JSONL and exit")
    parser.add_argument("--pubmed-credentials", default="configs/credentials/pubmed_api_credentials.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = create_keyword_extractor(
        args.extractor,
        {
            "language": args.keyword_lang,
            "vllm_base_url": args.vllm_url,
            "vllm_model": args.vllm_model,
        },
    )
    if args.emit_frozen_queries:
        output = emit_frozen_queries(args.questions, args.emit_frozen_queries, extractor)
        print(output)
        return
    clients: dict[str, Any] = {}
    for source in _sources_to_list(args.sources):
        clients[source] = create_search_client(
            source,
            {
                "lang": _to_wiki_lang(args.keyword_lang),
                "scienceon_credentials_path": args.scienceon_credentials,
                "scienceon_max_pages": args.scienceon_max_pages,
                "scienceon_max_concurrency": args.scienceon_max_concurrency,
                "scienceon_min_interval_sec": args.scienceon_min_interval_sec,
                "scienceon_fixed_concurrency": args.scienceon_fixed_concurrency,
                "scienceon_max_retries": args.scienceon_max_retries,
                "scienceon_retry_base_sleep_sec": args.scienceon_retry_base_sleep_sec,
                "scienceon_retry_max_sleep_sec": args.scienceon_retry_max_sleep_sec,
                "cache_root": args.cache_root,
                "disable_cache": args.no_cache,
                "pubmed_credentials_path": args.pubmed_credentials,
            },
        )

    _, docs_output = run_step1(
        questions_path=args.questions,
        output_dir=args.output_dir,
        keyword_extractor=extractor,
        search_clients=clients,
        target_documents=args.target_documents,
        frozen_queries_path=args.frozen_queries,
    )
    print(docs_output)


if __name__ == "__main__":
    main()
