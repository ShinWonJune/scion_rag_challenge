from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiments.shared.rerank.cross_encoder import CrossEncoderReranker
from shrag.pipeline._impl.build_vectordb import build_vectordb_search
from shrag.prompts.general.generate_answer_base_v3 import build_prompt
from shrag.retrieval import data_loader, query_encoder
from shrag.retrieval.retrievers import get_retriever
from shrag.search.extractors.extractor_factory import create_keyword_extractor
from shrag.search.extractors.search_terms import build_search_terms
from shrag.search.factories.search_client_factory import create_search_client
from shrag.utils.dedup import remove_duplicates


@dataclass
class ColdCase:
    case_id: str
    label: str
    encoder_config: Path
    retrieve_top_k: int
    llm_context_top_k: int
    rerank_model: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("id"))
            queries[qid] = str(row.get("query") or row.get("question") or "")
    return queries


def _load_gold(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): {str(item).strip() for item in value} for key, value in payload.items()}


def _case_from_dict(row: dict[str, Any]) -> ColdCase:
    return ColdCase(
        case_id=str(row["case_id"]),
        label=str(row.get("label") or row["case_id"]),
        encoder_config=Path(row["encoder_config"]),
        retrieve_top_k=int(row.get("retrieve_top_k", 5)),
        llm_context_top_k=int(row.get("llm_context_top_k", row.get("retrieve_top_k", 5))),
        rerank_model=row.get("rerank_model"),
    )


def _build_terms_by_lang(keywords: dict[str, list[str]], max_terms: int) -> dict[str, list[str]]:
    ko_kw = list(keywords.get("korean") or [])
    en_kw = list(keywords.get("english") or [])
    return {
        "korean": build_search_terms({"korean": ko_kw, "english": []}, number_of_operators=0)[:max_terms],
        "english": build_search_terms({"korean": [], "english": en_kw}, number_of_operators=0)[:max_terms],
    }


def _search_terms_until_target(
    client: Any,
    terms: list[str],
    target_documents: int,
    source_name: str,
    lang: str,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for term in terms:
        if len(collected) >= target_documents:
            break
        remaining = target_documents - len(collected)
        try:
            docs = client.search([term], max_results=remaining)
        except Exception as exc:
            print(f"[WARN] {source_name} search failed lang={lang} term={term}: {exc}")
            continue
        collected = remove_duplicates(collected + docs, key="doc_id", fallback_keys=("title",))
    return collected[:target_documents]


def _install_default_timeout(client: Any, timeout_sec: float) -> None:
    session = getattr(getattr(client, "client", None), "session", None)
    if session is None:
        return
    current_get = session.get

    def get_with_timeout(*args: Any, **kwargs: Any):
        kwargs.setdefault("timeout", timeout_sec)
        return current_get(*args, **kwargs)

    session.get = get_with_timeout  # type: ignore[method-assign]


def _prepare_question_file(qid: str, question: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"id": qid, "question": question}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_encoder_config(original: Path, docs_jsonl: Path, case_dir: Path) -> Path:
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["jsonl_path"] = str(docs_jsonl)
    payload["output_dir"] = str(case_dir)
    payload["output_file"] = str(case_dir / "vectordb.csv")
    payload.pop("cache_db_path", None)
    path = case_dir / "encoder_config.cold.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _find_vectordb(case_dir: Path) -> Path:
    direct = case_dir / "vectordb.csv"
    if direct.exists():
        return direct
    candidates = sorted(case_dir.rglob("vector_db_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No vector DB CSV found under {case_dir}")


def _context_payload(qid: str, question: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": qid,
        "retrieval_results": [
            {
                "query": question,
                "query_meta": {"type": "original"},
                "hits": hits,
            }
        ],
    }


def _gold_rank(hits: list[dict[str, Any]], gold_doc_ids: set[str]) -> int | None:
    for idx, hit in enumerate(hits, start=1):
        if str(hit.get("doc_id", "")).strip() in gold_doc_ids:
            return idx
    return None


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "cold_total_sec",
        "search_term_generation_sec",
        "collection_sec",
        "embedding_build_sec",
        "dense_retrieve_sec",
        "rerank_sec",
        "generation_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "docs_count",
    ]
    summary: dict[str, Any] = {"n": len(records)}
    for field in fields:
        values = [row[field] for row in records if row.get(field) is not None]
        if values:
            summary[f"mean_{field}"] = round(statistics.mean(values), 4)
            summary[f"median_{field}"] = round(statistics.median(values), 4)
    summary["hit_at_context"] = round(
        sum(1 for row in records if row.get("gold_rank") is not None) / len(records),
        6,
    ) if records else 0.0
    return summary


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Exp11. Cold E2E Latency Sample3",
        "",
        "Cold-path latency including search-term generation, live ScienceON collection, document embedding/vector DB build, dense retrieval, optional reranking, and vLLM answer generation.",
        "",
        "Notes:",
        "",
        "- Embedding cache is disabled by removing `cache_db_path` from temporary encoder configs.",
        "- ScienceON request cache is disabled for acquisition.",
        "- The external vLLM server is not restarted; endpoint/server startup is excluded.",
        "",
        f"- sample qids: `{', '.join(report['sample_qids'])}`",
        f"- search policy: `k={report['search']['k']}, n={report['search']['n']}, m={report['search']['m']}`",
        f"- vLLM endpoint: `{report['vllm_url']}`",
        f"- vLLM model: `{report['vllm_model']}`",
        "",
        "## Summary",
        "",
        "| case | hit@context | mean cold total | search terms | collection | embedding | dense retrieve | rerank | generation | docs/q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_id, case in report["cases"].items():
        summary = case["summary"]
        lines.append(
            f"| {case_id}"
            f" | {summary.get('hit_at_context', 0)}"
            f" | {summary.get('mean_cold_total_sec', 0)}"
            f" | {summary.get('mean_search_term_generation_sec', 0)}"
            f" | {summary.get('mean_collection_sec', 0)}"
            f" | {summary.get('mean_embedding_build_sec', 0)}"
            f" | {summary.get('mean_dense_retrieve_sec', 0)}"
            f" | {summary.get('mean_rerank_sec', 0)}"
            f" | {summary.get('mean_generation_sec', 0)}"
            f" | {summary.get('mean_docs_count', 0)} |"
        )
    lines.extend(["", "## Per Question", ""])
    for case_id, case in report["cases"].items():
        lines.extend(
            [
                f"### {case_id}",
                "",
                "| qid | gold_rank | total | terms | collect | embed | retrieve | rerank | generate | docs | prompt_tokens | completion_tokens |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in case["records"]:
            lines.append(
                f"| {row['qid']}"
                f" | {row.get('gold_rank') or '-'}"
                f" | {row['cold_total_sec']}"
                f" | {row['search_term_generation_sec']}"
                f" | {row['collection_sec']}"
                f" | {row['embedding_build_sec']}"
                f" | {row['dense_retrieve_sec']}"
                f" | {row['rerank_sec']}"
                f" | {row['generation_sec']}"
                f" | {row['docs_count']}"
                f" | {row.get('prompt_tokens') or '-'}"
                f" | {row.get('completion_tokens') or '-'} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(Path(args.cases))
    sample_qids = [str(qid) for qid in config["sample_qids"]]
    search_cfg = config.get("search", {})
    cases = [_case_from_dict(row) for row in config["cases"]]
    queries = _load_queries(Path(args.queries))
    gold = _load_gold(Path(args.gold))
    output_root = Path(args.output).parent
    output_root.mkdir(parents=True, exist_ok=True)
    answer_client = OpenAI(base_url=args.vllm_url, api_key=args.api_key)

    report: dict[str, Any] = {
        "sample_qids": sample_qids,
        "search": search_cfg,
        "vllm_url": args.vllm_url,
        "vllm_model": args.vllm_model,
        "max_answer_tokens": args.max_answer_tokens,
        "cases": {case.case_id: {
            "label": case.label,
            "encoder_config": str(case.encoder_config),
            "retrieve_top_k": case.retrieve_top_k,
            "llm_context_top_k": case.llm_context_top_k,
            "rerank_model": case.rerank_model,
            "records": [],
        } for case in cases},
    }

    for qid in sample_qids:
        question = queries[qid]
        q_dir = output_root / "questions" / f"q{qid}"
        _prepare_question_file(qid, question, q_dir / "question.jsonl")
        print(f"[INFO] q{qid}: cold acquisition start", flush=True)

        extractor = create_keyword_extractor(
            search_cfg.get("extractor", "vllm"),
            {
                "language": "all",
                "vllm_base_url": args.vllm_url,
                "vllm_model": args.vllm_model,
                "temperature": 0.0,
            },
        )
        source_name = str(search_cfg.get("source", "scienceon"))
        client = create_search_client(
            source_name,
            {
                "scienceon_credentials_path": args.scienceon_credentials,
                "scienceon_max_pages": int(search_cfg.get("scienceon_max_pages", 3)),
                "scienceon_max_concurrency": int(search_cfg.get("scienceon_max_concurrency", 2)),
                "scienceon_min_interval_sec": float(search_cfg.get("scienceon_min_interval_sec", 0.5)),
                "disable_cache": bool(search_cfg.get("no_cache", True)),
                "cache_root": str(output_root / "_request_cache_disabled"),
            },
        )
        _install_default_timeout(client, float(search_cfg.get("scienceon_timeout_sec", 20)))

        term_started = time.perf_counter()
        keywords = extractor.extract_keywords(question)
        terms_by_lang = _build_terms_by_lang(keywords, max_terms=int(search_cfg.get("n", 10)))
        search_term_generation_sec = time.perf_counter() - term_started
        print(
            f"[INFO] q{qid}: generated terms ko={len(terms_by_lang.get('korean', []))} "
            f"en={len(terms_by_lang.get('english', []))} in {search_term_generation_sec:.2f}s",
            flush=True,
        )

        collect_started = time.perf_counter()
        target_documents = int(search_cfg.get("m", 50))
        ko_docs = _search_terms_until_target(
            client, terms_by_lang.get("korean", []), target_documents, source_name, "korean"
        )
        en_docs = _search_terms_until_target(
            client, terms_by_lang.get("english", []), target_documents, source_name, "english"
        )
        docs = remove_duplicates(ko_docs + en_docs, key="doc_id", fallback_keys=("title",))
        collection_sec = time.perf_counter() - collect_started
        print(
            f"[INFO] q{qid}: collected docs={len(docs)} "
            f"(ko={len(ko_docs)}, en={len(en_docs)}) in {collection_sec:.2f}s",
            flush=True,
        )

        docs_jsonl = q_dir / "search_documents.jsonl"
        docs_jsonl.write_text(
            "\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs),
            encoding="utf-8",
        )
        (q_dir / "search_meta_results.json").write_text(
            json.dumps(
                {
                    "question_id": qid,
                    "query": question,
                    "keywords": keywords,
                    "search_terms_by_lang": terms_by_lang,
                    "document_count": len(docs),
                    "document_count_by_lang": {"korean": len(ko_docs), "english": len(en_docs)},
                    "search_term_generation_sec": round(search_term_generation_sec, 4),
                    "collection_sec": round(collection_sec, 4),
                    "request_stats": client.get_request_stats() if hasattr(client, "get_request_stats") else {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        for case in cases:
            print(f"[INFO] q{qid} {case.case_id}: cold case start", flush=True)
            case_dir = output_root / "cases" / case.case_id / f"q{qid}"
            case_dir.mkdir(parents=True, exist_ok=True)
            cold_started = time.perf_counter()

            encoder_cfg = _prepare_encoder_config(case.encoder_config, docs_jsonl, case_dir)
            embed_started = time.perf_counter()
            build_vectordb_search(
                config_path=str(encoder_cfg),
                data_schema=args.schema,
                docs_jsonl_path=str(docs_jsonl),
                auto_data_load=False,
                gpu_id=None,
            )
            embedding_build_sec = time.perf_counter() - embed_started
            vectordb_path = _find_vectordb(case_dir)

            retrieve_started = time.perf_counter()
            vectordb = data_loader.load_vectordb_from_csv(str(vectordb_path), args.schema)
            encoder_cfg_payload = json.loads(encoder_cfg.read_text(encoding="utf-8"))
            encoder = query_encoder.QueryEncoder(model_name=encoder_cfg_payload["model_name"], device="auto")
            retriever = get_retriever(vectordb.embeddings)
            query_vec = encoder.encode_queries([question])
            scores, indices = retriever.search(query_vec, top_k=case.retrieve_top_k)
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
            dense_retrieve_sec = time.perf_counter() - retrieve_started

            rerank_started = time.perf_counter()
            if case.rerank_model:
                reranker = CrossEncoderReranker(case.rerank_model)
                hits = reranker.rerank(question, hits, top_k=min(case.llm_context_top_k, len(hits)))
            else:
                reranker = None
                hits = hits[: case.llm_context_top_k]
            rerank_sec = time.perf_counter() - rerank_started if case.rerank_model else 0.0

            generate_started = time.perf_counter()
            prompt = build_prompt(
                question,
                json.dumps(_context_payload(qid, question, hits), ensure_ascii=False, indent=2),
            )
            response = answer_client.chat.completions.create(
                model=args.vllm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_answer_tokens,
                temperature=0.0,
            )
            generation_sec = time.perf_counter() - generate_started
            answer = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}

            record = {
                "qid": qid,
                "gold_rank": _gold_rank(hits, gold.get(qid, set())),
                "cold_total_sec": round(time.perf_counter() - cold_started + search_term_generation_sec + collection_sec, 4),
                "search_term_generation_sec": round(search_term_generation_sec, 4),
                "collection_sec": round(collection_sec, 4),
                "embedding_build_sec": round(embedding_build_sec, 4),
                "dense_retrieve_sec": round(dense_retrieve_sec, 4),
                "rerank_sec": round(rerank_sec, 4),
                "generation_sec": round(generation_sec, 4),
                "docs_count": len(docs),
                "context_docs": len(hits),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "answer_chars": len(answer),
                "answer": answer,
            }
            report["cases"][case.case_id]["records"].append(record)
            (case_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[INFO] q{qid} {case.case_id}: total={record['cold_total_sec']}s "
                f"embed={record['embedding_build_sec']}s retrieve={record['dense_retrieve_sec']}s "
                f"rerank={record['rerank_sec']}s gen={record['generation_sec']}s",
                flush=True,
            )

            del reranker, retriever, encoder, vectordb
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    for case in report["cases"].values():
        case["summary"] = _summarize(case["records"])
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure cold-path RAG e2e latency.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--schema", default="configs/csv_schema/test_2.json")
    parser.add_argument("--scienceon-credentials", default="configs/credentials/scienceon_api_credentials.json")
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, Path(args.report_md))
    print(json.dumps({case_id: case["summary"] for case_id, case in report["cases"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
