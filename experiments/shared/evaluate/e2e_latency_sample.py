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
from shrag.prompts.general.generate_answer_base_v3 import build_prompt
from shrag.retrieval import data_loader, query_encoder
from shrag.retrieval.retrievers import get_retriever


@dataclass
class E2ECase:
    case_id: str
    label: str
    encoder_model: str
    vector_db: Path
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


def _case_from_dict(row: dict[str, Any]) -> E2ECase:
    return E2ECase(
        case_id=str(row["case_id"]),
        label=str(row.get("label") or row["case_id"]),
        encoder_model=str(row["encoder_model"]),
        vector_db=Path(row["vector_db"]),
        retrieve_top_k=int(row.get("retrieve_top_k", 5)),
        llm_context_top_k=int(row.get("llm_context_top_k", row.get("retrieve_top_k", 5))),
        rerank_model=row.get("rerank_model"),
    )


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
        "e2e_sec",
        "encode_sec",
        "retrieve_sec",
        "rerank_sec",
        "prompt_build_sec",
        "vllm_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_chars",
        "answer_chars",
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
        "# Exp10. E2E Latency Sample3",
        "",
        "Hot-path latency from one user question to one vLLM answer. Corpus collection, embedding, and vector DB build are excluded as offline preparation steps.",
        "",
        f"- vLLM endpoint: `{report['vllm_url']}`",
        f"- vLLM model: `{report['vllm_model']}`",
        f"- sample qids: `{', '.join(report['sample_qids'])}`",
        f"- max answer tokens: `{report['max_answer_tokens']}`",
        "",
        "## Summary",
        "",
        "| case | context docs | hit@context | mean e2e sec | mean encode | mean retrieve | mean rerank | mean vLLM | mean prompt tokens | mean completion tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_id, case in report["cases"].items():
        summary = case["summary"]
        lines.append(
            f"| {case_id}"
            f" | {case['llm_context_top_k']}"
            f" | {summary.get('hit_at_context', 0)}"
            f" | {summary.get('mean_e2e_sec', 0)}"
            f" | {summary.get('mean_encode_sec', 0)}"
            f" | {summary.get('mean_retrieve_sec', 0)}"
            f" | {summary.get('mean_rerank_sec', 0)}"
            f" | {summary.get('mean_vllm_sec', 0)}"
            f" | {summary.get('mean_prompt_tokens', 0)}"
            f" | {summary.get('mean_completion_tokens', 0)} |"
        )
    lines.extend(["", "## Per Question", ""])
    for case_id, case in report["cases"].items():
        lines.extend(
            [
                f"### {case_id}",
                "",
                "| qid | gold_rank | e2e_sec | encode | retrieve | rerank | vLLM | prompt_tokens | completion_tokens |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in case["records"]:
            lines.append(
                f"| {row['qid']}"
                f" | {row.get('gold_rank') or '-'}"
                f" | {row['e2e_sec']}"
                f" | {row['encode_sec']}"
                f" | {row['retrieve_sec']}"
                f" | {row['rerank_sec']}"
                f" | {row['vllm_sec']}"
                f" | {row.get('prompt_tokens') or '-'}"
                f" | {row.get('completion_tokens') or '-'} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(Path(args.cases))
    sample_qids = [str(qid) for qid in config["sample_qids"]]
    cases = [_case_from_dict(row) for row in config["cases"]]
    queries = _load_queries(Path(args.queries))
    gold = _load_gold(Path(args.gold))
    client = OpenAI(base_url=args.vllm_url, api_key=args.api_key)

    report: dict[str, Any] = {
        "vllm_url": args.vllm_url,
        "vllm_model": args.vllm_model,
        "max_answer_tokens": args.max_answer_tokens,
        "sample_qids": sample_qids,
        "cases": {},
    }

    for case in cases:
        load_started = time.perf_counter()
        vectordb = data_loader.load_vectordb_from_csv(str(case.vector_db), args.schema)
        encoder = query_encoder.QueryEncoder(model_name=case.encoder_model, device="auto")
        retriever = get_retriever(vectordb.embeddings)
        reranker = CrossEncoderReranker(case.rerank_model) if case.rerank_model else None
        load_sec = time.perf_counter() - load_started

        records: list[dict[str, Any]] = []
        for qid in sample_qids:
            question = queries[qid]
            e2e_started = time.perf_counter()

            encode_started = time.perf_counter()
            query_vec = encoder.encode_queries([question])
            encode_sec = time.perf_counter() - encode_started

            retrieve_started = time.perf_counter()
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
            retrieve_sec = time.perf_counter() - retrieve_started

            rerank_started = time.perf_counter()
            if reranker is not None:
                hits = reranker.rerank(question, hits, top_k=min(case.llm_context_top_k, len(hits)))
            else:
                hits = hits[: case.llm_context_top_k]
            rerank_sec = time.perf_counter() - rerank_started if reranker is not None else 0.0

            prompt_started = time.perf_counter()
            payload = _context_payload(qid, question, hits)
            prompt = build_prompt(question, json.dumps(payload, ensure_ascii=False, indent=2))
            prompt_build_sec = time.perf_counter() - prompt_started

            vllm_started = time.perf_counter()
            response = client.chat.completions.create(
                model=args.vllm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_answer_tokens,
                temperature=0.0,
            )
            vllm_sec = time.perf_counter() - vllm_started
            e2e_sec = time.perf_counter() - e2e_started

            answer = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            records.append(
                {
                    "qid": qid,
                    "gold_rank": _gold_rank(hits, gold.get(qid, set())),
                    "e2e_sec": round(e2e_sec, 4),
                    "encode_sec": round(encode_sec, 4),
                    "retrieve_sec": round(retrieve_sec, 4),
                    "rerank_sec": round(rerank_sec, 4),
                    "prompt_build_sec": round(prompt_build_sec, 4),
                    "vllm_sec": round(vllm_sec, 4),
                    "n_context_docs": len(hits),
                    "prompt_chars": len(prompt),
                    "answer_chars": len(answer),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "answer": answer,
                }
            )

        report["cases"][case.case_id] = {
            "label": case.label,
            "encoder_model": case.encoder_model,
            "vector_db": str(case.vector_db),
            "retrieve_top_k": case.retrieve_top_k,
            "llm_context_top_k": case.llm_context_top_k,
            "rerank_model": case.rerank_model,
            "load_sec_excluded": round(load_sec, 4),
            "summary": _summarize(records),
            "records": records,
        }

        del reranker, retriever, encoder, vectordb
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure hot-path RAG e2e latency on a small sample.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--schema", default="configs/csv_schema/test_2.json")
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
