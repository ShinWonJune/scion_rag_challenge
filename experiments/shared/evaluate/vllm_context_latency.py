from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from shrag.prompts.general.generate_answer_base_v3 import build_prompt


def _load_queries(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("id"))
            rows[qid] = str(row.get("query") or row.get("question") or "")
    return rows


def _load_retrieval(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("id"))
            rows[qid] = row.get("hits", [])
    return rows


def _load_sample_qids(path: Path | None, all_qids: list[str], limit: int | None) -> list[str]:
    if path is None:
        qids = all_qids
    else:
        qids = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qids.append(str(row.get("question_id") or row.get("id")))
    if limit is not None:
        qids = qids[:limit]
    return qids


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


def _summarize(times: list[float], prompt_chars: list[int], answer_chars: list[int]) -> dict[str, Any]:
    if not times:
        return {}
    return {
        "n": len(times),
        "total_sec": round(sum(times), 4),
        "mean_sec": round(statistics.mean(times), 4),
        "median_sec": round(statistics.median(times), 4),
        "min_sec": round(min(times), 4),
        "max_sec": round(max(times), 4),
        "mean_prompt_chars": round(statistics.mean(prompt_chars), 1),
        "mean_answer_chars": round(statistics.mean(answer_chars), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure vLLM latency for retrieval contexts.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--retrieval", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--sample-queries")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries = _load_queries(Path(args.queries))
    qids = _load_sample_qids(Path(args.sample_queries) if args.sample_queries else None, list(queries), args.limit)
    client = OpenAI(base_url=args.vllm_url, api_key="token-abc123")

    report: dict[str, Any] = {
        "vllm_url": args.vllm_url,
        "vllm_model": args.vllm_model,
        "max_answer_tokens": args.max_answer_tokens,
        "qids": qids,
        "cases": {},
    }

    for label, retrieval_path in args.retrieval:
        retrieval = _load_retrieval(Path(retrieval_path))
        records = []
        times: list[float] = []
        prompt_chars: list[int] = []
        answer_chars: list[int] = []

        for qid in qids:
            question = queries[qid]
            hits = retrieval.get(qid, [])
            payload = _context_payload(qid, question, hits)
            prompt = build_prompt(question, json.dumps(payload, ensure_ascii=False, indent=2))
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=args.vllm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_answer_tokens,
                temperature=0.0,
            )
            elapsed = time.perf_counter() - started
            answer = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else None
            times.append(elapsed)
            prompt_chars.append(len(prompt))
            answer_chars.append(len(answer))
            records.append(
                {
                    "qid": qid,
                    "elapsed_sec": round(elapsed, 4),
                    "n_context_docs": len(hits),
                    "prompt_chars": len(prompt),
                    "answer_chars": len(answer),
                    "usage": usage,
                    "answer": answer,
                }
            )

        report["cases"][label] = {
            "summary": _summarize(times, prompt_chars, answer_chars),
            "records": records,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
