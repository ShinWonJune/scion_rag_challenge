"""Closed-book answer runner — generates answers without retrieved context.

Used for the SHRAG vs Closed-book ablation study. Reads questions from a CSV
or JSONL, calls the chosen LLM backend with `build_closed_book_prompt`, and
writes per-qid JSON files in the same shape as the SHRAG pipeline's `final/`
directory (so `experiments.eval_gold_judge.run_gold_judge` can consume them
unchanged with empty `retrival`).

Example
-------
    python -m experiments.closed_book.run_closed_book \
      --questions data/test.csv \
      --output-dir outputs/closed_gpt_oss_20b/final \
      --llm vllm \
      --vllm-url http://10.38.38.40:8004/v1 \
      --vllm-model openai/gpt-oss-20b
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from experiments.closed_book.prompts import build_closed_book_prompt
from shrag.llm.codex_client import CodexClient
from shrag.llm.openai_client import OpenAIClient
from shrag.llm.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                question = (row.get("Question") or row.get("question") or "").strip()
                if not question:
                    continue
                qid = str(row.get("question_id") or row.get("qid") or idx)
                rows.append({"qid": qid, "question": question})
    else:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                question = (obj.get("Question") or obj.get("question") or "").strip()
                if not question:
                    continue
                qid = str(obj.get("question_id") or obj.get("qid") or idx)
                rows.append({"qid": qid, "question": question})
    return rows


def build_client(args: argparse.Namespace) -> tuple[Any, str]:
    if args.llm == "vllm":
        client = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)
        return client, args.vllm_model
    if args.llm == "openai":
        client = OpenAIClient(
            model=args.openai_model,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            reasoning_effort=args.openai_reasoning_effort,
        )
        return client, args.openai_model
    if args.llm == "codex":
        client = CodexClient(model=args.codex_model, timeout_sec=args.codex_timeout_sec)
        return client, f"codex:{args.codex_model}"
    raise ValueError(f"Unsupported llm backend: {args.llm}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Closed-book answer generation (no retrieval).")
    parser.add_argument("--questions", required=True, help="Questions CSV (column Question) or JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory to write per-qid JSON files.")
    parser.add_argument("--llm", choices=["vllm", "openai", "codex"], default="vllm")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-20b")
    parser.add_argument("--openai-model", default="gpt-5.4")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-reasoning-effort", choices=["low", "medium", "high", "xhigh"], default=None)
    parser.add_argument("--codex-model", default="gpt-5.4")
    parser.add_argument("--codex-timeout-sec", type=int, default=600)
    parser.add_argument("--max-answer-tokens", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions.")
    parser.add_argument(
        "--prompt-type-tag",
        default="closed_book",
        help="prompt_type field written into final JSON (kept distinct from 'general' SHRAG outputs).",
    )
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        logger.error("No questions loaded from %s", args.questions)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client, model_name = build_client(args)
    logger.info("Closed-book run: %d questions, llm=%s, model=%s, output=%s",
                len(questions), args.llm, model_name, out_dir)

    written: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.time()
    for idx, item in enumerate(questions, 1):
        qid = item["qid"]
        question = item["question"]
        prompt = build_closed_book_prompt(question)
        try:
            answer = client.generate_answer_with_prompt(prompt, max_tokens=args.max_answer_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.error("qid=%s generation failed: %s", qid, exc)
            failures.append({"qid": qid, "error": f"{type(exc).__name__}: {exc}"})
            continue

        record = {
            "id": qid,
            "question_id": qid,
            "question": question,
            "result": answer,
            "answer": answer,
            "prompt": prompt,
            "model": model_name,
            "prompt_type": args.prompt_type_tag,
            "used_context": [],
            "retrival": {"retrieval_results": []},
        }
        out_path = out_dir / f"{qid}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=4), encoding="utf-8")
        written.append(record)
        if idx % 5 == 0 or idx == len(questions):
            elapsed = time.time() - start
            logger.info("Done [%d/%d] elapsed=%.1fs", idx, len(questions), elapsed)

    summary = {
        "total_questions": len(questions),
        "written": len(written),
        "failures": failures,
        "model": model_name,
        "llm_backend": args.llm,
        "elapsed_sec": time.time() - start,
    }
    (out_dir / "_closed_book_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d files to %s (failures=%d)", len(written), out_dir, len(failures))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
