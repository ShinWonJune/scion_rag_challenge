from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# google.generativeai and openai are imported lazily inside call_judge so the
# module can be imported for offline analysis (dry-run, unit tests) without
# requiring both provider SDKs to be installed.

from experiments.eval_gold_judge.prompts import build_gold_answer_judge_prompt

PROMPT_VERSION = "gold_answer_judge_v2"
ESCAPE_SENTENCES = (
    "제공된 문서에서는 이 질문에 답할 정보를 찾을 수 없습니다.",
    "The provided documents do not contain sufficient information to answer this question.",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def find_final_files(final_dir: Path) -> dict[str, Path]:
    if not final_dir.exists():
        raise FileNotFoundError(f"Final output directory not found: {final_dir}")
    direct = sorted(final_dir.glob("*.json"))
    if direct:
        return {path.stem: path for path in direct}

    subdirs = [path for path in final_dir.iterdir() if path.is_dir()]
    if not subdirs:
        return {}
    latest = max(subdirs, key=lambda path: path.stat().st_mtime)
    return {path.stem: path for path in sorted(latest.glob("*.json"))}


def load_prediction(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "qid": str(payload.get("id") or path.stem),
        "answer": payload.get("result") or payload.get("answer") or "",
        "raw": payload,
    }


def _original_query_obj(raw_payload: dict[str, Any]) -> dict[str, Any]:
    # Accept both the typo'd 'retrival' (existing pipeline schema) and 'retrieval'.
    retrieval = raw_payload.get("retrival") or raw_payload.get("retrieval") or {}
    results = retrieval.get("retrieval_results") or []
    return next(
        (
            row
            for row in results
            if row.get("query_meta", {}).get("type") == "original"
        ),
        results[0] if results else {},
    )


def original_hits_from_final(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_original_query_obj(raw_payload).get("hits") or [])


def retrieved_context_for_judge(
    hits: list[dict[str, Any]], max_docs: int = 5
) -> list[dict[str, Any]]:
    """Project retrieval hits into a compact structure for the judge payload."""
    projected: list[dict[str, Any]] = []
    for hit in hits[:max_docs]:
        projected.append(
            {
                "doc_id": hit.get("doc_id") or hit.get("CN") or hit.get("cn") or "",
                "rank": hit.get("rank"),
                "title": hit.get("title", ""),
                "abstract": hit.get("abstract") or hit.get("text") or "",
            }
        )
    return projected


def compute_retrieval_metrics(gold_doc_id: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    gold_norm = str(gold_doc_id).strip()
    rank = None
    for idx, hit in enumerate(hits, 1):
        doc_id = str(hit.get("doc_id") or hit.get("CN") or hit.get("cn") or "").strip()
        if doc_id == gold_norm:
            rank = idx
            break
    return {
        "gold_rank": rank,
        "hit_at_1": bool(rank and rank <= 1),
        "hit_at_3": bool(rank and rank <= 3),
        "hit_at_5": bool(rank and rank <= 5),
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def call_judge(
    prompt: str,
    *,
    backend: str,
    model: str,
    vllm_url: str,
    api_key: str | None,
    max_tokens: int,
    openai_base_url: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    if backend == "vllm":
        from openai import OpenAI  # lazy

        client = OpenAI(base_url=vllm_url, api_key="dummy-key")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
    if backend == "openai":
        from openai import OpenAI  # lazy

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key is required via --api-key or OPENAI_API_KEY.")
        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        client = OpenAI(**client_kwargs)
        request: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if reasoning_effort:
            request["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**request)
        text = getattr(response, "output_text", None)
        if text:
            return str(text).strip()
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(str(value))
        return "".join(chunks).strip()
    if backend == "gemini":
        import google.generativeai as genai  # lazy

        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                candidate_count=1,
                max_output_tokens=max_tokens,
            ),
        )
        return (response.text or "").strip()
    if backend == "codex":
        from shrag.llm.codex_client import CodexClient  # lazy

        client = CodexClient(model=model, timeout_sec=600)
        return client.generate_answer_with_prompt(prompt, max_tokens=max_tokens)
    raise ValueError(f"Unsupported judge backend: {backend}")


def parse_judge_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except Exception:
        return {
            "answer_relevance": 0,
            "evidence_coverage": 0,
            "faithfulness_to_gold": 0,
            "faithfulness_to_retrieved": 0,
            "specificity": 0,
            "grounding_flag": "parse_error",
            "escape_used": False,
            "overall": 0,
            "label": "parse_error",
            "missing_key_points": [],
            "unsupported_claims": [],
            "rationale": "Failed to parse judge response as JSON.",
            "raw_response": raw,
        }
    parsed.setdefault("raw_response", raw)
    return parsed


def clamp_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def server_label(parsed: dict[str, Any], pipeline_answer: str) -> str:
    """Decide label from scores rather than trusting the judge's self-reported label.

    Priority order: parse_error -> api_error -> dry_run -> abstain -> incorrect -> correct -> partial.
    """
    status = parsed.get("label")
    if status in {"parse_error", "api_error", "dry_run"}:
        return str(status)
    answer_text = (pipeline_answer or "").strip()
    if answer_text in ESCAPE_SENTENCES or parsed.get("escape_used") is True:
        return "abstain"
    overall = clamp_int(parsed.get("overall"))
    faithfulness_gold = clamp_int(parsed.get("faithfulness_to_gold"))
    coverage = clamp_int(parsed.get("evidence_coverage"))
    relevance = clamp_int(parsed.get("answer_relevance"))
    if overall <= 3 or faithfulness_gold == 0 or relevance == 0:
        return "incorrect"
    if overall >= 8 and faithfulness_gold >= 2 and coverage >= 2 and relevance >= 1:
        return "correct"
    return "partial"


def normalize_judge_row(parsed: dict[str, Any], pipeline_answer: str) -> dict[str, Any]:
    judge_label = str(parsed.get("label", "unknown"))
    computed = server_label(parsed, pipeline_answer)
    parsed["judge_label"] = judge_label
    parsed["server_computed_label"] = computed
    parsed["label"] = computed
    if computed in {"parse_error", "api_error", "dry_run"}:
        parsed["label_mismatch"] = False
    else:
        parsed["label_mismatch"] = judge_label not in {computed, "unknown", ""}
    return parsed


def summarize(
    rows: list[dict[str, Any]],
    missing_qids: list[str],
    total_gold_count: int,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    labels: dict[str, int] = {}
    grounding_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label", "unknown"))
        labels[label] = labels.get(label, 0) + 1
        grounding = str(row.get("grounding_flag", "unknown"))
        grounding_counts[grounding] = grounding_counts.get(grounding, 0) + 1

    def _nums(key: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            v = row.get(key)
            if isinstance(v, (int, float)):
                values.append(float(v))
            elif isinstance(v, str) and v.replace(".", "", 1).isdigit():
                values.append(float(v))
        return values

    def _mean(values: list[float]) -> float:
        return round(statistics.mean(values), 4) if values else 0.0

    hit_at_1 = sum(1 for row in rows if row["retrieval"]["hit_at_1"])
    hit_at_3 = sum(1 for row in rows if row["retrieval"]["hit_at_3"])
    hit_at_5 = sum(1 for row in rows if row["retrieval"]["hit_at_5"])
    mrr_sum = sum(float(row["retrieval"]["mrr"]) for row in rows)
    n_eval = len(rows)

    return {
        "evaluated_count": n_eval,
        "total_gold_count": total_gold_count,
        "missing_prediction_count": len(missing_qids),
        "missing_prediction_qids": missing_qids,
        "label_counts": labels,
        "grounding_flag_counts": grounding_counts,
        "escape_used_count": sum(1 for row in rows if row.get("escape_used") is True),
        "overall_avg": _mean(_nums("overall")),
        "answer_relevance_avg": _mean(_nums("answer_relevance")),
        "evidence_coverage_avg": _mean(_nums("evidence_coverage")),
        "faithfulness_to_gold_avg": _mean(_nums("faithfulness_to_gold")),
        "faithfulness_to_retrieved_avg": _mean(_nums("faithfulness_to_retrieved")),
        "specificity_avg": _mean(_nums("specificity")),
        "hit_at_1_on_evaluated": round(hit_at_1 / n_eval, 4) if n_eval else 0.0,
        "hit_at_3_on_evaluated": round(hit_at_3 / n_eval, 4) if n_eval else 0.0,
        "hit_at_5_on_evaluated": round(hit_at_5 / n_eval, 4) if n_eval else 0.0,
        "hit_at_1_on_full_gold": round(hit_at_1 / total_gold_count, 4)
        if total_gold_count
        else 0.0,
        "hit_at_3_on_full_gold": round(hit_at_3 / total_gold_count, 4)
        if total_gold_count
        else 0.0,
        "hit_at_5_on_full_gold": round(hit_at_5 / total_gold_count, 4)
        if total_gold_count
        else 0.0,
        "mrr_on_evaluated": round(mrr_sum / n_eval, 4) if n_eval else 0.0,
        "mrr_on_full_gold": round(mrr_sum / total_gold_count, 4)
        if total_gold_count
        else 0.0,
        "label_mismatch_count": sum(1 for row in rows if row.get("label_mismatch")),
        "api_error_count": sum(1 for row in rows if row.get("label") == "api_error"),
        "parse_error_count": sum(1 for row in rows if row.get("label") == "parse_error"),
        "warnings": warnings or [],
    }


def write_review_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    scored = [
        row
        for row in rows
        if isinstance(row.get("overall"), (int, float))
        or (
            isinstance(row.get("overall"), str)
            and row.get("overall", "").replace(".", "", 1).isdigit()
        )
    ]
    worst = sorted(scored, key=lambda row: float(row.get("overall", 0)))[:5]
    error_rows = [
        row
        for row in rows
        if row.get("label") in {"parse_error", "api_error"} or row.get("label_mismatch")
    ]
    ungrounded = [row for row in rows if row.get("grounding_flag") == "ungrounded"]
    lines = [
        "# Gold Judge Review",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Worst Overall Scores",
        "",
    ]
    if worst:
        for row in worst:
            lines.append(
                f"- qid `{row['qid']}` overall `{row.get('overall')}` label `{row.get('label')}`: {row.get('gold_title')}"
            )
            rationale = str(row.get("rationale", "")).strip()
            if rationale:
                lines.append(f"  - rationale: {rationale}")
    else:
        lines.append("- No scored rows available.")
    lines.extend(["", "## Ungrounded Answers (possible prior-knowledge leakage)", ""])
    if ungrounded:
        for row in ungrounded:
            lines.append(
                f"- qid `{row['qid']}` faithfulness_to_retrieved `{row.get('faithfulness_to_retrieved')}` "
                f"faithfulness_to_gold `{row.get('faithfulness_to_gold')}` gold_title: {row.get('gold_title')}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Errors And Mismatches", ""])
    if error_rows:
        for row in error_rows:
            lines.append(
                f"- qid `{row['qid']}` label `{row.get('label')}` judge_label `{row.get('judge_label')}` "
                f"server_label `{row.get('server_computed_label')}` mismatch `{row.get('label_mismatch')}`"
            )
            if row.get("error_message"):
                lines.append(f"  - error: {row.get('error_type')}: {row.get('error_message')}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FULL ScienceON gold answers with an LLM judge.")
    parser.add_argument("--final-dir", required=True, help="Pipeline final output directory.")
    parser.add_argument(
        "--gold",
        default="experiments/gold_scienceon/artifacts/scienceon_gold_full_only.jsonl",
        help="FULL gold JSONL file.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for judge outputs.")
    parser.add_argument("--judge-backend", choices=["vllm", "gemini", "openai", "codex"], default="vllm")
    parser.add_argument("--judge-model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--generation-model",
        default=None,
        help="Optional answer-generation model name for self-judge warnings.",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for the openai backend. Leave unset for api.openai.com.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning effort for OpenAI reasoning models such as gpt-5.4.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--retrieved-max-docs",
        type=int,
        default=5,
        help="How many top retrieval hits to include in retrieved_context passed to the judge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call LLM; emit parse/retrieval-only rows.",
    )
    return parser.parse_args()


def _dry_run_parsed(answer_text: str) -> dict[str, Any]:
    return {
        "answer_relevance": None,
        "evidence_coverage": None,
        "faithfulness_to_gold": None,
        "faithfulness_to_retrieved": None,
        "specificity": None,
        "grounding_flag": "dry_run",
        "escape_used": answer_text.strip() in ESCAPE_SENTENCES,
        "overall": None,
        "label": "dry_run",
        "missing_key_points": [],
        "unsupported_claims": [],
        "rationale": "Dry run; judge was not called.",
    }


def _api_error_parsed(exc: BaseException) -> dict[str, Any]:
    return {
        "answer_relevance": 0,
        "evidence_coverage": 0,
        "faithfulness_to_gold": 0,
        "faithfulness_to_retrieved": 0,
        "specificity": 0,
        "grounding_flag": "api_error",
        "escape_used": False,
        "overall": 0,
        "label": "api_error",
        "missing_key_points": [],
        "unsupported_claims": [],
        "rationale": "Judge API call failed.",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "raw_response": "",
    }


def main() -> None:
    args = parse_args()
    started_at = datetime.now().astimezone()
    gold_path = Path(args.gold)
    gold_rows = load_jsonl(gold_path)
    final_files = find_final_files(Path(args.final_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if args.generation_model and args.generation_model.strip() == args.judge_model.strip():
        warnings.append(
            f"judge model matches generation model ({args.judge_model}); "
            "use a different model family when possible."
        )
        print(f"WARNING: {warnings[-1]}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    missing_qids: list[str] = []
    results_path = output_dir / "judge_results.jsonl"
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "git_sha": git_sha(),
        "gold_file": str(gold_path),
        "gold_file_sha1": sha1_file(gold_path),
        "final_dir": str(Path(args.final_dir)),
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model,
        "generation_model": args.generation_model,
        "openai_base_url": args.openai_base_url,
        "reasoning_effort": args.reasoning_effort,
        "cli_args": vars(args),
        "started_at": started_at.isoformat(),
        "finished_at": None,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out = results_path.open("w", encoding="utf-8", newline="\n")
    for gold in gold_rows:
        qid = str(gold["qid"])
        pred_path = final_files.get(qid)
        if pred_path is None:
            missing_qids.append(qid)
            continue
        prediction = load_prediction(pred_path)
        hits = original_hits_from_final(prediction["raw"])
        retrieval = compute_retrieval_metrics(gold["gold_doc_id"], hits)
        retrieved_ctx = retrieved_context_for_judge(hits, max_docs=args.retrieved_max_docs)
        judge_payload = {
            "qid": qid,
            "question": gold["question"],
            "gold_document": {
                "doc_id": gold["gold_doc_id"],
                "title": gold["gold_title"],
                "abstract": gold["gold_abstract"],
            },
            "retrieved_context": retrieved_ctx,
            "pipeline_answer": prediction["answer"],
        }
        prompt = build_gold_answer_judge_prompt(judge_payload)
        if args.dry_run:
            parsed = _dry_run_parsed(prediction["answer"])
        else:
            try:
                raw_response = call_judge(
                    prompt,
                    backend=args.judge_backend,
                    model=args.judge_model,
                    vllm_url=args.vllm_url,
                    api_key=args.api_key,
                    max_tokens=args.max_tokens,
                    openai_base_url=args.openai_base_url,
                    reasoning_effort=args.reasoning_effort,
                )
                parsed = parse_judge_json(raw_response)
            except Exception as exc:  # noqa: BLE001
                parsed = _api_error_parsed(exc)
        parsed = normalize_judge_row(parsed, prediction["answer"])
        parsed.update(
            {
                "qid": qid,
                "question": gold["question"],
                "gold_doc_id": gold["gold_doc_id"],
                "gold_title": gold["gold_title"],
                "gold_abstract": gold["gold_abstract"],
                "pipeline_answer": prediction["answer"],
                "answer_length_chars": len(prediction["answer"]),
                "retrieved_doc_ids": [ctx.get("doc_id") for ctx in retrieved_ctx],
                "prompt_sha1": sha1_text(prompt),
                "response_sha1": sha1_text(str(parsed.get("raw_response", ""))),
                "prediction_file": str(pred_path),
                "retrieval": retrieval,
                "judge_backend": args.judge_backend,
                "judge_model": args.judge_model,
                "prompt_version": PROMPT_VERSION,
            }
        )
        results.append(parsed)
        out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        out.flush()
    out.close()
    summary = summarize(
        results,
        missing_qids,
        total_gold_count=len(gold_rows),
        warnings=warnings,
    )
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["runtime_sec"] = round(time.time() - started_at.timestamp(), 4)
    manifest["summary"] = summary
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_review_markdown(output_dir / "judge_review.md", results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
