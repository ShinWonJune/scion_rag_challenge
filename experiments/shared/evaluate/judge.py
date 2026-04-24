from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import google.generativeai as genai
from openai import OpenAI

from experiments.shared.prompts.judge_pointwise import build_judge_prompt


def _extract_retrieved_contexts(retrieval_payload: dict[str, Any]) -> list[dict[str, Any]]:
    original = next(
        (
            item
            for item in retrieval_payload.get("retrieval_results", [])
            if item.get("query_meta", {}).get("type") == "original"
        ),
        None,
    )
    if not original:
        return []
    return original.get("hits", [])


def _call_judge(payload: dict[str, Any], backend: str, model: str, base_url: str, api_key: str | None) -> str:
    prompt = build_judge_prompt(payload)
    if backend == "vllm":
        client = OpenAI(base_url=base_url, api_key="dummy-key")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
    if backend == "gemini":
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.0, candidate_count=1),
        )
        return (response.text or "").strip()
    raise ValueError(f"Unsupported judge backend: {backend}")


def _parse_judge_response(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except Exception:
        return {
            "context_relevance": 1,
            "answer_faithfulness": 1,
            "answer_completeness": 1,
            "unsupported_claims": [],
            "missing_key_points": [],
            "verdict": "fail_parse",
            "raw_response": raw_response,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for RAG predictions")
    parser.add_argument("--predictions", required=True, help="predictions.json path")
    parser.add_argument("--retrieval_dir", required=True, help="Retrieval JSON directory")
    parser.add_argument("--judge_backend", choices=["gemini", "vllm"], default="gemini")
    parser.add_argument("--judge_model", default="gemini-2.5-flash")
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    retrieval_dir = Path(args.retrieval_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for prediction in predictions:
            qid = str(prediction.get("question_id") or prediction.get("id"))
            retrieval_candidates = sorted(retrieval_dir.glob(f"{qid}_*.json"))
            if not retrieval_candidates:
                continue
            retrieval_payload = json.loads(retrieval_candidates[0].read_text(encoding="utf-8"))
            judge_input = {
                "question_id": qid,
                "question": prediction.get("question") or "",
                "retrieved_contexts": _extract_retrieved_contexts(retrieval_payload),
                "answer": prediction.get("answer") or prediction.get("result") or "",
                "gold_answer": prediction.get("gold_answer"),
                "gold_doc_titles": prediction.get("gold_doc_titles"),
            }
            raw_response = _call_judge(
                judge_input,
                backend=args.judge_backend,
                model=args.judge_model,
                base_url=args.vllm_url,
                api_key=args.api_key,
            )
            parsed = _parse_judge_response(raw_response)
            parsed["question_id"] = qid
            parsed["judge_model"] = args.judge_model
            parsed.setdefault("raw_response", raw_response)
            out.write(json.dumps(parsed, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
