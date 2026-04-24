from __future__ import annotations

import json


def build_judge_prompt(payload: dict) -> str:
    return (
        "You are evaluating a RAG answer.\n"
        "Use only the provided contexts. Do not use outside knowledge.\n"
        "Do not reward longer answers. Penalize unsupported claims.\n"
        "Return JSON with keys: context_relevance, answer_faithfulness, answer_completeness, "
        "unsupported_claims, missing_key_points, verdict.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
