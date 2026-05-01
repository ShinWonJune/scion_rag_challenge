from __future__ import annotations

import json
from typing import Any


def build_gold_answer_judge_prompt(payload: dict[str, Any]) -> str:
    """Gold answer judge prompt v2.

    Changes from v1:
      - Adds `retrieved_context` to the payload.
      - Splits faithfulness into
          faithfulness_to_gold       (answer vs gold abstract)
          faithfulness_to_retrieved  (answer claims vs retrieved context shown to pipeline)
      - Adds grounding_flag and escape_used helper fields.
      - overall computation hint.
      - Server-side 'label' is NOT decided by the judge; the runner computes it.
    """
    return (
        "You are a strict evaluator for a ScienceON RAG answer.\n"
        "You receive: the Question, the gold document (title + abstract), the retrieved_context that was actually shown to the pipeline, and the pipeline_answer.\n"
        "Judge only against the gold and the retrieved_context provided here. Do not use outside knowledge.\n"
        "Do not prefer longer answers over shorter answers if both are equally supported by the evidence.\n"
        "Write the rationale in Korean.\n\n"
        "Axes:\n"
        "- answer_relevance (0-2): 0 not answering, 1 partly answers, 2 directly answers the Question.\n"
        "- evidence_coverage (0-3): how many key points from the gold abstract needed to answer the Question are present in the pipeline_answer. 0 none, 1 minor, 2 most, 3 all/near-all.\n"
        "- faithfulness_to_gold (0-3): 0 contradicted/hallucinated vs gold, 1 major unsupported claims, 2 minor unsupported claims, 3 fully consistent with gold.\n"
        "- faithfulness_to_retrieved (0-3): how many of the pipeline_answer's factual claims are traceable to the retrieved_context. 0 none (or retrieved_context is empty and the answer is non-trivial), 1 minor, 2 most, 3 all/near-all. Inline citations such as [doc_id=...] or [rank=N] count as traceability signals when they actually match a retrieved hit.\n"
        "- specificity (0-2): 0 generic, 1 some document-specific details, 2 concrete details drawn from gold or retrieved docs.\n"
        "- grounding_flag: derived flag. Set as:\n"
        "    'no_context' if retrieved_context is empty;\n"
        "    'grounded' if faithfulness_to_retrieved >= 2;\n"
        "    'partially_grounded' if faithfulness_to_retrieved == 1;\n"
        "    'ungrounded' if faithfulness_to_retrieved == 0 AND the pipeline_answer is a non-trivial answer (not the insufficient-information sentence).\n"
        "- escape_used: true if the pipeline_answer is exactly one of:\n"
        "    '제공된 문서에서는 이 질문에 답할 정보를 찾을 수 없습니다.'\n"
        "    'The provided documents do not contain sufficient information to answer this question.'\n"
        "  else false.\n"
        "- overall (0-10): weighted composite. Rule of thumb: 2*answer_relevance + evidence_coverage + faithfulness_to_gold + specificity, capped at 10. Adjust by at most 1 point when the rubric clearly requires.\n\n"
        "Return ONLY valid JSON with this exact schema. Do NOT include a 'label' field; label is computed server-side.\n"
        "{\n"
        '  "answer_relevance": 0|1|2,\n'
        '  "evidence_coverage": 0|1|2|3,\n'
        '  "faithfulness_to_gold": 0|1|2|3,\n'
        '  "faithfulness_to_retrieved": 0|1|2|3,\n'
        '  "specificity": 0|1|2,\n'
        '  "grounding_flag": "grounded"|"partially_grounded"|"ungrounded"|"no_context",\n'
        '  "escape_used": true|false,\n'
        '  "overall": 0-10,\n'
        '  "missing_key_points": ["..."],\n'
        '  "unsupported_claims": ["..."],\n'
        '  "rationale": "..."\n'
        "}\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
