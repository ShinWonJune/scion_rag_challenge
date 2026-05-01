"""Closed-book answer prompt — no retrieved context.

Mirrors the SHRAG `build_prompt` output format (Korean/English section headers,
plain text, no markdown emphasis) but does not reference any retrieved
documents. The model must answer purely from its parametric knowledge.

Used by the SHRAG vs Closed-book ablation study to isolate the contribution of
the retrieval pipeline.
"""
from __future__ import annotations


def build_closed_book_prompt(query: str) -> str:
    template = """You are an answering model. Provide a factual answer to the Question below using your own knowledge. There is no external context — answer from what you already know.

== ANSWER RULES ==
1. Provide a factual, accurate answer. Do not fabricate facts.
2. If you do not have sufficient knowledge to answer reliably, respond with exactly ONE of:
   - For a Korean question: 답할 정보가 부족합니다.
   - For an English question: I do not have sufficient information to answer.
   In that case, do NOT produce section headers and do NOT guess or fabricate.
3. Do NOT pad. Prefer accuracy and brevity over length.
4. Do NOT cite documents — there is no retrieved context. Just answer directly.

== OUTPUT FORMAT ==
If the Question is in Korean, structure the answer with:
##제목##

##서론##

##본론##

##결론##

If the Question is in English, structure the answer with:
##{{Title}}##

##Introduction##

##Main Body##

##Conclusion##

Plain text only. Do not use "*", "-", or "**" for emphasis. Do not use markdown.
If Rule 2 applies (insufficient knowledge), skip the section headers and output only the single insufficient-information sentence.

--- Question ---
{query}
"""
    return template.format(query=query)
