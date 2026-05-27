"""Answer generation: serialize reranked Documents to the Context JSON the
grounding prompt expects, then run ``prompt | llm | StrOutputParser``.

Mirrors ``shrag/pipeline/_impl/generate.py`` — the context is a JSON object
with a ``queries`` list; the 'original' query carries the ranked ``hits`` the
model must cite from.
"""

from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .prompts import build_answer_prompt


def build_context_json(query: str, docs: list[Document]) -> str:
    """Build the Context JSON string (queries -> hits) for the prompt."""
    hits = []
    for doc in docs:
        meta = doc.metadata
        hits.append(
            {
                "rank": meta.get("rank"),
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "abstract": meta.get("abstract", "") or doc.page_content,
                "score": meta.get("rerank_score"),
            }
        )
    context = {
        "queries": [
            {"query": query, "query_meta": {"type": "original"}, "hits": hits}
        ]
    }
    return json.dumps(context, ensure_ascii=False)


class AnswerGenerator:
    """Grounded answer generator over reranked Documents."""

    def __init__(self, llm: BaseChatModel):
        self.chain = build_answer_prompt() | llm | StrOutputParser()

    def generate(self, query: str, docs: list[Document]) -> dict:
        context_json = build_context_json(query, docs)
        answer = self.chain.invoke({"context": context_json, "query": query})
        return {
            "question": query,
            "answer": answer,
            "used_context": [
                {"rank": d.metadata.get("rank"), "doc_id": d.metadata.get("doc_id", "")}
                for d in docs
            ],
        }
