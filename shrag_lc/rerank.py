"""Cross-encoder reranking as a LangChain ``BaseDocumentCompressor``.

Ported from ``experiments/shared/rerank/cross_encoder.py``: pairs the query
with each candidate's ``title\\nabstract``, scores with a cross-encoder, sorts
descending, and keeps the top-N. Default model & top_n follow BEST_CONFIG
(dragonkue/bge-reranker-v2-m3-ko, candidates=5 -> top-3).

Implemented as a self-contained compressor (no dependency on the legacy
``langchain_classic`` shim that ships LangChain 1.x's CrossEncoderReranker).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from .config import PipelineConfig
from .embeddings import _resolve_device


class CrossEncoderRerankCompressor(BaseDocumentCompressor):
    """Rerank documents by cross-encoder relevance to the query.

    ``model`` is any object exposing ``score(list[(str, str)]) -> list[float]``
    (e.g. ``HuggingFaceCrossEncoder``); typed ``Any`` so stubs work in tests.
    """

    model: Any
    top_n: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        docs = list(documents)
        if not docs:
            return []
        # Match the original (cross_encoder.py): pair = query vs "title\nabstract".
        # Do NOT fall back to page_content — that is the 3T+A embedding text and
        # would pollute the cross-encoder input for abstract-less documents.
        pairs = [
            (query, f"{d.metadata.get('title', '')}\n{d.metadata.get('abstract', '')}".strip())
            for d in docs
        ]
        scores = self.model.score(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        out: list[Document] = []
        for rank, (doc, score) in enumerate(ranked[: self.top_n], start=1):
            out.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_score": float(score), "rank": rank},
                )
            )
        return out


def build_reranker(cfg: PipelineConfig) -> CrossEncoderRerankCompressor:
    model = HuggingFaceCrossEncoder(
        model_name=cfg.reranker_model,
        model_kwargs={"device": _resolve_device(cfg.device)},
    )
    return CrossEncoderRerankCompressor(model=model, top_n=cfg.rerank_top_n)
