"""Document acquisition: keywords -> search terms -> platform search -> corpus.

Mirrors ``shrag/pipeline/steps/step1_search.py``: term-major search that
accumulates + dedups until the per-query target (m) is met, run separately for
Korean and English terms then merged. Returns LangChain Documents ready for
embedding (page_content = 3T+A text).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from ..config import PipelineConfig
from ..embeddings import doc_to_document
from .clients import SearchClient, _dedup
from .keywords import KeywordExtractor
from .terms import build_search_terms_by_lang

logger = logging.getLogger(__name__)


def _search_until_target(
    client: SearchClient, terms: list[str], target: int, lang: str
) -> list[dict]:
    collected: list[dict] = []
    for term in terms:
        if len(collected) >= target:
            break
        remaining = target - len(collected)
        try:
            docs = client.search([term], max_results=remaining)
        except Exception as e:  # noqa: BLE001
            logger.error("%s search failed (lang=%s, term=%s): %s", client.source_name, lang, term, e)
            continue
        collected = _dedup(collected + docs, key="doc_id")
    return collected[:target]


class Acquirer:
    """Acquire a per-query document corpus from a single platform source."""

    def __init__(self, extractor: KeywordExtractor, client: SearchClient, cfg: PipelineConfig):
        self.extractor = extractor
        self.client = client
        self.cfg = cfg

    def acquire(self, query: str) -> tuple[list[Document], dict[str, Any]]:
        try:
            keywords = self.extractor.extract(query)
        except Exception as e:  # noqa: BLE001
            logger.error("Keyword extraction failed for %r: %s", query, e)
            keywords = {"korean": [], "english": [query]}

        terms_by_lang = build_search_terms_by_lang(keywords, self.cfg.number_of_operators)
        ko_terms = terms_by_lang.get("korean") or []
        en_terms = terms_by_lang.get("english") or []

        target = self.cfg.target_documents
        ko_docs = _search_until_target(self.client, ko_terms, target, "korean")
        en_docs = _search_until_target(self.client, en_terms, target, "english")
        merged = _dedup(ko_docs + en_docs, key="doc_id")

        documents = [doc_to_document(d, self.cfg.embedding_mode) for d in merged]
        meta = {
            "query": query,
            "source": self.client.source_name,
            "keywords": keywords,
            "search_terms": ko_terms + en_terms,
            "document_count": len(documents),
        }
        return documents, meta
