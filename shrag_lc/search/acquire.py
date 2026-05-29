"""Document acquisition: keywords -> search terms -> platform search -> corpus.

Mirrors ``shrag/pipeline/steps/step1_search.py``: term-major search that
accumulates + dedups until the per-query target (m) is met, run separately for
Korean and English terms then merged. Returns LangChain Documents ready for
embedding (page_content = 3T+A text).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from ..config import PipelineConfig
from ..embeddings import doc_to_document
from ..schemas import PipelineError, StageName, StageStatus, stage_report
from .clients import SearchClient, _dedup
from .keywords import KeywordExtractor
from .terms import build_search_terms_by_lang

logger = logging.getLogger(__name__)


def _request_stats(client: SearchClient) -> dict[str, Any]:
    getter = getattr(client, "get_request_stats", None)
    if not callable(getter):
        return {}
    try:
        return dict(getter() or {})
    except Exception:  # noqa: BLE001
        return {}


def _stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | float]:
    delta: dict[str, int | float] = {}
    for key in ["request_count", "rate_limit_count", "api_error_count"]:
        b = before.get(key, 0) or 0
        a = after.get(key, 0) or 0
        try:
            diff = a - b
        except TypeError:
            continue
        if diff:
            delta[key] = diff
    return delta


@dataclass
class SearchAttemptResult:
    documents: list[dict] = field(default_factory=list)
    attempted_terms: list[str] = field(default_factory=list)
    failed_terms: list[dict[str, Any]] = field(default_factory=list)
    errors: list[PipelineError] = field(default_factory=list)


def _search_until_target(
    client: SearchClient, terms: list[str], target: int, lang: str
) -> SearchAttemptResult:
    collected: list[dict] = []
    result = SearchAttemptResult()
    for term in terms:
        if len(collected) >= target:
            break
        result.attempted_terms.append(term)
        remaining = target - len(collected)
        try:
            docs = client.search([term], max_results=remaining)
        except Exception as e:  # noqa: BLE001
            logger.error("%s search failed (lang=%s, term=%s): %s", client.source_name, lang, term, e)
            error = PipelineError.from_exception(
                StageName.SEARCH,
                e,
                source=client.source_name,
                term=term,
                language=lang,
            )
            result.errors.append(error)
            result.failed_terms.append(error.to_dict())
            continue
        collected = _dedup(collected + docs, key="doc_id")
    result.documents = collected[:target]
    return result


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
        stats_before = _request_stats(self.client)
        ko_result = _search_until_target(self.client, ko_terms, target, "korean")
        en_result = _search_until_target(self.client, en_terms, target, "english")
        stats_after = _request_stats(self.client)
        request_stats_delta = _stats_delta(stats_before, stats_after)
        ko_docs = ko_result.documents
        en_docs = en_result.documents
        merged = _dedup(ko_docs + en_docs, key="doc_id")
        attempted_terms = ko_result.attempted_terms + en_result.attempted_terms
        errors = ko_result.errors + en_result.errors
        warnings: list[str] = []
        rate_limit_delta = int(request_stats_delta.get("rate_limit_count", 0) or 0)
        if rate_limit_delta:
            warnings.append(f"Search client reported {rate_limit_delta} rate-limit response(s).")

        if merged and errors:
            search_status = StageStatus.PARTIAL_SUCCESS
        elif merged:
            search_status = StageStatus.SUCCESS
        elif attempted_terms and errors and len(errors) == len(attempted_terms):
            search_status = StageStatus.FAILED
        elif attempted_terms and rate_limit_delta:
            search_status = StageStatus.FAILED
        else:
            search_status = StageStatus.NO_DOCUMENTS

        documents = [doc_to_document(d, self.cfg.embedding_mode) for d in merged]
        meta = {
            "query": query,
            "source": self.client.source_name,
            "keywords": keywords,
            "search_terms": ko_terms + en_terms,
            "document_count": len(documents),
            "failed_terms": ko_result.failed_terms + en_result.failed_terms,
            "stages": {
                StageName.SEARCH: stage_report(
                    StageName.SEARCH,
                    search_status,
                    counts={
                        "attempted_terms": len(attempted_terms),
                        "failed_terms": len(errors),
                        "document_count": len(documents),
                    },
                    warnings=warnings,
                    errors=errors,
                    metadata={
                        "source": self.client.source_name,
                        "request_stats": stats_after,
                        "request_stats_delta": request_stats_delta,
                    },
                ).to_dict()
            },
        }
        return documents, meta
