"""Keyword extraction as an LCEL chain.

Ported from ``shrag/search/extractors/llm_keyword_extractor.py``: runs the
Korean and/or English prompt on every query (cross-lingual recall), parses the
comma-list output (comma -> whitespace/hyphen split, dedup, max 10).

Chain shape per language:  prompt | chat_model | StrOutputParser | parse
"""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from ..prompts import build_keyword_prompt, build_structured_keyword_prompt
from ..schemas import KeywordList, normalize_keyword_values

logger = logging.getLogger(__name__)


def parse_keyword_list(text: str, prefix: str = "") -> List[str]:
    """Comma-list -> deduped keyword tokens (whitespace/hyphen split, max 10)."""
    text = (text or "").strip()
    if prefix:
        text = text.replace(prefix, "").strip()
    keywords = [kw.strip() for kw in text.split(",")]
    keywords = [kw for kw in keywords if kw and len(kw) > 1]
    return normalize_keyword_values(keywords)


def _build_lang_chain(llm: BaseChatModel, lang: str, prefix: str) -> Runnable:
    prompt = build_keyword_prompt(lang)
    return prompt | llm | StrOutputParser() | RunnableLambda(
        lambda raw, _p=prefix: parse_keyword_list(raw, prefix=_p)
    )


def _build_structured_lang_chain(llm: BaseChatModel, lang: str) -> Runnable:
    prompt = build_structured_keyword_prompt(lang)
    parser = PydanticOutputParser(pydantic_object=KeywordList)
    return (prompt | llm | parser | RunnableLambda(lambda parsed: parsed.keywords)).with_retry(
        stop_after_attempt=3
    )


class KeywordExtractor:
    """LLM keyword extractor returning ``{"korean": [...], "english": [...]}``."""

    def __init__(self, llm: BaseChatModel, language: str = "all"):
        if language not in {"korean", "english", "all"}:
            raise ValueError(f"language must be korean/english/all, got {language!r}")
        self.language = language
        self._structured_ko_chain = _build_structured_lang_chain(llm, "korean")
        self._structured_en_chain = _build_structured_lang_chain(llm, "english")
        self._ko_chain = _build_lang_chain(llm, "korean", "키워드:")
        self._en_chain = _build_lang_chain(llm, "english", "Keywords:")

    def extract(self, query: str) -> Dict[str, List[str]]:
        korean: List[str] = []
        english: List[str] = []
        if self.language in {"korean", "all"}:
            try:
                korean = self._structured_ko_chain.invoke({"query": query})
            except Exception as e:  # noqa: BLE001
                logger.error("Structured Korean keyword extraction failed: %s", e)
                try:
                    korean = self._ko_chain.invoke({"query": query})
                except Exception as fallback_e:  # noqa: BLE001
                    logger.error("Legacy Korean keyword extraction failed: %s", fallback_e)
        if self.language in {"english", "all"}:
            try:
                english = self._structured_en_chain.invoke({"query": query})
            except Exception as e:  # noqa: BLE001
                logger.error("Structured English keyword extraction failed: %s", e)
                try:
                    english = self._en_chain.invoke({"query": query})
                except Exception as fallback_e:  # noqa: BLE001
                    logger.error("Legacy English keyword extraction failed: %s", fallback_e)
        return {"korean": korean, "english": english}
