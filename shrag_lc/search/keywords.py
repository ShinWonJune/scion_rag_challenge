"""Keyword extraction as an LCEL chain.

Ported from ``shrag/search/extractors/llm_keyword_extractor.py``: runs the
Korean and/or English prompt on every query (cross-lingual recall), parses the
comma-list output (comma -> whitespace/hyphen split, dedup, max 10).

Chain shape per language:  prompt | chat_model | StrOutputParser | parse
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from ..prompts import build_keyword_prompt

_KEYWORD_SPLIT_RE = re.compile(r"[\s\-]+")
logger = logging.getLogger(__name__)


def parse_keyword_list(text: str, prefix: str = "") -> List[str]:
    """Comma-list -> deduped keyword tokens (whitespace/hyphen split, max 10)."""
    text = (text or "").strip()
    if prefix:
        text = text.replace(prefix, "").strip()
    keywords = [kw.strip() for kw in text.split(",")]
    keywords = [kw for kw in keywords if kw and len(kw) > 1]

    final: List[str] = []
    seen: set[str] = set()
    for kw in keywords:
        for word in _KEYWORD_SPLIT_RE.split(kw):
            word = word.strip()
            if not word or len(word) <= 1:
                continue
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            final.append(word)
    return final[:10]


def _build_lang_chain(llm: BaseChatModel, lang: str, prefix: str) -> Runnable:
    prompt = build_keyword_prompt(lang)
    return prompt | llm | StrOutputParser() | RunnableLambda(
        lambda raw, _p=prefix: parse_keyword_list(raw, prefix=_p)
    )


class KeywordExtractor:
    """LLM keyword extractor returning ``{"korean": [...], "english": [...]}``."""

    def __init__(self, llm: BaseChatModel, language: str = "all"):
        if language not in {"korean", "english", "all"}:
            raise ValueError(f"language must be korean/english/all, got {language!r}")
        self.language = language
        self._ko_chain = _build_lang_chain(llm, "korean", "키워드:")
        self._en_chain = _build_lang_chain(llm, "english", "Keywords:")

    def extract(self, query: str) -> Dict[str, List[str]]:
        korean: List[str] = []
        english: List[str] = []
        if self.language in {"korean", "all"}:
            try:
                korean = self._ko_chain.invoke({"query": query})
            except Exception as e:  # noqa: BLE001
                logger.error("Korean keyword extraction failed: %s", e)
        if self.language in {"english", "all"}:
            try:
                english = self._en_chain.invoke({"query": query})
            except Exception as e:  # noqa: BLE001
                logger.error("English keyword extraction failed: %s", e)
        return {"korean": korean, "english": english}
