"""Unified LLM-based keyword extractor.

Single base class that defines:
  - the Korean keyword prompt
  - the English keyword prompt
  - the comma-list + space-split + dedup parsing logic
  - the search-term builder hookup (`search_terms.build_search_terms`)

Subclasses ONLY override `_call_llm(prompt: str) -> str` to swap the LLM
backend (Gemini, ChatGPT, vLLM, Codex CLI, ...). Both prompts always run on
every query (regardless of question language) because ScienceON gold docs may
be in the other language than the question — cross-lingual retrieval requires
both Korean and English keyword sets.

Source of the two prompt templates: `keyword_extractor.py`'s original
`_extract_korean_keywords` / `_extract_english_keywords`. Reproduced here
verbatim so all backends share an identical prompt surface.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from .base_extractor import BaseKeywordExtractor

# 공백(\s) 또는 하이픈(-)을 한꺼번에 분리. 다른 특수기호는 추가 요청 시 확장.
_KEYWORD_SPLIT_RE = re.compile(r"[\s\-]+")

KOREAN_PROMPT_TEMPLATE = """
당신은 검색을 위한 키워드 추출 전문가입니다. 주어진 질문에서 가장 중요하고 검색에 유용한 키워드를 한국어로 추출해주세요.

질문: "{query}"

요구사항:
1. 질의에 포함된 단어를 그대로 추출해 주세요. 
2. 질문이 영어로 이루어져도 한국어 키워드를 추출해 주세요. (예: "input-output" → "입력", "출력")
3. 전문용어와 기술용어를 우선적으로 선택하세요.
4. 축약어의 경우에는 축약어와 전체단어 모두 사용 키워드로 추출하세요.
5. 2~5개 단어 추출하세요.
6. 중요도가 높은 순서대로 나열하세요.
7. 키워드에 "()", "-", "/" 등이 포함되지 않도록 하세요.
8. 영어 부연설명 없이 오직 한국어로 키워드를 추출하세요.


출력 형식:키워드1, 키워드2, 키워드3, 키워드4


키워드:
"""

ENGLISH_PROMPT_TEMPLATE = """
You are an expert in extracting keywords for academic paper searches. From the given query, please extract the most important and useful keywords for the search in English.

Query: "{query}"

Requirements:
1. Extract keywords exactly as they appear in the query when possible.
2. Extract English keywords even if the question is in English. (e.g., "입력-출력" → "input", "output")
3. Prioritize technical terms and scientific jargon
4. In the case of abbreviations, use both the abbreviation and the full term as keywords
5. After extracting the keywords, list 2~5 in order of importance
6. List them in descending order of importance
7. Do not use "()", "-", or "/" in the keywords. If the keyword contains them, split into separate keywords.
8. Extract keywords only in English, without Korean explanations.

Output Format: keyword1, keyword2, keyword3, keyword4

Keywords:
"""


class LLMKeywordExtractor(BaseKeywordExtractor):
    """Abstract LLM-driven extractor. Subclasses only implement ``_call_llm``."""

    KOREAN_PROMPT_TEMPLATE = KOREAN_PROMPT_TEMPLATE
    ENGLISH_PROMPT_TEMPLATE = ENGLISH_PROMPT_TEMPLATE

    def __init__(self, language: str = "all"):
        if language not in {"korean", "english", "all"}:
            raise ValueError(
                f"language must be one of 'korean'/'english'/'all', got {language!r}"
            )
        self.language = language

    def _call_llm(self, prompt: str) -> str:
        raise NotImplementedError("Subclass must implement _call_llm(prompt) -> str")

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        korean_keywords: List[str] = []
        english_keywords: List[str] = []
        if self.language in {"korean", "all"}:
            try:
                raw = self._call_llm(self.KOREAN_PROMPT_TEMPLATE.format(query=query))
                korean_keywords = self._parse_keyword_list(raw, prefix="키워드:")
            except Exception as e:  # noqa: BLE001
                logging.error(f"한국어 키워드 추출 실패: {e}")
        if self.language in {"english", "all"}:
            try:
                raw = self._call_llm(self.ENGLISH_PROMPT_TEMPLATE.format(query=query))
                english_keywords = self._parse_keyword_list(raw, prefix="Keywords:")
            except Exception as e:  # noqa: BLE001
                logging.error(f"영어 키워드 추출 실패: {e}")
        return {"korean": korean_keywords, "english": english_keywords}

    def _parse_keyword_list(self, text: str, prefix: str) -> List[str]:
        """Parse comma-list LLM 응답을 키워드 배열로 변환.

        절차:
          1. prefix("키워드:" / "Keywords:") 제거 → 콤마로 1차 split.
          2. 각 토큰을 공백 OR 하이픈(`-`) 으로 추가 분해 (예: "input-output" → ["input","output"],
             "vector mapping" → ["vector","mapping"]).
          3. 길이 ≤ 1 토큰 제거, 대소문자 무시 dedup, 최대 10개.
        """
        text = (text or "").strip().replace(prefix, "").strip()
        keywords = [kw.strip() for kw in text.split(",")]
        keywords = [kw for kw in keywords if kw and len(kw) > 1]

        final: List[str] = []
        seen = set()
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

    def generate_search_terms(self, keywords: Dict[str, List[str]]) -> List[str]:
        from .search_terms import build_search_terms
        try:
            return build_search_terms(keywords, number_of_operators=0)
        except Exception as e:  # noqa: BLE001
            logging.error(f"검색어 생성 실패: {e}")
            return []

    def get_extractor_info(self) -> Dict[str, str]:
        return {
            "name": self.__class__.__name__,
            "language": self.language,
        }
