"""Gemini API 기반 키워드 추출기.

`LLMKeywordExtractor`를 상속해 한/영 프롬프트와 파싱·검색어 생성 로직을 그대로 사용.
이 클래스는 Gemini 백엔드 호출(`_call_llm`)만 담당.
"""
from __future__ import annotations

import logging
from typing import Dict

import google.generativeai as genai

from .llm_keyword_extractor import LLMKeywordExtractor


class KeywordExtractor(LLMKeywordExtractor):
    """Gemini API 기반 키워드 추출기. 한국어 + 영어 키워드를 동시 추출."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        language: str = "all",
    ):
        if not api_key:
            raise ValueError("API 키가 필요합니다. api_key 매개변수가 None이거나 비어있습니다.")
        super().__init__(language=language)
        self.api_key = api_key
        self.model_name = model_name
        self.model = self._init_gemini()

    def _init_gemini(self) -> genai.GenerativeModel:
        genai.configure(api_key=self.api_key)
        generation_config = genai.GenerationConfig(
            temperature=0.2,
            candidate_count=1,
        )
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
        )

    def _call_llm(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return (response.text or "").strip()

    def get_extractor_info(self) -> Dict[str, str]:
        return {
            "name": "Gemini KeywordExtractor",
            "model": self.model_name,
            "language": self.language,
        }
