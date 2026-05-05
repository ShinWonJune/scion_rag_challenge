"""ChatGPT API 기반 키워드 추출기.

`LLMKeywordExtractor`를 상속해 한/영 프롬프트와 파싱·검색어 생성 로직을 그대로 사용.
이 클래스는 OpenAI Chat Completions 호출(`_call_llm`)만 담당.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from openai import OpenAI

from .llm_keyword_extractor import LLMKeywordExtractor


class ChatGPTKeywordExtractor(LLMKeywordExtractor):
    """ChatGPT API 기반 키워드 추출기. 한국어 + 영어 키워드를 동시 추출."""

    def __init__(
        self,
        credentials_file: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        language: str = "all",
        temperature: Optional[float] = 0.0,
    ):
        super().__init__(language=language)

        if api_key:
            openai_api_key = api_key
        elif credentials_file:
            with open(credentials_file, "r", encoding="utf-8") as f:
                credentials = json.load(f)
            openai_api_key = credentials["api_key"]
        else:
            raise ValueError("api_key 또는 credentials_file 중 하나는 필수입니다.")

        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature

    def _call_llm(self, prompt: str) -> str:
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**request)
        return (resp.choices[0].message.content or "").strip()

    def get_extractor_info(self) -> Dict[str, str]:
        return {
            "name": "ChatGPT KeywordExtractor",
            "model": self.model,
            "language": self.language,
        }
