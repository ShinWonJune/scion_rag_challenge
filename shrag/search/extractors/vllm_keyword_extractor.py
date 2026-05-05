"""vLLM (OpenAI 호환) 기반 키워드 추출기.

`LLMKeywordExtractor`를 상속해 한/영 프롬프트와 파싱·검색어 생성 로직을 그대로 사용.
이 클래스는 vLLM Chat Completions 호출(`_call_llm`)만 담당.
"""
from __future__ import annotations

from typing import Dict

from openai import OpenAI

from .llm_keyword_extractor import LLMKeywordExtractor


class VLLMKeywordExtractor(LLMKeywordExtractor):
    """vLLM OpenAI 호환 API 기반 키워드 추출기. 한국어 + 영어 키워드를 동시 추출."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "openai/gpt-oss-120B",
        language: str = "all",
        temperature: float = 0.0,
    ):
        super().__init__(language=language)
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key="dummy-key")

    def _call_llm(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def get_extractor_info(self) -> Dict[str, str]:
        return {
            "name": "vLLM KeywordExtractor",
            "model": self.model,
            "base_url": self.base_url,
            "language": self.language,
        }
