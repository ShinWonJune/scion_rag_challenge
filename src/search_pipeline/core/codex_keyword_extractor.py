"""Codex CLI (ChatGPT 구독 인증) 기반 키워드 추출기.

`LLMKeywordExtractor`를 상속해 한/영 프롬프트와 파싱·검색어 생성 로직을 그대로 사용.
이 클래스는 Codex CLI 호출(`_call_llm`)만 담당. OpenAI API quota 무관.
"""
from __future__ import annotations

from typing import Dict

from src.codex_client import CodexClient

from .llm_keyword_extractor import LLMKeywordExtractor


class CodexKeywordExtractor(LLMKeywordExtractor):
    """Codex CLI 기반 키워드 추출기. 한국어 + 영어 키워드를 동시 추출."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        timeout_sec: int = 600,
        language: str = "all",
        max_tokens: int = 1500,
    ):
        super().__init__(language=language)
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens
        self.client = CodexClient(model=model, timeout_sec=timeout_sec)

    def _call_llm(self, prompt: str) -> str:
        return self.client.generate_answer_with_prompt(prompt, max_tokens=self.max_tokens)

    def get_extractor_info(self) -> Dict[str, str]:
        return {
            "name": "Codex KeywordExtractor",
            "model": self.model,
            "language": self.language,
        }
