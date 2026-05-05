from __future__ import annotations

import os
from typing import Any, Dict

from .base_extractor import BaseKeywordExtractor


class _DryRunExtractor(BaseKeywordExtractor):
    def __init__(self, backend: str):
        self.backend = backend

    def extract_keywords(self, query: str) -> Dict[str, list[str]]:
        return {"korean": [], "english": []}

    def get_extractor_info(self) -> Dict[str, str]:
        return {"name": f"dry-run-{self.backend}"}

    def generate_search_terms(self, keywords: Dict[str, list[str]]) -> list[str]:
        return []


def create_keyword_extractor(backend: str, config: Dict[str, Any]) -> BaseKeywordExtractor:
    backend_norm = backend.strip().lower()
    if config.get("dry_run"):
        return _DryRunExtractor(backend_norm)

    if backend_norm == "gemini":
        from .keyword_extractor import KeywordExtractor

        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        language = config.get("language", "all")
        model_name = config.get("model_name", "gemini-2.5-flash")
        return KeywordExtractor(api_key=api_key, model_name=model_name, language=language)

    if backend_norm == "chatgpt":
        from .chatgpt_keyword_extractor import ChatGPTKeywordExtractor

        api_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        language = config.get("language", "all")
        model = config.get("model", "gpt-4o-mini")
        temperature = config.get("temperature", 0.0)
        credentials_file = config.get("credentials_file", "./configs/chatgpt_api_credentials.json")
        if api_key:
            return ChatGPTKeywordExtractor(
                api_key=api_key,
                model=model,
                language=language,
                temperature=temperature,
            )
        return ChatGPTKeywordExtractor(
            credentials_file=credentials_file,
            model=model,
            language=language,
            temperature=temperature,
        )

    if backend_norm == "vllm":
        from .vllm_keyword_extractor import VLLMKeywordExtractor

        return VLLMKeywordExtractor(
            base_url=config.get("vllm_base_url", "http://localhost:8000/v1"),
            model=config.get("vllm_model", "openai/gpt-oss-120B"),
            language=config.get("language", "all"),
        )

    if backend_norm == "codex":
        from .codex_keyword_extractor import CodexKeywordExtractor

        return CodexKeywordExtractor(
            model=config.get("model", "gpt-5.4"),
            timeout_sec=int(config.get("timeout_sec", 600)),
            language=config.get("language", "all"),
            max_tokens=int(config.get("max_tokens", 1500)),
        )

    raise ValueError(f"Unsupported extractor backend: {backend}")
