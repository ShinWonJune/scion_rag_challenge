from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMClient(ABC):
    def __init__(self, backend: str, config: Dict[str, Any]):
        self.backend = backend
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class _DryRunLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        return ""


class GeminiLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Gemini generation is handled by step scripts.")


class ChatGPTLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("ChatGPT generation is handled by step scripts.")


class VLLMLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("vLLM generation is handled by step scripts.")


def create_llm_client(backend: str, config: Dict[str, Any]) -> BaseLLMClient:
    backend_norm = backend.strip().lower()
    if config.get("dry_run"):
        return _DryRunLLMClient(backend_norm, config)
    if backend_norm == "gemini":
        return GeminiLLMClient(backend_norm, config)
    if backend_norm == "chatgpt":
        return ChatGPTLLMClient(backend_norm, config)
    if backend_norm == "vllm":
        return VLLMLLMClient(backend_norm, config)
    raise ValueError(f"Unsupported llm backend: {backend}")

