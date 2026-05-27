"""Chat model factory.

Returns a LangChain ``BaseChatModel`` for the requested backend. Default is a
local vLLM server (OpenAI-compatible) running gpt-oss-20b, matching the
original pipeline's ``.env`` defaults.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Backend, PipelineConfig, Settings


def build_chat_model(
    backend: Backend,
    settings: Settings,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Create a chat model for ``backend`` (vllm | openai | gemini)."""
    if backend == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key="EMPTY",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash",
            google_api_key=settings.gemini_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM backend: {backend!r}")


def build_generation_model(cfg: PipelineConfig, settings: Settings) -> BaseChatModel:
    return build_chat_model(
        cfg.llm_backend,
        settings,
        model=cfg.llm_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_answer_tokens,
    )


def build_extractor_model(cfg: PipelineConfig, settings: Settings) -> BaseChatModel:
    # Keyword output is tiny, but reasoning models (e.g. gpt-oss) spend tokens on
    # reasoning first — too small a budget yields empty content. Give headroom.
    return build_chat_model(
        cfg.extractor_backend,
        settings,
        model=cfg.extractor_model,
        temperature=0.0,
        max_tokens=2048,
    )
