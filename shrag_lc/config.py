"""Configuration for shrag_lc.

Two layers:
  * ``Settings`` — environment-derived secrets/endpoints (.env), via
    pydantic-settings. Mirrors the original ``.env.example`` keys.
  * ``PipelineConfig`` — pipeline hyperparameters, defaulted to the values
    documented in ``experiments/BEST_CONFIG.md`` (gte + dragonkue rerank).

An encoder JSON (``configs/query_encoder/*.json``) can be loaded to override
the embedding model / dim / mode, so the LangChain build reuses the same
config files as the original pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Insufficient-context escape sentences (must match the grounding prompt).
INSUFFICIENT_KO = "제공된 문서에서는 이 질문에 답할 정보를 찾을 수 없습니다."
INSUFFICIENT_EN = "The provided documents do not contain sufficient information to answer this question."


class Settings(BaseSettings):
    """Secrets and endpoints sourced from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM backends
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "openai/gpt-oss-20b"
    openai_api_key: str = ""
    google_api_key: str = ""
    gemini_api_key: str = ""

    # Search credentials
    pubmed_api_key: str = ""
    pubmed_email: str = ""
    scienceon_credentials_path: str = "configs/credentials/scienceon_api_credentials.json"
    pubmed_credentials_path: str = "configs/credentials/pubmed_api_credentials.json"

    @property
    def gemini_key(self) -> str:
        return self.gemini_api_key or self.google_api_key


Backend = Literal["vllm", "openai", "gemini"]
Source = Literal["scienceon", "pubmed", "wikipedia"]


@dataclass
class PipelineConfig:
    """Pipeline hyperparameters. Defaults from BEST_CONFIG.md."""

    # --- search / acquisition ---
    source: Source = "wikipedia"
    target_documents: int = 50          # m: dedup cap per query
    keyword_lang: Literal["all", "korean", "english"] = "all"
    number_of_operators: int = 0        # 0 = OR-only (BEST_CONFIG conclusion)
    scienceon_max_pages: int = 5
    scienceon_max_concurrency: int = 2
    scienceon_min_interval_sec: float = 0.5
    scienceon_fixed_concurrency: bool = False
    scienceon_max_retries: int = 5
    scienceon_retry_base_sleep_sec: float = 2.0
    scienceon_retry_max_sleep_sec: float = 60.0
    cache_root: str = "outputs/_shared_cache"
    disable_cache: bool = False

    # --- embedding ---
    embedding_model: str = "Alibaba-NLP/gte-multilingual-base"
    embedding_dim: int = 768
    embedding_mode: str = "3*title+abstract"
    embedding_batch_size: int = 32
    embedding_max_seq_length: int = 512
    use_fp16: bool = True
    device: str = "auto"                # auto -> cuda if available else cpu

    # --- retrieval / rerank ---
    dense_top_k: int = 5
    use_reranker: bool = True
    reranker_model: str = "dragonkue/bge-reranker-v2-m3-ko"
    rerank_top_n: int = 3

    # --- generation ---
    llm_backend: Backend = "vllm"
    llm_model: str | None = None        # None -> backend default
    max_answer_tokens: int = 4000
    temperature: float = 0.0

    # --- keyword extraction backend (defaults to same as llm) ---
    extractor_backend: Backend = "vllm"
    extractor_model: str | None = None

    @classmethod
    def from_encoder_json(cls, path: str | Path, **overrides) -> "PipelineConfig":
        """Build config from a configs/query_encoder/*.json file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(
            embedding_model=data.get("model_name", cls.embedding_model),
            embedding_dim=data.get("embedding_dim", cls.embedding_dim),
            embedding_mode=data.get("embedding_mode", cls.embedding_mode),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg
