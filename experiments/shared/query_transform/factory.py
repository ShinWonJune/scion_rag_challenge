from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.shared.query_transform.hyde import HyDETransform


@dataclass
class QueryTransformConfig:
    name: str
    hyde_backend: str = "vllm"
    hyde_model: str = "openai/gpt-oss-20b"
    hyde_base_url: str = "http://localhost:8000/v1"
    hyde_api_key: str | None = None
    hyde_seed: int = 42


def create_transform(name: str, **kwargs: Any) -> Any:
    if name == "raw":
        return None
    if name == "keywords":
        return None
    if name.startswith("hyde_mean_"):
        n_samples = int(name.rsplit("_", 1)[-1])
        return HyDETransform(n_samples=n_samples, **kwargs)
    if name.startswith("hyde_union_"):
        n_samples = int(name.rsplit("_", 1)[-1])
        return HyDETransform(n_samples=n_samples, **kwargs)
    raise ValueError(f"Unsupported query transform: {name}")
