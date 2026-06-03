"""FAISS vector store helpers.

Uses MAX_INNER_PRODUCT over L2-normalized embeddings, which equals cosine
similarity — the LangChain equivalent of the original FAISS ``IndexFlatIP``.

Note: the original ``shrag`` code calls SentenceTransformers with
``normalize_embeddings=False`` in some places, but normalizes vectors before
FAISS search. This LC build normalizes inside the embedding wrapper, so the
retrieval metric is intended to match: inner product over unit vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import PipelineConfig


def build_faiss(documents: list[Document], embeddings: Embeddings) -> FAISS:
    """Build an in-memory FAISS store (inner product = cosine on unit vectors)."""
    if not documents:
        raise ValueError("Cannot build FAISS store from zero documents.")
    return FAISS.from_documents(
        documents,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def save_faiss(store: FAISS, path: str | Path) -> None:
    store.save_local(str(path))


def load_faiss(path: str | Path, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(
        str(path),
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        allow_dangerous_deserialization=True,
    )


def index_metadata(cfg: PipelineConfig, document_count: int) -> dict:
    return {
        "embedding_model": cfg.embedding_model,
        "embedding_dim": cfg.embedding_dim,
        "embedding_mode": cfg.embedding_mode,
        "embedding_max_seq_length": cfg.embedding_max_seq_length,
        "normalize_embeddings": True,
        "document_count": document_count,
    }


def save_faiss_artifact(store: FAISS, path: str | Path, cfg: PipelineConfig, document_count: int) -> dict:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_faiss(store, path)
    metadata = index_metadata(cfg, document_count)
    (path / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def load_faiss_artifact(path: str | Path, embeddings: Embeddings, cfg: PipelineConfig) -> tuple[FAISS, dict]:
    path = Path(path)
    metadata_path = path / "index_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing FAISS index metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = index_metadata(cfg, int(metadata.get("document_count", 0)))
    for key in ["embedding_model", "embedding_dim", "embedding_mode", "embedding_max_seq_length"]:
        if metadata.get(key) != expected.get(key):
            raise ValueError(
                f"Cannot reuse FAISS index with mismatched {key}: "
                f"{metadata.get(key)!r} != {expected.get(key)!r}"
            )
    return load_faiss(path, embeddings), metadata
