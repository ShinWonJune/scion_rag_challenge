"""FAISS vector store helpers.

Uses MAX_INNER_PRODUCT over L2-normalized embeddings, which equals cosine
similarity — the LangChain equivalent of the original FAISS ``IndexFlatIP``.

Note: the original ``shrag`` pipeline encoded documents with
``normalize_embeddings=False`` (raw-magnitude inner product), whereas this build
normalizes both documents and queries (cosine). gte-multilingual is trained for
cosine, so this is intentional — but rankings will not be bit-identical to the
original, which matters when comparing eval numbers head-to-head.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


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
