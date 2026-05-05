# -*- coding: utf-8 -*-
"""Factory for creating retriever instances."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from shrag.retrieval.retrievers.base import Retriever

_HAS_FAISS = False
try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    pass


def get_retriever(embeddings: np.ndarray, force_numpy: bool = False) -> Retriever:
    """
    Factory function to get the best available retriever.
    Prefers FaissRetriever if available, otherwise falls back to NumpyRetriever.
    """
    if _HAS_FAISS and not force_numpy:
        from shrag.retrieval.retrievers.faiss_retriever import FaissRetriever

        print("[INFO] Using FAISS for retrieval.", file=sys.stderr)
        return FaissRetriever(embeddings)

    from shrag.retrieval.retrievers.numpy_retriever import NumpyRetriever

    print(
        "[INFO] FAISS not found or disabled. Using NumPy for retrieval (slower).",
        file=sys.stderr,
    )
    return NumpyRetriever(embeddings)
