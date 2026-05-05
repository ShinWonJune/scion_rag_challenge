# -*- coding: utf-8 -*-
"""Retriever implementation using FAISS."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from .base import Retriever

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FaissRetriever(Retriever):
    """A fast retriever using FAISS for dense search."""
    def __init__(self, embeddings: np.ndarray):
        if not _HAS_FAISS:
            raise ImportError("FAISS is not installed. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'.")
        super().__init__(embeddings)
        # For L2-normalized vectors, inner product is equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings.astype(np.float32))

    def search(
        self, query_vecs: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a search using the FAISS index."""
        if query_vecs.ndim != 2 or query_vecs.shape[1] != self.dim:
            raise ValueError(f"Query vectors must have shape (Q, {self.dim})")
        
        k = min(top_k, self.n_docs)
        scores, indices = self.index.search(query_vecs.astype(np.float32), k)
        return scores, indices


