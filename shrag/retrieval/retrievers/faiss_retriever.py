# -*- coding: utf-8 -*-
"""Retriever implementations using FAISS."""
from __future__ import annotations

import os
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
        
        k = min(top_k, self.num_docs)
        scores, indices = self.index.search(query_vecs.astype(np.float32), k)
        return scores, indices


class GpuFaissRetriever(Retriever):
    """A FAISS GPU retriever using a flat inner-product index."""

    def __init__(
        self,
        embeddings: np.ndarray,
        device_id: int = 0,
        temp_memory_gb: float | None = None,
    ):
        if not _HAS_FAISS:
            raise ImportError("FAISS is not installed.")
        if not hasattr(faiss, "StandardGpuResources"):
            raise ImportError("Installed FAISS does not include GPU support.")
        if faiss.get_num_gpus() <= device_id:
            raise RuntimeError(
                f"FAISS sees {faiss.get_num_gpus()} GPU(s), cannot use device {device_id}."
            )
        super().__init__(embeddings)
        self.device_id = device_id
        self.resources = faiss.StandardGpuResources()

        if temp_memory_gb is None:
            raw = os.environ.get("SHRAG_FAISS_GPU_TEMP_MEMORY_GB", "8")
            temp_memory_gb = float(raw)
        if temp_memory_gb > 0:
            self.resources.setTempMemory(int(temp_memory_gb * 1024**3))

        config = faiss.GpuIndexFlatConfig()
        config.device = device_id
        self.index = faiss.GpuIndexFlatIP(self.resources, self.dim, config)
        self.index.add(self.embeddings.astype(np.float32))

    def search(
        self, query_vecs: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a search using the FAISS GPU index."""
        if query_vecs.ndim != 2 or query_vecs.shape[1] != self.dim:
            raise ValueError(f"Query vectors must have shape (Q, {self.dim})")

        k = min(top_k, self.num_docs)
        scores, indices = self.index.search(query_vecs.astype(np.float32), k)
        return scores, indices
