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
    Uses CPU FAISS by default when available. Set SHRAG_FAISS_DEVICE=gpu to use
    FAISS GPU, with SHRAG_FAISS_GPU_TEMP_MEMORY_GB limiting scratch memory.
    """
    import os

    faiss_device = os.environ.get("SHRAG_FAISS_DEVICE", "cpu").strip().lower()

    if _HAS_FAISS and not force_numpy and faiss_device == "gpu":
        from shrag.retrieval.retrievers.faiss_retriever import GpuFaissRetriever

        device_id = int(os.environ.get("SHRAG_FAISS_GPU_ID", "0"))
        temp_gb = float(os.environ.get("SHRAG_FAISS_GPU_TEMP_MEMORY_GB", "8"))
        print(
            f"[INFO] Using FAISS GPU for retrieval "
            f"(device={device_id}, temp_memory_gb={temp_gb}).",
            file=sys.stderr,
        )
        return GpuFaissRetriever(
            embeddings,
            device_id=device_id,
            temp_memory_gb=temp_gb,
        )

    if _HAS_FAISS and not force_numpy:
        from shrag.retrieval.retrievers.faiss_retriever import FaissRetriever

        print("[INFO] Using FAISS CPU for retrieval.", file=sys.stderr)
        return FaissRetriever(embeddings)

    from shrag.retrieval.retrievers.numpy_retriever import NumpyRetriever

    print(
        "[INFO] FAISS not found or disabled. Using NumPy for retrieval (slower).",
        file=sys.stderr,
    )
    return NumpyRetriever(embeddings)
