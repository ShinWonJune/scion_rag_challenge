# -*- coding: utf-8 -*-
"""Retriever implementation using pure NumPy."""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Use absolute import path
from shrag.retrieval.retrievers.base import Retriever


class NumpyRetriever(Retriever):
    """
    A retriever using pure NumPy for matrix multiplication.
    This serves as a fallback when FAISS is not available.
    """

    def __init__(self, embeddings: np.ndarray):
        super().__init__(embeddings)

    def search(
        self, query_vecs: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a search using NumPy's dot product.
        For L2-normalized vectors, this is equivalent to cosine similarity.
        """
        if query_vecs.shape[1] != self.dim:
            raise ValueError(
                f"Query vector dimension {query_vecs.shape[1]} does not match index dimension {self.dim}"
            )

        # Compute cosine similarities with matrix multiplication
        # Shape: (num_queries, D) @ (D, num_docs) -> (num_queries, num_docs)
        sim_matrix = query_vecs @ self.embeddings.T

        # Get the indices of the top_k similarities for each query
        # Using argpartition for efficiency is better than a full sort
        k = min(top_k, self.num_docs)
        top_k_indices = np.argpartition(-sim_matrix, kth=k - 1, axis=1)[:, :k]

        # Get the scores for the top_k indices
        top_k_scores = np.take_along_axis(sim_matrix, top_k_indices, axis=1)

        # Sort within the top_k results to get the correct ranking
        sorted_order = np.argsort(-top_k_scores, axis=1)
        final_indices = np.take_along_axis(top_k_indices, sorted_order, axis=1)
        final_scores = np.take_along_axis(top_k_scores, sorted_order, axis=1)

        return final_scores, final_indices
