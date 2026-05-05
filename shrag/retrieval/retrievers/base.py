# -*- coding: utf-8 -*-
"""Abstract base class for all retrievers."""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np


class Retriever(abc.ABC):
    """Abstract Base Class for a retriever index."""

    def __init__(self, embeddings: np.ndarray):
        """
        Initializes the retriever with document embeddings.
        Args:
            embeddings (np.ndarray): A 2D numpy array of shape (N, D) containing
                                     L2-normalized document embeddings.
        """
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D numpy array.")
        self.embeddings = embeddings
        self.num_docs, self.dim = embeddings.shape

    @abc.abstractmethod
    def search(
        self, query_vecs: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the index for the top_k most similar documents for each query vector.

        Args:
            query_vecs (np.ndarray): A 2D numpy array of shape (Q, D) containing
                                     L2-normalized query embeddings.
            top_k (int): The number of top documents to retrieve for each query.

        Returns:
            A tuple containing:
            - scores (np.ndarray): A (Q, top_k) array of similarity scores.
            - indices (np.ndarray): A (Q, top_k) array of document indices.
        """
        raise NotImplementedError
