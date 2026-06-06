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
        top_k_indices = np.argpartition(-sim_matrix, kth=k - 1, axis=1)[:, :k] # sim_matrix index 기준으로 top_k_indices 가져옴
        # numpy는 기본적으로 오름차순, 따라서 음수를 취해서 오름차순 정렬을 통해 내림차순 효과를 냄. argpartition은 완전한 정렬이 아니라 top_k 위치를 보장하는 효율적인 방법입니다.
        # Get the scores for the top_k indices
        top_k_scores = np.take_along_axis(sim_matrix, top_k_indices, axis=1) # top_k_indices를 index 삼아 sim_matrix의 값 가져옴
        #top_k_* 변수는 top_k개의 유사도 점수와 해당 문서의 인덱스를 담고 있음. 하지만 이들은 아직 정렬되지 않은 상태임.
        # Sort within the top_k results to get the correct ranking)
        sorted_order = np.argsort(-top_k_scores, axis=1) # top_k_score의 value를 기준으로 내림 차순 정렬을 위한 index 순서 반환
        final_indices = np.take_along_axis(top_k_indices, sorted_order, axis=1) # Reorder indices according to sorted scores
        final_scores = np.take_along_axis(top_k_scores, sorted_order, axis=1) # Get the final scores in the correct order
        # sorted_order는 top_k_scores의 값에 따라 내림차순으로 정렬된 인덱스 배열. 이를 top_k_indices에 적용하여 최종적으로 올바른 순서로 정렬된 인덱스와 점수를 얻습니다.
        return final_scores, final_indices
