# -*- coding: utf-8 -*-
"""QueryEncoder class for embedding text queries."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# retrieval_system 패키지의 모듈들을 절대 경로로 임포트
from src.retrieval_system import utils


class QueryEncoder:
    """Wraps a SentenceTransformer model for encoding queries."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()
        # Default instruction for models like Snowflake Arctic Embed
        self.default_instruction = (
            "Represent this sentence for searching relevant passages: "
        )

    def _load_model(self):
        """Loads the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=None if self.device == "auto" else self.device,
            )
        except ImportError:
            raise ImportError(
                "SentenceTransformers is not installed. Please install it with "
                "'pip install sentence-transformers'."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")

    def encode_queries(
        self, queries: List[str], instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Encodes a list of query strings into a numpy array of embeddings.
        Returns L2-normalized (Q, D) float32 array.
        """
        # Use the provided instruction, the default one, or none if explicitly empty
        final_instruction = (
            instruction if instruction is not None else self.default_instruction
        )

        prefixed_queries = [final_instruction + q for q in queries]

        q_vecs = self.model.encode(
            prefixed_queries,
            batch_size=32,
            normalize_embeddings=False,  # We normalize manually
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        return utils.l2_normalize(q_vecs)
