from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Protocol

import numpy as np


class Retriever(Protocol):
    def build(self, corpus_texts: list[str]) -> None: ...
    def search(self, queries: list[str], top_k: int): ...


class BM25Retriever:
    def __init__(self, tokenizer: str = "whitespace", k1: float = 1.5, b: float = 0.75) -> None:
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.corpus_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.doc_lens: list[int] = []

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer == "whitespace":
            return [tok for tok in re.split(r"\s+", text.lower().strip()) if tok]
        return [tok for tok in re.split(r"\s+", text.lower().strip()) if tok]

    def build(self, corpus_texts: list[str]) -> None:
        self.corpus_tokens = [self._tokenize(text) for text in corpus_texts]
        self.doc_freqs = [Counter(tokens) for tokens in self.corpus_tokens]
        self.doc_lens = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        n_docs = len(self.corpus_tokens)
        df: defaultdict[str, int] = defaultdict(int)
        for tokens in self.corpus_tokens:
            for token in set(tokens):
                df[token] += 1
        self.idf = {
            token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        doc_tf = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx] if self.doc_lens else 0
        for token in query_tokens:
            tf = doc_tf.get(token, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
            score += idf * ((tf * (self.k1 + 1)) / denom)
        return score

    def search(self, queries: list[str], top_k: int):
        all_scores = []
        all_indices = []
        for query in queries:
            query_tokens = self._tokenize(query)
            scores = np.array([self._score(query_tokens, idx) for idx in range(len(self.corpus_tokens))], dtype=np.float32)
            k = min(top_k, len(scores))
            if k == 0:
                all_scores.append(np.array([], dtype=np.float32))
                all_indices.append(np.array([], dtype=np.int32))
                continue
            top_idx = np.argsort(-scores)[:k]
            all_scores.append(scores[top_idx])
            all_indices.append(top_idx.astype(np.int32))
        return np.stack(all_scores), np.stack(all_indices)
