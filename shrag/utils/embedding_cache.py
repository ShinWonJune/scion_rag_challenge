"""Persistent embedding cache.

Cache key: (encoder_id, embedding_mode, doc_id, text_sha1)
- encoder_id: model_name + truncate_dim (e.g., "Alibaba-NLP/gte-multilingual-base@768")
- embedding_mode: canonical form (e.g., "3T+A"; alias "3*title+abstract" normalized)
- doc_id: ScienceON CN or equivalent
- text_sha1: SHA1 of the actual embedding_text — invalidates cache when doc content (title/abstract) changes

Storage: SQLite single file (atomic writes, concurrency-safe via WAL).

Risks addressed (per design discussion 2026-05-06):
  R1. embedding_text definition change → text_sha1 captures actual text used.
  R2. doc content change at ScienceON → text_sha1 differs → cache miss.
  R3. encoder change → encoder_id differs (model_name + truncate_dim).
  R4. embedding_mode aliases (e.g., "3*title+abstract" vs "3T+A") → normalized in encoder_id key.
  R5. Concurrent writes → SQLite WAL mode + INSERT OR REPLACE.

Behavior:
  - get_batch returns (cached_vectors_dict, missing_indices) for a list of (doc_id, text) pairs.
  - put_batch atomically inserts new vectors.
  - Vector format: float32 numpy bytes.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


LOGGER = logging.getLogger(__name__)


_EMBEDDING_MODE_CANONICAL = {
    "3*title+abstract": "3T+A",
    "title+abstract": "T+A",
    "title": "T",
    "abstract": "A",
}


def normalize_embedding_mode(mode: str) -> str:
    return _EMBEDDING_MODE_CANONICAL.get(mode, mode)


def make_encoder_id(model_name: str, truncate_dim: int | None) -> str:
    return f"{model_name}@{truncate_dim if truncate_dim else 'native'}"


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """SQLite-backed embedding cache.

    Schema:
        CREATE TABLE embeddings (
          encoder_id   TEXT NOT NULL,
          mode         TEXT NOT NULL,
          doc_id       TEXT NOT NULL,
          text_sha1    TEXT NOT NULL,
          dim          INTEGER NOT NULL,
          vector       BLOB NOT NULL,
          created_at   TEXT NOT NULL,
          PRIMARY KEY (encoder_id, mode, doc_id, text_sha1)
        );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                  encoder_id  TEXT NOT NULL,
                  mode        TEXT NOT NULL,
                  doc_id      TEXT NOT NULL,
                  text_sha1   TEXT NOT NULL,
                  dim         INTEGER NOT NULL,
                  vector      BLOB NOT NULL,
                  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                  PRIMARY KEY (encoder_id, mode, doc_id, text_sha1)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_id ON embeddings(doc_id)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_batch(
        self,
        encoder_id: str,
        mode: str,
        doc_ids: list[str],
        texts: list[str],
    ) -> tuple[dict[int, np.ndarray], list[int]]:
        """Look up vectors for a batch.

        Returns:
            (cached_by_idx, missing_indices)
            cached_by_idx maps input index → vector (np.ndarray, dtype float32)
            missing_indices is a sorted list of indices not found in cache.

        Cache hit requires text_sha1 to match (prevents stale hits when doc content changes).
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts length mismatch")
        norm_mode = normalize_embedding_mode(mode)

        cached: dict[int, np.ndarray] = {}
        missing: list[int] = []

        with self._connect() as conn:
            for idx, (doc_id, text) in enumerate(zip(doc_ids, texts)):
                if not doc_id:
                    missing.append(idx)
                    continue
                sha1 = sha1_text(text)
                row = conn.execute(
                    "SELECT vector, dim FROM embeddings "
                    "WHERE encoder_id=? AND mode=? AND doc_id=? AND text_sha1=?",
                    (encoder_id, norm_mode, doc_id, sha1),
                ).fetchone()
                if row is not None:
                    blob, dim = row
                    vec = np.frombuffer(blob, dtype=np.float32).reshape(dim)
                    cached[idx] = vec
                else:
                    missing.append(idx)
        return cached, missing

    def put_batch(
        self,
        encoder_id: str,
        mode: str,
        doc_ids: list[str],
        texts: list[str],
        vectors: np.ndarray,
    ) -> int:
        """Insert/replace vectors. Vectors expected shape (n, dim), float32.

        Returns number of rows inserted.
        """
        if len(doc_ids) != len(texts) or len(doc_ids) != vectors.shape[0]:
            raise ValueError("length mismatch")
        norm_mode = normalize_embedding_mode(mode)
        dim = int(vectors.shape[1])

        rows = []
        for doc_id, text, vec in zip(doc_ids, texts, vectors):
            if not doc_id:
                continue
            sha1 = sha1_text(text)
            blob = np.asarray(vec, dtype=np.float32).tobytes()
            rows.append((encoder_id, norm_mode, doc_id, sha1, dim, blob))

        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings "
                "(encoder_id, mode, doc_id, text_sha1, dim, vector) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def stats(self, encoder_id: str | None = None) -> dict:
        with self._connect() as conn:
            if encoder_id:
                total = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE encoder_id=?",
                    (encoder_id,),
                ).fetchone()[0]
                modes = conn.execute(
                    "SELECT mode, COUNT(*) FROM embeddings WHERE encoder_id=? GROUP BY mode",
                    (encoder_id,),
                ).fetchall()
            else:
                total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                modes = conn.execute(
                    "SELECT encoder_id, mode, COUNT(*) FROM embeddings GROUP BY encoder_id, mode"
                ).fetchall()
        return {"total": total, "by_mode": [list(m) for m in modes]}
