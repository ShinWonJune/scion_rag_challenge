"""Corpus construction helpers for corpus-first RAG runs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

_WS_RE = re.compile(r"\s+")


@dataclass
class CorpusArtifact:
    documents: list[Document]
    document_count: int
    duplicate_count: int
    source_counts: dict[str, int] = field(default_factory=dict)

    def to_manifest(self) -> dict:
        return {
            "document_count": self.document_count,
            "duplicate_count": self.duplicate_count,
            "source_counts": self.source_counts,
        }


def _doc_key(doc: Document) -> tuple[str, str]:
    source = str(doc.metadata.get("source") or "").strip().lower()
    doc_id = str(doc.metadata.get("doc_id") or "").strip().lower()
    if doc_id:
        return source, f"id:{doc_id}"
    title = _WS_RE.sub(" ", str(doc.metadata.get("title") or "").strip().lower())
    return source, f"title:{title or doc.page_content[:120].lower()}"


def build_corpus(document_groups: Iterable[Iterable[Document]]) -> CorpusArtifact:
    seen: set[tuple[str, str]] = set()
    documents: list[Document] = []
    duplicates = 0
    source_counts: dict[str, int] = {}

    for group in document_groups:
        for doc in group:
            key = _doc_key(doc)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            documents.append(doc)
            source = str(doc.metadata.get("source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

    return CorpusArtifact(
        documents=documents,
        document_count=len(documents),
        duplicate_count=duplicates,
        source_counts=source_counts,
    )


def save_corpus_jsonl(artifact: CorpusArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in artifact.documents:
            f.write(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_corpus_jsonl(path: str | Path) -> CorpusArtifact:
    documents: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            documents.append(
                Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {}),
                )
            )
    return build_corpus([documents])
