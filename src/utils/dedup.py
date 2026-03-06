from __future__ import annotations

from typing import Any, Dict, List


def remove_duplicates(docs: List[Dict[str, Any]], key: str = "title") -> List[Dict[str, Any]]:
    seen = set()
    unique_docs: List[Dict[str, Any]] = []
    for doc in docs:
        raw = doc.get(key, "")
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        norm = value.lower()
        if norm in seen:
            continue
        seen.add(norm)
        unique_docs.append(doc)
    return unique_docs

