from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence


def _normalized_value(doc: Dict[str, Any], key: str) -> str | None:
    raw = doc.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    return value.lower()


def remove_duplicates(
    docs: Iterable[Dict[str, Any]],
    key: str = "doc_id",
    fallback_keys: Sequence[str] = ("title",),
) -> list[Dict[str, Any]]:
    seen: set[str] = set()
    unique_docs: list[Dict[str, Any]] = []
    for doc in docs:
        norm = _normalized_value(doc, key)
        if norm is None:
            for fallback_key in fallback_keys:
                norm = _normalized_value(doc, fallback_key)
                if norm is not None:
                    break
        if norm is None:
            unique_docs.append(doc)
            continue
        if norm in seen:
            continue
        seen.add(norm)
        unique_docs.append(doc)
    return unique_docs
