"""Shared search-term builder.

Single source of truth for converting `{korean: [...], english: [...]}` keyword
dictionaries into the platform query strings used by ScienceON / Wikipedia /
PubMed retrieval. Mirrors the rotation + AND-operator logic originally written
for Wikipedia retrieval in `keyword_extractor.py`:

  1. For each language, emit one OR-joined string per progressive truncation
     (drop one keyword from the tail and repeat).  N keywords ⇒ N OR-strings.
  2. After all OR-strings are collected, rewrite the **last `number_of_operators`
     pipe-separated tokens** of each string into AND (`+`) joins. Tokens above
     that limit stay OR-joined. If a string has fewer tokens than the operator
     count, every `|` becomes `+`.
  3. Deduplicate while preserving insertion order; drop empties.

All extractors (Gemini / ChatGPT / vLLM / Codex) now route through this
function so the platform receives the same query expansion regardless of which
LLM produced the keywords.
"""
from __future__ import annotations

from typing import Dict, Iterable, List


def _add_operator(search_terms: Iterable[str], number_of_operators: int) -> List[str]:
    if number_of_operators <= 0:
        return list(search_terms)
    out: List[str] = []
    for term in search_terms:
        if "|" not in term:
            out.append(term)
            continue
        parts = term.split("|")
        if len(parts) <= number_of_operators:
            out.append("+".join(parts))
            continue
        front_parts = parts[:-number_of_operators]
        back_parts = parts[-number_of_operators:]
        front_term = "|".join(front_parts) if front_parts else ""
        back_term = "+".join(back_parts) if back_parts else ""
        if front_term and back_term:
            out.append(front_term + "+" + back_term)
        else:
            out.append(front_term + back_term)
    return out


def _rotation_or_terms(keywords: List[str]) -> List[str]:
    """For [a,b,c] return ["a|b|c", "a|b", "a"] (drop tail, repeat)."""
    out: List[str] = []
    kw = [k for k in keywords if k]
    for _ in range(len(kw)):
        out.append("|".join(kw))
        kw = kw[:-1]
    return out


def build_search_terms(
    keywords: Dict[str, List[str]],
    number_of_operators: int = 3,
) -> List[str]:
    """Build platform-ready search terms from per-language keyword lists."""
    terms: List[str] = []
    terms.extend(_rotation_or_terms(list(keywords.get("korean") or [])))
    terms.extend(_rotation_or_terms(list(keywords.get("english") or [])))
    terms = _add_operator(terms, number_of_operators=number_of_operators)
    # Order-preserving dedup + drop blanks
    seen = set()
    final: List[str] = []
    for t in terms:
        t = (t or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        final.append(t)
    return final
