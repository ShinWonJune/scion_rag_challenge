"""Search-term builder, ported verbatim from
``shrag/search/extractors/search_terms.py``.

Converts ``{korean: [...], english: [...]}`` keyword lists into platform query
strings via progressive OR-rotation, with optional trailing AND operators.
Default ``number_of_operators=0`` => OR-only (BEST_CONFIG conclusion).
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
    number_of_operators: int = 0,
) -> List[str]:
    """Build platform-ready search terms from per-language keyword lists."""
    terms: List[str] = []
    terms.extend(_rotation_or_terms(list(keywords.get("korean") or [])))
    terms.extend(_rotation_or_terms(list(keywords.get("english") or [])))
    terms = _add_operator(terms, number_of_operators=number_of_operators)
    seen = set()
    final: List[str] = []
    for t in terms:
        t = (t or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        final.append(t)
    return final


def build_search_terms_by_lang(
    keywords: Dict[str, List[str]], number_of_operators: int = 0
) -> Dict[str, List[str]]:
    """Per-language term lists (each language isolated), as in step1_search."""
    ko_kw = list(keywords.get("korean") or [])
    en_kw = list(keywords.get("english") or [])
    return {
        "korean": build_search_terms({"korean": ko_kw, "english": []}, number_of_operators)
        if ko_kw
        else [],
        "english": build_search_terms({"korean": [], "english": en_kw}, number_of_operators)
        if en_kw
        else [],
    }
