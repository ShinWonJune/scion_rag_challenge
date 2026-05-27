"""Parity: shrag_lc.search.terms.build_search_terms == original shrag builder."""

import pytest

from shrag_lc.search.terms import build_search_terms, build_search_terms_by_lang

orig = pytest.importorskip("shrag.search.extractors.search_terms")

CASES = [
    {"korean": ["기계학습", "추천시스템", "알고리즘"], "english": ["machine", "learning"]},
    {"korean": [], "english": ["vector", "mapping", "index"]},
    {"korean": ["딥러닝"], "english": []},
    {"korean": [], "english": []},
]


@pytest.mark.parametrize("kw", CASES)
@pytest.mark.parametrize("ops", [0, 1, 2, 3])
def test_build_search_terms_parity(kw, ops):
    assert build_search_terms(kw, number_of_operators=ops) == orig.build_search_terms(
        kw, number_of_operators=ops
    )


def test_rotation_or_only():
    # OR-only (ops=0): progressive truncation, KO then EN.
    out = build_search_terms({"korean": ["a", "b", "c"], "english": ["x", "y"]}, 0)
    assert out == ["a|b|c", "a|b", "a", "x|y", "x"]


def test_by_lang_isolation():
    by_lang = build_search_terms_by_lang(
        {"korean": ["가", "나"], "english": ["foo", "bar"]}, number_of_operators=0
    )
    assert by_lang["korean"] == ["가|나", "가"]
    assert by_lang["english"] == ["foo|bar", "foo"]
