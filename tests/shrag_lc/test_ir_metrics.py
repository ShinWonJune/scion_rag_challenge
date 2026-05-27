"""Parity: shrag_lc.eval.ir_metrics == original experiments metrics_ir."""

import pytest

from shrag_lc.eval import ir_metrics as lc

orig = pytest.importorskip("experiments.shared.evaluate.metrics_ir")

HITS = ["d1", "d2", "d3", "d4", "d5"]
GOLD = {"d3", "d9"}


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_pointwise_parity(k):
    assert lc.recall_at_k(HITS, GOLD, k) == orig.recall_at_k(HITS, GOLD, k)
    assert lc.precision_at_k(HITS, GOLD, k) == orig.precision_at_k(HITS, GOLD, k)
    assert lc.mrr_at_k(HITS, GOLD, k) == orig.mrr_at_k(HITS, GOLD, k)
    assert lc.ndcg_at_k(HITS, GOLD, k) == orig.ndcg_at_k(HITS, GOLD, k)


def test_summarize_parity():
    ranked = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"]}
    gold = {"q1": {"b"}, "q2": {"x"}}
    lc_out = lc.summarize_ir_metrics(ranked, gold, ks=(1, 3))
    orig_out = orig.summarize_ir_metrics(ranked, gold, ks=(1, 3))
    for key in ["recall_at_1", "recall_at_3", "mrr_at_3", "ndcg_at_3", "precision_at_3"]:
        assert lc_out[key] == orig_out[key]


def test_known_values():
    # gold at rank 3 -> mrr 1/3, recall@3=1, recall@1=0
    assert lc.recall_at_k(HITS, GOLD, 1) == 0.0
    assert lc.recall_at_k(HITS, GOLD, 3) == 1.0
    assert lc.mrr_at_k(HITS, GOLD, 5) == pytest.approx(1 / 3)
