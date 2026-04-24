from experiments.shared.evaluate.metrics_ir import summarize_ir_metrics


def test_metrics_summary_has_expected_keys() -> None:
    ranked = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
    gold = {"q1": {"d1"}, "q2": {"d5"}}
    metrics = summarize_ir_metrics(ranked, gold)
    assert "recall_at_5" in metrics
    assert "mrr_at_10" in metrics
