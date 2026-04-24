from experiments.shared.evaluate.significance import paired_bootstrap


def test_a_wins_on_clear_margin() -> None:
    result = paired_bootstrap([1.0] * 20, [0.9] * 20, n_resample=200)
    assert result.verdict == "A_wins"


def test_tie_on_equal_scores() -> None:
    result = paired_bootstrap([0.5] * 20, [0.5] * 20, n_resample=200)
    assert result.verdict == "tie"
