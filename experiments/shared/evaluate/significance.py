from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    verdict: str


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resample: int = 1000,
    confidence: float = 0.95,
    min_abs_diff: float = 0.03,
    seed: int = 42,
) -> BootstrapResult:
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")
    if not scores_a:
        return BootstrapResult(mean_diff=0.0, ci_low=0.0, ci_high=0.0, verdict="tie")
    rng = random.Random(seed)
    diffs = []
    indices = list(range(len(scores_a)))
    for _ in range(n_resample):
        sample = [rng.choice(indices) for _ in indices]
        mean_a = sum(scores_a[i] for i in sample) / len(sample)
        mean_b = sum(scores_b[i] for i in sample) / len(sample)
        diffs.append(mean_a - mean_b)
    diffs.sort()
    mean_diff = sum(diffs) / len(diffs)
    lower_idx = max(0, int(((1.0 - confidence) / 2.0) * len(diffs)))
    upper_idx = min(len(diffs) - 1, int((1.0 - (1.0 - confidence) / 2.0) * len(diffs)) - 1)
    ci_low = diffs[lower_idx]
    ci_high = diffs[upper_idx]
    if mean_diff >= min_abs_diff and ci_low > 0:
        verdict = "A_wins"
    elif mean_diff <= -min_abs_diff and ci_high < 0:
        verdict = "B_wins"
    else:
        verdict = "tie"
    return BootstrapResult(
        mean_diff=round(mean_diff, 6),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
        verdict=verdict,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap significance test")
    parser.add_argument("--scores-a", required=True, help="JSON file containing list[float]")
    parser.add_argument("--scores-b", required=True, help="JSON file containing list[float]")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scores_a = json.loads(Path(args.scores_a).read_text(encoding="utf-8"))
    scores_b = json.loads(Path(args.scores_b).read_text(encoding="utf-8"))
    result = paired_bootstrap(scores_a=scores_a, scores_b=scores_b)
    Path(args.output).write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
