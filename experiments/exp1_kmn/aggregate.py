"""Aggregate Exp1 cell metrics + timings into a CSV.

Usage:
  python -m experiments.exp1_kmn.aggregate --root experiments/outputs/exp1_kmn --out experiments/outputs/exp1_kmn/_aggregate.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_cells(root: Path) -> list[dict]:
    rows = []
    for phase in ["phaseA", "phaseB", "phaseC"]:
        phase_dir = root / phase
        if not phase_dir.exists():
            continue
        for cell_dir in sorted(phase_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            timing = _load_json(cell_dir / "timing.json")
            metrics = _load_json(cell_dir / "metrics.json")
            row = {
                "phase": phase,
                "cell": cell_dir.name,
                "k": timing.get("k"),
                "n": timing.get("n"),
                "m": timing.get("m"),
                "step1_sec": timing.get("step1_sec"),
                "step3_sec": timing.get("step3_sec"),
                "step4_sec": timing.get("step4_sec"),
                "total_sec": timing.get("total_sec"),
                "hit_at_1": metrics.get("hit_at_1"),
                "hit_at_3": metrics.get("hit_at_3"),
                "hit_at_5": metrics.get("hit_at_5"),
                "hit_at_10": metrics.get("hit_at_10"),
                "hit_at_20": metrics.get("hit_at_20"),
                "mrr": metrics.get("mrr_at_10") or metrics.get("mrr"),
                "mean_gold_rank": metrics.get("mean_gold_rank"),
                "gold_found_rate": metrics.get("gold_found_rate"),
                "n_questions": metrics.get("n_questions"),
            }
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/outputs/exp1_kmn")
    ap.add_argument("--out", default="experiments/outputs/exp1_kmn/_aggregate.csv")
    args = ap.parse_args()

    rows = collect_cells(Path(args.root))
    if not rows:
        print(f"No cells found under {args.root}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")

    # Pick best Phase A cell by Hit@5, MRR tiebreak
    phaseA = [r for r in rows if r["phase"] == "phaseA" and r["hit_at_5"] is not None]
    if phaseA:
        phaseA.sort(key=lambda r: (-(r["hit_at_5"] or 0), -(r["mrr"] or 0), (r["total_sec"] or 1e9)))
        best = phaseA[0]
        print(f"Phase A best: {best['cell']} hit@5={best['hit_at_5']} mrr={best['mrr']} t={best['total_sec']}s")


if __name__ == "__main__":
    main()
