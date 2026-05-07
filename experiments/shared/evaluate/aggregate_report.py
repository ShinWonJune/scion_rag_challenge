"""Aggregate per-cell experiment outputs into a combined CSV + Pareto plot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_metrics(cell_dir: Path) -> dict[str, Any] | None:
    metrics_path = cell_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _classify_failure(
    cell_dir: Path,
    gold_by_qid: dict[str, set[str]],
    top_k: int = 5,
) -> dict[str, int]:
    retrieval_path = cell_dir / "retrieval.jsonl"
    if not retrieval_path.exists():
        return {"f1_gold_not_in_corpus": 0, "f2_gold_in_corpus_rank_gt_k": 0}

    f1 = f2 = 0
    with retrieval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", ""))
            hits = row.get("hits", [])
            gold_ids = gold_by_qid.get(qid, set())
            if not gold_ids:
                continue
            corpus_ids = {str(h.get("doc_id", "")) for h in hits}
            top_ids = [str(h.get("doc_id", "")) for h in hits[:top_k]]
            gold_in_corpus = bool(gold_ids & corpus_ids)
            gold_in_top = bool(gold_ids & set(top_ids))
            if not gold_in_corpus:
                f1 += 1
            elif not gold_in_top:
                f2 += 1
    return {"f1_gold_not_in_corpus": f1, "f2_gold_in_corpus_rank_gt_k": f2}


def _mean_gold_rank(cell_dir: Path, gold_by_qid: dict[str, set[str]]) -> float | None:
    retrieval_path = cell_dir / "retrieval.jsonl"
    if not retrieval_path.exists():
        return None
    ranks = []
    with retrieval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", ""))
            gold_ids = gold_by_qid.get(qid, set())
            for hit in row.get("hits", []):
                if str(hit.get("doc_id", "")) in gold_ids:
                    ranks.append(hit.get("rank", len(row["hits"])))
                    break
    return round(sum(ranks) / len(ranks), 4) if ranks else None


def build_aggregate(
    cell_dirs: list[Path],
    gold_path: Path,
    output_csv: Path,
    plot_output: Path | None = None,
) -> list[dict[str, Any]]:
    gold_by_qid: dict[str, set[str]] = {}
    if gold_path.exists():
        raw = json.loads(gold_path.read_text(encoding="utf-8"))
        gold_by_qid = {str(k): {str(v) for v in vals} for k, vals in raw.items()}

    records: list[dict[str, Any]] = []
    for cell_dir in cell_dirs:
        metrics = _load_metrics(cell_dir)
        if metrics is None:
            continue
        failures = _classify_failure(cell_dir, gold_by_qid)
        mean_rank = _mean_gold_rank(cell_dir, gold_by_qid)
        record: dict[str, Any] = {
            "cell": cell_dir.name,
            "cell_path": str(cell_dir),
            "hit_at_5": metrics.get("recall_at_5", ""),
            "hit_at_10": metrics.get("recall_at_10", ""),
            "mrr_at_10": metrics.get("mrr_at_10", ""),
            "ndcg_at_10": metrics.get("ndcg_at_10", ""),
            "mean_gold_rank": mean_rank,
            "runtime_sec": metrics.get("runtime_sec", ""),
            "f1_gold_not_in_corpus": failures["f1_gold_not_in_corpus"],
            "f2_gold_in_corpus_rank_gt_k": failures["f2_gold_in_corpus_rank_gt_k"],
            **{k: v for k, v in metrics.items() if k not in (
                "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10",
                "runtime_sec", "case_id", "n_queries",
            ) and not k.startswith("per_query_") and not k.startswith("precision_")},
        }
        records.append(record)

    if records:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0].keys())
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    if plot_output is not None and records:
        from experiments.shared.evaluate.plot_utils import pareto_plot
        xs = [float(r.get("runtime_sec") or 0) for r in records]
        ys = [float(r.get("hit_at_5") or 0) for r in records]
        labels = [str(r["cell"]) for r in records]
        pareto_plot(xs, ys, labels, xlabel="Total Time (s)", ylabel="Hit@5", title="Pareto: Hit@5 vs Runtime", output_path=plot_output)

    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-cell metrics into combined CSV + Pareto plot")
    parser.add_argument("--cells", nargs="+", required=True, help="Cell directories")
    parser.add_argument("--gold", required=True, help="Gold JSON path")
    parser.add_argument("--output_csv", required=True, help="Output aggregate CSV path")
    parser.add_argument("--plot", default=None, help="Optional Pareto plot PNG output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cell_dirs = [Path(c) for c in args.cells]
    records = build_aggregate(
        cell_dirs,
        gold_path=Path(args.gold),
        output_csv=Path(args.output_csv),
        plot_output=Path(args.plot) if args.plot else None,
    )
    print(f"Aggregated {len(records)} cells -> {args.output_csv}")


if __name__ == "__main__":
    main()
