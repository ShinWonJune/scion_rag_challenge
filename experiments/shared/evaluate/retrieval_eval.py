"""CLI: compute Hit@k, MRR, mean_gold_rank from retrieval JSONL + gold JSON.

Usage:
    python -m experiments.shared.evaluate.retrieval_eval \
        --retrieval path/to/retrieval.jsonl \
        --gold path/to/gold.json \
        [--ks 1 3 5 7 10 20]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_retrieval(path: Path) -> dict[str, list[str]]:
    """Load retrieval JSONL: each line {"question_id": ..., "hits": [{"doc_id": ...}, ...]}"""
    result: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("id") or row.get("qid", ""))
            hits = row.get("hits", [])
            if hits and isinstance(hits[0], dict):
                doc_ids = [str(h.get("doc_id", h.get("cn", ""))) for h in hits]
            else:
                doc_ids = [str(h) for h in hits]
            result[qid] = doc_ids
    return result


def _load_gold(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): {str(v).strip() for v in vs} for k, vs in payload.items()}


def _hit_at_k(hits: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(d in gold for d in hits[:k]) else 0.0


def _mrr(hits: list[str], gold: set[str], k: int) -> float:
    for rank, d in enumerate(hits[:k], start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def _mean_gold_rank(hits: list[str], gold: set[str]) -> float | None:
    ranks = [i + 1 for i, d in enumerate(hits) if d in gold]
    return sum(ranks) / len(ranks) if ranks else None


def evaluate(
    retrieval: dict[str, list[str]],
    gold: dict[str, set[str]],
    ks: list[int],
) -> dict:
    qids = [qid for qid in retrieval if qid in gold]
    if not qids:
        return {"error": "no matching qids between retrieval and gold", "n_queries": 0}

    metrics: dict = {"n_queries": len(qids)}
    for k in ks:
        hits_list = [_hit_at_k(retrieval[q], gold[q], k) for q in qids]
        mrr_list = [_mrr(retrieval[q], gold[q], k) for q in qids]
        metrics[f"hit_at_{k}"] = round(sum(hits_list) / len(hits_list), 6)
        metrics[f"mrr_at_{k}"] = round(sum(mrr_list) / len(mrr_list), 6)

    gold_ranks = [_mean_gold_rank(retrieval[q], gold[q]) for q in qids]
    valid_ranks = [r for r in gold_ranks if r is not None]
    metrics["mean_gold_rank"] = round(sum(valid_ranks) / len(valid_ranks), 4) if valid_ranks else None
    metrics["gold_found_rate"] = round(len(valid_ranks) / len(qids), 6)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluator: Hit@k, MRR, mean_gold_rank")
    parser.add_argument("--retrieval", required=True, help="Retrieval JSONL path")
    parser.add_argument("--gold", required=True, help="Gold JSON path")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 7, 10, 20], help="k values")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    retrieval = _load_retrieval(Path(args.retrieval))
    gold = _load_gold(Path(args.gold))
    metrics = evaluate(retrieval, gold, args.ks)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
