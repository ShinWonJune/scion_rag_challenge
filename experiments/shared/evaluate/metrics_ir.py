from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(hits: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return 1.0 if any(doc_id in gold for doc_id in hits[:k]) else 0.0


def precision_at_k(hits: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_hits = hits[:k]
    if not top_hits:
        return 0.0
    return sum(1 for doc_id in top_hits if doc_id in gold) / float(k)


def mrr_at_k(hits: list[str], gold: set[str], k: int) -> float:
    for rank, doc_id in enumerate(hits[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(hits[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_len = min(k, len(gold))
    if ideal_len <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
    return dcg / idcg if idcg else 0.0


def summarize_ir_metrics(
    ranked_doc_ids: dict[str, list[str]],
    gold_by_qid: dict[str, set[str]],
    ks: Iterable[int] = (5, 7, 10, 50),
) -> dict[str, float | int | list[float]]:
    qids = [qid for qid in ranked_doc_ids if qid in gold_by_qid]
    metrics: dict[str, float | int | list[float]] = {"n_queries": len(qids)}
    for k in ks:
        recalls = [recall_at_k(ranked_doc_ids[qid], gold_by_qid[qid], k) for qid in qids]
        precisions = [precision_at_k(ranked_doc_ids[qid], gold_by_qid[qid], k) for qid in qids]
        mrrs = [mrr_at_k(ranked_doc_ids[qid], gold_by_qid[qid], k) for qid in qids]
        ndcgs = [ndcg_at_k(ranked_doc_ids[qid], gold_by_qid[qid], k) for qid in qids]
        metrics[f"recall_at_{k}"] = round(sum(recalls) / len(recalls), 6) if recalls else 0.0
        metrics[f"precision_at_{k}"] = round(sum(precisions) / len(precisions), 6) if precisions else 0.0
        metrics[f"mrr_at_{k}"] = round(sum(mrrs) / len(mrrs), 6) if mrrs else 0.0
        metrics[f"ndcg_at_{k}"] = round(sum(ndcgs) / len(ndcgs), 6) if ndcgs else 0.0
        metrics[f"per_query_recall_at_{k}"] = recalls
    return metrics
