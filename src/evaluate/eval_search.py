from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.evaluate.metrics import calc_qsr, normalize_title


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_result_rows(results_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(results_payload, dict):
        rows = results_payload.get("results", [])
    elif isinstance(results_payload, list):
        rows = results_payload
    else:
        rows = []

    normalized = []
    for idx, row in enumerate(rows):
        docs = row.get("documents", [])
        titles = [normalize_title(d.get("title", "")) for d in docs if d.get("title")]
        normalized.append(
            {
                "idx": idx,
                "question_id": str(row.get("question_id", idx)),
                "query": row.get("query") or row.get("question") or "",
                "titles": set(t for t in titles if t),
            }
        )
    return normalized


def _load_miracl_gt(gt_payload: Any) -> Dict[str, set[str]]:
    gt_map: Dict[str, set[str]] = {}
    items = gt_payload if isinstance(gt_payload, list) else gt_payload.get("data", [])
    for item in items:
        query = item.get("query", "")
        docs = item.get("documents", [])
        titles = {normalize_title(d.get("title", "")) for d in docs if d.get("title")}
        gt_map[query] = {t for t in titles if t}
    return gt_map


def _load_scienceon_gt(gt_payload: Any) -> Dict[str, str]:
    if isinstance(gt_payload, dict):
        return {str(k): normalize_title(v) for k, v in gt_payload.items()}
    mapping: Dict[str, str] = {}
    if isinstance(gt_payload, list):
        for idx, item in enumerate(gt_payload):
            qid = str(item.get("question_id", idx))
            title = item.get("title") or item.get("answer_title") or ""
            mapping[qid] = normalize_title(title)
    return mapping


def evaluate_search(results_path: str, ground_truth_path: str, source: str) -> Dict[str, Any]:
    rows = _extract_result_rows(_load_json(results_path))
    gt_payload = _load_json(ground_truth_path)
    source = source.lower()

    failures: List[Dict[str, Any]] = []
    success_count = 0

    if source in {"miracl_en", "miracl_ko"}:
        gt_map = _load_miracl_gt(gt_payload)
        eval_rows = [r for r in rows if r["query"] in gt_map]
        for row in eval_rows:
            gt_titles = gt_map.get(row["query"], set())
            if row["titles"].intersection(gt_titles):
                success_count += 1
            else:
                failures.append({"query": row["query"], "question_id": row["question_id"]})
        total = len(eval_rows)
    elif source == "scienceon":
        gt_map = _load_scienceon_gt(gt_payload)
        eval_rows = [r for r in rows if r["question_id"] in gt_map]
        for row in eval_rows:
            gt_title = gt_map.get(row["question_id"], "")
            matched = any(
                gt_title == t or gt_title in t or t in gt_title for t in row["titles"] if t and gt_title
            )
            if matched:
                success_count += 1
            else:
                failures.append({"query": row["query"], "question_id": row["question_id"]})
        total = len(eval_rows)
    else:
        raise ValueError(f"Unsupported source: {source}")

    return {
        "source": source,
        "results_path": str(Path(results_path)),
        "ground_truth_path": str(Path(ground_truth_path)),
        "total_queries": total,
        "successful_queries": success_count,
        "failed_queries": total - success_count,
        "qsr": calc_qsr(success_count, total),
        "failures": failures,
    }

