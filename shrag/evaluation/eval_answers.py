from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from shrag.evaluation.metrics import calc_bleu, calc_meteor


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _find_col(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    lookup = {c.lower(): c for c in fieldnames}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _normalize_label(value: str) -> str:
    v = (value or "").strip().lower()
    mapping = {
        "supported": "support",
        "support": "support",
        "refuted": "refute",
        "refute": "refute",
        "true": "yes",
        "false": "no",
        "yes": "yes",
        "no": "no",
        "maybe": "maybe",
    }
    return mapping.get(v, v)


def _macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1_sum = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_sum += f1
    return f1_sum / len(labels)


def _align_rows(gt_rows: List[Dict[str, str]], pred_rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not gt_rows or not pred_rows:
        return [], []

    gt_keys = gt_rows[0].keys()
    pred_keys = pred_rows[0].keys()
    key_col = _find_col(gt_keys, ["pubid", "id", "question_id", "query_id", "question"])
    pred_key_col = _find_col(pred_keys, ["pubid", "id", "question_id", "query_id", "question"])
    if not key_col or not pred_key_col:
        n = min(len(gt_rows), len(pred_rows))
        return gt_rows[:n], pred_rows[:n]

    pred_map = {row.get(pred_key_col, ""): row for row in pred_rows}
    aligned_gt, aligned_pred = [], []
    for row in gt_rows:
        key = row.get(key_col, "")
        if key in pred_map:
            aligned_gt.append(row)
            aligned_pred.append(pred_map[key])
    return aligned_gt, aligned_pred


def evaluate_answers(predictions_path: str, ground_truth_path: str, dataset: str) -> Dict[str, Any]:
    gt_rows = _read_csv(ground_truth_path)
    pred_rows = _read_csv(predictions_path)
    gt_rows, pred_rows = _align_rows(gt_rows, pred_rows)
    dataset = dataset.lower()

    result: Dict[str, Any] = {
        "dataset": dataset,
        "predictions_path": str(Path(predictions_path)),
        "ground_truth_path": str(Path(ground_truth_path)),
        "total_samples": len(gt_rows),
    }

    if not gt_rows:
        result["error"] = "No aligned rows for evaluation."
        return result

    gt_keys = gt_rows[0].keys()
    pred_keys = pred_rows[0].keys()

    ref_text_col = _find_col(gt_keys, ["long_answer", "answer", "reference", "gold_answer"])
    pred_text_col = _find_col(pred_keys, ["long_answer", "answer", "prediction", "predicted_answer"])
    if ref_text_col and pred_text_col:
        refs = [row.get(ref_text_col, "") for row in gt_rows]
        preds = [row.get(pred_text_col, "") for row in pred_rows]
        result["bleu"] = calc_bleu(refs, preds)
        result["meteor"] = calc_meteor(refs, preds)

    gt_label_col = _find_col(gt_keys, ["final_decision", "label", "stance", "decision"])
    pred_label_col = _find_col(pred_keys, ["final_decision", "predicted_label", "label", "prediction"])
    if gt_label_col and pred_label_col:
        y_true = [_normalize_label(row.get(gt_label_col, "")) for row in gt_rows]
        y_pred = [_normalize_label(row.get(pred_label_col, "")) for row in pred_rows]
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        result["accuracy"] = correct / len(y_true) if y_true else 0.0
        result["macro_f1"] = _macro_f1(y_true, y_pred)
        result["label_distribution_true"] = dict(Counter(y_true))
        result["label_distribution_pred"] = dict(Counter(y_pred))

    if dataset == "scienceon" and ref_text_col and pred_text_col:
        refs = [row.get(ref_text_col, "").strip().lower() for row in gt_rows]
        preds = [row.get(pred_text_col, "").strip().lower() for row in pred_rows]
        result["exact_match"] = (
            sum(1 for r, p in zip(refs, preds) if r and r == p) / len(refs) if refs else 0.0
        )

    return result

