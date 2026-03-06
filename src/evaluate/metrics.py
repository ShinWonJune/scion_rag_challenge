from __future__ import annotations

import re
from typing import Sequence

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score


def normalize_title(title: str) -> str:
    text = (title or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s가-힣]", "", text)
    return text.strip()


def calc_qsr(success_count: int, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    return success_count / total_count


def calc_bleu(references: Sequence[str], predictions: Sequence[str]) -> float:
    if not references or not predictions:
        return 0.0
    ref_tokens = [[[t for t in (ref or "").lower().split()]] for ref in references]
    pred_tokens = [[t for t in (pred or "").lower().split()] for pred in predictions]
    if not any(any(x[0] for x in refs) for refs in ref_tokens):
        return 0.0
    if not any(pred_tokens):
        return 0.0
    try:
        return float(corpus_bleu(ref_tokens, pred_tokens))
    except Exception:
        return 0.0


def calc_meteor(references: Sequence[str], predictions: Sequence[str]) -> float:
    if not references or not predictions:
        return 0.0
    scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = (ref or "").lower().split()
        pred_tokens = (pred or "").lower().split()
        if not ref_tokens or not pred_tokens:
            scores.append(0.0)
            continue
        try:
            scores.append(float(meteor_score([ref_tokens], pred_tokens)))
        except Exception:
            scores.append(0.0)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))

