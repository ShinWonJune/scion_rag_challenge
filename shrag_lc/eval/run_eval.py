"""Retrieval evaluation against ``data/test.csv`` gold titles.

The gold document titles per question live in the ``retrieved_article_name_1..N``
columns. We match retrieved titles to gold titles by a normalized form
(lowercase + strip non-word/Korean chars), mirroring
``shrag/evaluation/metrics.py:normalize_title``, then compute hit@k / MRR / nDCG.

Usage:
    python -m shrag_lc.eval.run_eval --retrieval retrieved.jsonl --gold data/test.csv

``retrieved.jsonl`` lines: {"id": "<qid>", "titles": ["...", "..."]}  (ranked).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .ir_metrics import summarize_ir_metrics

_NORM_RE = re.compile(r"[^\w가-힣]", flags=re.UNICODE)


def normalize_title(title: str) -> str:
    return _NORM_RE.sub("", (title or "").lower().strip())


def load_gold_titles_from_csv(csv_path: str | Path) -> dict[str, set[str]]:
    """Map question id -> set of normalized gold titles (retrieved_article_name_*)."""
    gold: dict[str, set[str]] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        lower = {name.lower(): name for name in (reader.fieldnames or [])}
        idcol = lower.get("id") or lower.get("question_id")
        title_cols = [orig for low, orig in lower.items() if low.startswith("retrieved_article_name")]
        for idx, row in enumerate(reader):
            qid = (row.get(idcol) or str(idx)).strip() if idcol else str(idx)
            titles = {
                normalize_title(row[c]) for c in title_cols if (row.get(c) or "").strip()
            }
            titles.discard("")
            if titles:
                gold[qid] = titles
    return gold


def evaluate_titles(
    ranked_titles_by_qid: dict[str, list[str]],
    gold_titles_by_qid: dict[str, set[str]],
    ks=(1, 3, 5, 10),
) -> dict:
    ranked_norm = {
        qid: [normalize_title(t) for t in titles]
        for qid, titles in ranked_titles_by_qid.items()
    }
    return summarize_ir_metrics(ranked_norm, gold_titles_by_qid, ks=ks)


def _load_retrieval_jsonl(path: str | Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item.get("id") or item.get("question_id") or idx)
            out[qid] = list(item.get("titles") or [])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval vs test.csv gold titles")
    parser.add_argument("--retrieval", required=True, help="JSONL: {id, titles:[...]}")
    parser.add_argument("--gold", default="data/test.csv", help="Gold CSV with retrieved_article_name_*")
    parser.add_argument("--ks", default="1,3,5,10")
    args = parser.parse_args()

    ks = tuple(int(k) for k in args.ks.split(","))
    ranked = _load_retrieval_jsonl(args.retrieval)
    gold = load_gold_titles_from_csv(args.gold)
    metrics = evaluate_titles(ranked, gold, ks=ks)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
