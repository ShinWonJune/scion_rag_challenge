from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    base = Path("experiments/gold_scienceon/artifacts")
    gold_rows = {row["qid"]: row for row in load_jsonl(base / "scienceon_gold_docs.jsonl")}
    judge_rows = load_jsonl(base / "scienceon_gold_llm_judge.jsonl")
    full_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []

    for judge in judge_rows:
        qid = str(judge["qid"])
        gold = gold_rows[qid]
        item = {
            "qid": qid,
            "question": gold["question"],
            "gold_doc_id": gold["gold_doc_id"],
            "gold_title": gold["gold_title"],
            "gold_abstract": gold["gold_abstract"],
            "gold_url": gold["gold_url"],
            "source": gold["source"],
            "source_kind": gold["source_kind"],
            "llm_judge_label": judge["llm_judge_label"],
            "llm_judge_rationale": judge["llm_judge_rationale"],
            "multi_gold_label": judge["multi_gold_label"],
            "multi_gold_note": judge["multi_gold_note"],
        }
        if judge["llm_judge_label"] == "FULL":
            full_rows.append(item)
        else:
            partial_rows.append(item)

    full_rows.sort(key=lambda row: int(row["qid"]))
    partial_rows.sort(key=lambda row: int(row["qid"]))
    write_jsonl(base / "scienceon_gold_full_only.jsonl", full_rows)
    write_jsonl(base / "scienceon_gold_excluded_partial.jsonl", partial_rows)
    (base / "scienceon_gold_full_qids.txt").write_text(
        "\n".join(row["qid"] for row in full_rows) + "\n", encoding="utf-8"
    )

    with (base / "scienceon_gold_full_only.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "qid",
            "gold_doc_id",
            "gold_title",
            "gold_url",
            "multi_gold_label",
            "llm_judge_rationale",
            "question",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in full_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "total_gold_docs": len(gold_rows),
        "full_gold_count": len(full_rows),
        "excluded_partial_count": len(partial_rows),
        "full_qids": [row["qid"] for row in full_rows],
        "excluded_partial_qids": [row["qid"] for row in partial_rows],
        "primary_eval_file": "scienceon_gold_full_only.jsonl",
        "qid_filter_file": "scienceon_gold_full_qids.txt",
    }
    (base / "scienceon_gold_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = (
        "# ScienceON Gold Artifacts\n\n"
        "Final evaluation should use only manually judged FULL gold rows.\n\n"
        "Primary files:\n"
        "- `scienceon_gold_full_only.jsonl`: 41 FULL gold questions for final evaluation.\n"
        "- `scienceon_gold_full_only.csv`: compact review table for the 41 FULL rows.\n"
        "- `scienceon_gold_full_qids.txt`: qid allowlist for filtering pipeline outputs.\n"
        "- `scienceon_gold_excluded_partial.jsonl`: 9 excluded PARTIAL rows, kept for audit.\n"
        "- `scienceon_gold_llm_judge.jsonl`: manual LLM-as-judge review of gold abstract sufficiency.\n"
        "- `scienceon_gold_docs.jsonl`: all 50 selected ScienceON gold candidates.\n\n"
        "Do not use the heuristic `score` as an answer-quality metric. It is only a candidate-selection signal.\n"
        "For final answer evaluation, use `scienceon_gold_full_only.jsonl` as evidence and judge pipeline answers against the gold title/abstract.\n"
    )
    (base / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
