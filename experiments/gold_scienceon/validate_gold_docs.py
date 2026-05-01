from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


DUPLICATE_OR_NEAR_DUPLICATE: dict[str, str] = {
    "21": "동일 제목의 ScienceON 중복 레코드가 있어 같은 논문을 가리키는 대체 gold가 존재한다.",
    "24": "동일 제목의 ScienceON 중복 레코드가 있어 같은 논문을 가리키는 대체 gold가 존재한다.",
    "32": "동일 제목의 ScienceON 중복 레코드가 있어 같은 논문을 가리키는 대체 gold가 존재한다.",
    "36": "동일/근접 제목의 특허 분석 논문 레코드가 있어 같은 연구를 가리키는 대체 gold가 존재한다.",
    "39": "동일 제목의 ScienceON 레코드가 추가로 확인되어 같은 논문을 가리키는 대체 gold가 존재한다.",
}

INDEPENDENT_MULTI_GOLD: dict[str, str] = {
    "37": "저탄소 녹색산업 스마트팜 시스템 논문도 질문 일부에 답할 수 있으나, 현재 gold가 더 직접적으로 농업 분야 IT융합기술의 적용과 효과를 포괄한다.",
}

PARTIAL_GOLD: dict[str, str] = {}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    base = Path("experiments/gold_scienceon/artifacts")
    golds = load_jsonl(base / "scienceon_gold_docs.jsonl")
    candidates = load_jsonl(base / "scienceon_candidate_judgments_top10.jsonl")
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_qid[str(row["qid"])].append(row)

    rows: list[dict[str, str | int | float | None]] = []
    for gold in golds:
        qid = str(gold["qid"])
        judged = by_qid.get(qid, [])
        auto_sufficient = [
            c
            for c in judged
            if c.get("judgment") == "sufficient"
            and c.get("doc", {}).get("doc_id") != gold.get("gold_doc_id")
        ]

        if qid in PARTIAL_GOLD:
            sufficiency = "partial"
            rationale = PARTIAL_GOLD[qid]
        else:
            sufficiency = "sufficient"
            rationale = str(gold.get("judgment_reason") or "")

        if qid in DUPLICATE_OR_NEAR_DUPLICATE:
            multi_gold = "duplicate_same_work"
            multi_gold_note = DUPLICATE_OR_NEAR_DUPLICATE[qid]
        elif qid in INDEPENDENT_MULTI_GOLD:
            multi_gold = "possible_secondary"
            multi_gold_note = INDEPENDENT_MULTI_GOLD[qid]
        else:
            multi_gold = "no"
            multi_gold_note = "질문은 특정 논문/문건의 방법, 결과, 구조를 묻고 있어 현재 gold 1개가 주 근거이다."

        rows.append(
            {
                "qid": qid,
                "gold_doc_id": gold.get("gold_doc_id"),
                "gold_title": gold.get("gold_title"),
                "gold_sufficiency": sufficiency,
                "gold_rationale": rationale,
                "score": gold.get("score"),
                "score_reasons": "; ".join(gold.get("score_reasons") or []),
                "auto_sufficient_alt_in_top10": len(auto_sufficient),
                "actual_multiple_gold": multi_gold,
                "multiple_gold_note": multi_gold_note,
                "best_auto_alt_doc_id": auto_sufficient[0]["doc"]["doc_id"] if auto_sufficient else "",
                "best_auto_alt_title": auto_sufficient[0]["doc"]["title"] if auto_sufficient else "",
                "question": gold.get("question"),
            }
        )

    out_csv = base / "scienceon_gold_validation.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_questions": len(rows),
        "gold_sufficiency_counts": {
            "sufficient": sum(r["gold_sufficiency"] == "sufficient" for r in rows),
            "partial": sum(r["gold_sufficiency"] == "partial" for r in rows),
        },
        "actual_multiple_gold_counts": {
            "no": sum(r["actual_multiple_gold"] == "no" for r in rows),
            "duplicate_same_work": sum(r["actual_multiple_gold"] == "duplicate_same_work" for r in rows),
            "possible_secondary": sum(r["actual_multiple_gold"] == "possible_secondary" for r in rows),
        },
        "auto_sufficient_alt_questions": [
            r["qid"] for r in rows if int(r["auto_sufficient_alt_in_top10"] or 0) > 0
        ],
        "caveat": (
            "auto_sufficient_alt_in_top10 is heuristic and often means topical overlap, "
            "not an independent gold document."
        ),
    }
    (base / "scienceon_gold_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
