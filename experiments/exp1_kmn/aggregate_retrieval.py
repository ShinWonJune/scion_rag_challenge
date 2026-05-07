"""Combine per-question retrieval JSONs (from step4) into a single retrieval.jsonl
compatible with retrieval_eval (one line per qid: {question_id, hits:[{doc_id}]}).

Usage:
  python -m experiments.exp1_kmn.aggregate_retrieval --in DIR --out FILE.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Directory containing per-question step4 JSON files")
    ap.add_argument("--out", required=True, help="Output retrieval.jsonl")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.exists():
        raise SystemExit(f"input dir not found: {indir}")

    rows = []
    for f in sorted(indir.iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        qid = str(doc.get("id") or doc.get("question_id") or "")
        if not qid:
            continue
        # Pick the "original" query result for hits — fall back to first.
        rr = doc.get("retrieval_results") or []
        hits = []
        if isinstance(rr, list) and rr:
            chosen = next((r for r in rr if (r.get("query_meta") or {}).get("type") == "original"), rr[0])
            hits = chosen.get("hits") or []
        elif isinstance(rr, dict):
            hits = rr.get("hits") or []
        out_hits = [{"doc_id": str(h.get("doc_id", "")), "rank": h.get("rank"), "score": h.get("score")} for h in hits]
        rows.append({"question_id": qid, "hits": out_hits})

    rows.sort(key=lambda r: int(r["question_id"]) if r["question_id"].isdigit() else r["question_id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} retrieval records to {out_path}")


if __name__ == "__main__":
    main()
