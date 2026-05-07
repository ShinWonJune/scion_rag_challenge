"""Slice a frozen-queries JSONL to keep only top-N search_terms per language.

CRITICAL: step1_search.py reads `search_terms_by_lang` (line 173) preferentially.
Both `search_terms` AND `search_terms_by_lang.{korean,english}` must be sliced.

Usage:
  python slice_frozen_queries.py --input QUERIES.jsonl --n 3 --output OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def slice_record(rec: dict, n: int) -> dict:
    """Slice both `search_terms` and `search_terms_by_lang` to top-N per language."""
    by_lang = dict(rec.get("search_terms_by_lang") or {})
    ko = list(by_lang.get("korean") or [])[:n]
    en = list(by_lang.get("english") or [])[:n]

    out = dict(rec)
    out["search_terms_by_lang"] = {"korean": ko, "english": en}
    out["search_terms"] = ko + en
    out["search_terms_meta"] = {"n": n, "ko_count": len(ko), "en_count": len(en)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, required=True, help="top-N search_terms per language")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_records = 0
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sliced = slice_record(rec, args.n)
            g.write(json.dumps(sliced, ensure_ascii=False) + "\n")
            n_records += 1
    print(f"Wrote {n_records} records to {outp} (n={args.n})")


if __name__ == "__main__":
    main()
