"""셔플 frozen queries 의 ScienceON 캐시 커버리지 확인.

step1_search 의 캐시 키는 (source, term, cur_page, row_count, fields) 이다.
n=10, k=30 (= max_pages=3, row_count=10), m=50 조건에서 각 검색식 term 에 대해
3개 페이지(cur_page=1..3) × row_count=10 의 캐시 hit 여부를 보고한다.

baseline 과 셔플의 cache 커버리지 차이를 먼저 확인해야 offline simulation 의
공정성을 판단할 수 있다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIELDS = ["CN", "title", "abstract", "author", "year", "link"]
CACHE_ROOT = Path("outputs/_shared_cache/scienceon")


def key_path(term: str, page: int) -> Path:
    key = {
        "source": "scienceon",
        "term": term.strip(),
        "cur_page": int(page),
        "row_count": 10,
        "fields": list(FIELDS),
    }
    digest = hashlib.sha1(
        json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return CACHE_ROOT / f"{digest}.json"


def coverage_for_record(rec: dict, n: int, max_pages: int) -> dict:
    ko = (rec.get("search_terms_by_lang", {}) or {}).get("korean", [])[:n]
    en = (rec.get("search_terms_by_lang", {}) or {}).get("english", [])[:n]
    terms = list(ko) + list(en)

    total = 0
    hits = 0
    miss_terms: list[tuple[str, int]] = []
    for term in terms:
        for page in range(1, max_pages + 1):
            total += 1
            p = key_path(term, page)
            if p.exists():
                hits += 1
            else:
                miss_terms.append((term, page))
    return {
        "qid": rec.get("question_id"),
        "total_lookups": total,
        "hits": hits,
        "miss_count": total - hits,
        "miss_examples": miss_terms[:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-pages", type=int, default=3, help="k=30 → 3 pages of 10")
    args = ap.parse_args()

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    rows = [coverage_for_record(json.loads(l), args.n, args.max_pages) for l in lines if l.strip()]

    total = sum(r["total_lookups"] for r in rows)
    hits = sum(r["hits"] for r in rows)
    miss = total - hits
    perq_full = sum(1 for r in rows if r["miss_count"] == 0)
    print(f"file={args.input}")
    print(f"queries={len(rows)} total_lookups={total} hits={hits} miss={miss} hit_rate={hits/max(1,total):.3f}")
    print(f"queries with ZERO cache miss: {perq_full}/{len(rows)}")
    worst = sorted(rows, key=lambda r: -r["miss_count"])[:3]
    for r in worst:
        print(f"  qid={r['qid']} miss={r['miss_count']}/{r['total_lookups']} ex={r['miss_examples']}")


if __name__ == "__main__":
    main()
