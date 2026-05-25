"""Offline acquisition simulation using set-aware cache replay.

전제 — ScienceON OR(|) 검색은 키워드 집합에 대해 commutative 하다.
즉 'A|B|C' 와 'C|A|B' 는 동일 결과 셋을 반환한다. 캐시는 정확 문자열로 키를
구성하지만, 본 시뮬레이터는 cache 의 모든 entry 를 (frozenset(keywords), page)
인덱스로 다시 정리해, 셔플된 검색식에 대해서도 동일 set 의 캐시를 재사용한다.

step1_search 의 동작을 재현:
  - 각 질문, 언어 별로 search_terms 를 차례로 사용
  - 누적 doc 수가 target_documents(m) 이상이면 중단
  - max_pages 페이지까지 시도, 빈 페이지면 종료
  - dedup by doc_id

출력: gold_found_rate, hits/총 질문, term coverage 통계.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CACHE_ROOT = Path("outputs/_shared_cache/scienceon")
FIELDS = ["CN", "title", "abstract", "author", "year", "link"]
ROW_COUNT = 10


def term_to_keyset(term: str) -> frozenset[str]:
    return frozenset(t for t in term.split("|") if t)


def build_keyset_index(cache_root: Path) -> dict[tuple[frozenset[str], int], list[dict[str, Any]]]:
    """Index every cache entry by (frozenset(keywords), cur_page)."""
    index: dict[tuple[frozenset[str], int], list[dict[str, Any]]] = {}
    for f in cache_root.glob("*.json"):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = payload.get("key") or {}
        term = key.get("term", "")
        page = int(key.get("cur_page", 0))
        if key.get("row_count") != ROW_COUNT or list(key.get("fields") or []) != FIELDS:
            continue
        keyset = term_to_keyset(term)
        if not keyset:
            continue
        val = payload.get("value") or []
        index[(keyset, page)] = val
    return index


def is_quality(row: dict[str, Any]) -> bool:
    title = str(row.get("title", "") or "").strip()
    if not title or len(title) < 5:
        return False
    abstract = str(row.get("abstract", "") or "").strip()
    if (not abstract or abstract == "없음") and len(title) < 20:
        return False
    return bool(str(row.get("CN", "") or "").strip())


def to_doc(row: dict[str, Any]) -> dict[str, Any]:
    doc_id = str(row.get("CN", "") or "").strip()
    return {"doc_id": doc_id, "title": row.get("title", "")}


def dedup(docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for d in docs:
        key = d.get("doc_id") or d.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def search_with_cache(
    index: dict[tuple[frozenset[str], int], list[dict[str, Any]]],
    term: str,
    max_results: int,
    max_pages: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Replay ScienceONAdapter.search() against the cache index. Returns (docs, hit_pages, miss_pages)."""
    keyset = term_to_keyset(term)
    if not keyset:
        return [], 0, 0
    all_docs: list[dict[str, Any]] = []
    hit_pages = miss_pages = 0
    for page in range(1, max_pages + 1):
        if len(all_docs) >= max_results:
            break
        remaining = max_results - len(all_docs)
        rows = index.get((keyset, page))
        if rows is None:
            miss_pages += 1
            continue  # mirror "empty page → break" via the outer loop; here just skip
        hit_pages += 1
        filtered = [to_doc(r) for r in rows if is_quality(r)]
        if not filtered:
            break
        take = filtered[: min(ROW_COUNT, remaining)]
        all_docs.extend(take)
    return dedup(all_docs)[:max_results], hit_pages, miss_pages


def simulate_question(
    rec: dict[str, Any],
    index: dict[tuple[frozenset[str], int], list[dict[str, Any]]],
    n: int,
    m: int,
    max_pages: int,
) -> dict[str, Any]:
    by_lang = rec.get("search_terms_by_lang") or {}
    ko_terms = list(by_lang.get("korean") or [])[:n]
    en_terms = list(by_lang.get("english") or [])[:n]

    per_lang_docs: dict[str, list[dict[str, Any]]] = {}
    hits = miss = 0
    for lang, terms in (("korean", ko_terms), ("english", en_terms)):
        collected: list[dict[str, Any]] = []
        for term in terms:
            if len(collected) >= m:
                break
            remaining = m - len(collected)
            docs, h, mi = search_with_cache(index, term, remaining, max_pages)
            hits += h
            miss += mi
            collected = dedup(collected + docs)[:m]
        per_lang_docs[lang] = collected[:m]

    combined = dedup(per_lang_docs["korean"] + per_lang_docs["english"])
    return {
        "question_id": rec.get("question_id"),
        "doc_ids": [d["doc_id"] for d in combined],
        "ko_count": len(per_lang_docs["korean"]),
        "en_count": len(per_lang_docs["english"]),
        "total": len(combined),
        "cache_hit_pages": hits,
        "cache_miss_pages": miss,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--gold", default="data/gold/scienceon_gold.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--m", type=int, default=50)
    ap.add_argument("--k", type=int, default=30, help="k = 10 * max_pages")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    max_pages = max(1, args.k // 10)
    index = build_keyset_index(CACHE_ROOT)
    print(f"[{args.label}] cache index size: {len(index)} (keyset, page) entries")

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    frozen = [json.loads(l) for l in Path(args.frozen).read_text(encoding="utf-8").splitlines() if l.strip()]

    per_q = []
    for rec in frozen:
        out = simulate_question(rec, index, args.n, args.m, max_pages)
        per_q.append(out)

    found = total_with_gold = total_docs = 0
    coverage_rates = []
    for r in per_q:
        qid = str(r["question_id"])
        gids = set(gold.get(qid, []) or [])
        if not gids:
            continue
        total_with_gold += 1
        collected = set(r["doc_ids"])
        if gids & collected:
            found += 1
        total_docs += r["total"]
        coverage_rates.append(len(gids & collected) / len(gids))

    summary = {
        "label": args.label,
        "n_questions": total_with_gold,
        "gold_found_rate": round(found / max(1, total_with_gold), 4),
        "gold_found": found,
        "mean_docs_per_q": round(total_docs / max(1, total_with_gold), 2),
        "mean_gold_coverage": round(sum(coverage_rates) / max(1, len(coverage_rates)), 4),
        "cache_hit_pages": sum(r["cache_hit_pages"] for r in per_q),
        "cache_miss_pages": sum(r["cache_miss_pages"] for r in per_q),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = Path(f"experiments/outputs/exp_no_importance_order/{args.label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "per_question.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_q) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
