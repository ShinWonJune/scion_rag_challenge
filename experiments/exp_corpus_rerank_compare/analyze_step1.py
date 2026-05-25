"""검색어별 수집 행태 분석.

각 (질문, 언어)에 대해 step1_search 의 _search_terms_until_target 루프를
캐시 replay 로 시뮬레이션하여:
  - 실제로 m=50에 도달하기까지 몇 개의 검색어가 사용됐는가
  - 검색어당 평균 페이지 호출 수, 평균 신규 doc 기여 수
  - 마지막에 m=50을 채웠는지(saturated) 아니면 검색어를 다 소진했는지
를 정량 비교한다.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

CACHE_ROOT = Path("outputs/_shared_cache/scienceon")
FIELDS = ["CN", "title", "abstract", "author", "year", "link"]
ROW_COUNT = 10


def cache_path(term: str, page: int) -> Path:
    key = {"source": "scienceon", "term": term.strip(),
           "cur_page": page, "row_count": ROW_COUNT, "fields": list(FIELDS)}
    h = hashlib.sha1(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return CACHE_ROOT / f"{h}.json"


def is_quality(r):
    title = str(r.get("title","") or "").strip()
    if not title or len(title) < 5: return False
    abstract = str(r.get("abstract","") or "").strip()
    if (not abstract or abstract == "없음") and len(title) < 20: return False
    return bool(str(r.get("CN","") or "").strip())


def simulate(meta_path: str, max_pages: int, target=50):
    meta = json.load(open(meta_path))
    summary = {
        "n_questions": 0,
        "terms_used_total": 0,
        "pages_fetched_total": 0,
        "saturated_count": 0,            # questions×langs that filled m=50
        "exhausted_count": 0,            # questions×langs that ran out of terms
        "early_break_pages": 0,          # term-page combos skipped because m reached
        "per_q": [],
    }
    for r in meta["results"]:
        qid = r["question_id"]
        for lang in ("korean", "english"):
            terms = (r.get("search_terms_by_lang") or {}).get(lang) or []
            summary["n_questions"] += 1
            collected_ids = set()
            terms_used = 0
            pages = 0
            saturated = False
            for term in terms:
                if len(collected_ids) >= target:
                    break
                terms_used += 1
                used_this_term = 0
                for page in range(1, max_pages + 1):
                    if len(collected_ids) >= target:
                        break
                    p = cache_path(term, page)
                    if not p.exists():
                        continue  # treat miss as empty (shouldn't happen here)
                    pages += 1
                    rows = json.loads(p.read_text(encoding="utf-8")).get("value") or []
                    rows = [x for x in rows if is_quality(x)]
                    new = 0
                    for row in rows:
                        did = str(row.get("CN","")).strip()
                        if did and did not in collected_ids and len(collected_ids) < target:
                            collected_ids.add(did); new += 1
                            used_this_term += 1
                    if not rows:
                        break  # empty page → stop term
                if len(collected_ids) >= target:
                    saturated = True; break
            summary["terms_used_total"] += terms_used
            summary["pages_fetched_total"] += pages
            if saturated: summary["saturated_count"] += 1
            else: summary["exhausted_count"] += 1
            summary["per_q"].append({"qid": qid, "lang": lang, "terms_used": terms_used,
                                     "pages": pages, "docs": len(collected_ids), "saturated": saturated})
    return summary


def main():
    targets = {
        "A_k20_c10 (max_pages=2)": (
            "experiments/outputs/exp_corpus_rerank_compare/A_k20_c10/search/260519_151048/search_meta_results.json", 2),
        "B_k30_c5 (max_pages=3)": (
            "experiments/outputs/exp_corpus_rerank_compare/B_k30_c5/search/260519_151105/search_meta_results.json", 3),
    }
    out = {}
    for label, (p, mp) in targets.items():
        s = simulate(p, mp)
        out[label] = {k: v for k, v in s.items() if k != "per_q"}
        out[label]["mean_terms_per_qlang"] = round(s["terms_used_total"] / s["n_questions"], 3)
        out[label]["mean_pages_per_qlang"] = round(s["pages_fetched_total"] / s["n_questions"], 3)
        out[label]["mean_pages_per_term"] = round(s["pages_fetched_total"] / max(1, s["terms_used_total"]), 3)
        out[label]["saturation_rate"] = round(s["saturated_count"] / s["n_questions"], 3)
        out[label]["per_q"] = s["per_q"]

    # print summary
    print(f"{'metric':28s}  {'A_k20_c10':>14s}  {'B_k30_c5':>14s}")
    keys = ["n_questions","terms_used_total","pages_fetched_total",
            "mean_terms_per_qlang","mean_pages_per_qlang","mean_pages_per_term",
            "saturated_count","exhausted_count","saturation_rate"]
    for k in keys:
        a = out["A_k20_c10 (max_pages=2)"][k]; b = out["B_k30_c5 (max_pages=3)"][k]
        print(f"  {k:26s}  {str(a):>14s}  {str(b):>14s}")

    Path("experiments/outputs/exp_corpus_rerank_compare/step1_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
