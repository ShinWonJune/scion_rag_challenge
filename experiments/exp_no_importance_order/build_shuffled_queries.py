"""중요도 순 정렬을 제거한 frozen queries 생성.

원본 frozen queries(queries_v3_chatgpt_gold41.jsonl)는 키워드가 중요도 순으로
정렬되어 있다. 본 스크립트는 동일 키워드 집합에 대해 고정 시드로 셔플한
순서를 적용하고, build_search_terms 로 rotation truncation 검색식을 재생성한다.

키워드 집합 자체는 보존하므로 LLM 재호출 변동 없이 '중요도 순서' 변수만
격리해 acquisition 영향을 비교할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from shrag.search.extractors.search_terms import build_search_terms


def shuffle_record(rec: dict, rng: random.Random) -> dict:
    keywords = dict(rec.get("keywords", {}))
    shuffled = {}
    for lang in ("korean", "english"):
        kws = list(keywords.get(lang, []))
        rng.shuffle(kws)
        shuffled[lang] = kws

    terms = build_search_terms(shuffled, number_of_operators=0)
    ko_terms = build_search_terms({"korean": shuffled["korean"], "english": []}, number_of_operators=0)
    en_terms = build_search_terms({"korean": [], "english": shuffled["english"]}, number_of_operators=0)

    out = dict(rec)
    out["keywords"] = shuffled
    out["search_terms"] = terms
    out["search_terms_by_lang"] = {"korean": ko_terms, "english": en_terms}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="원본 frozen queries jsonl")
    ap.add_argument("--output", required=True, help="셔플된 frozen queries 출력 경로")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.input).read_text(encoding="utf-8").splitlines()
    out_lines = []
    for line in src:
        if not line.strip():
            continue
        rec = json.loads(line)
        out_lines.append(json.dumps(shuffle_record(rec, rng), ensure_ascii=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(out_lines)} records to {out_path}")


if __name__ == "__main__":
    main()
