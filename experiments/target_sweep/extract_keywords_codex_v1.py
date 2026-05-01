"""v1 (legacy) 프롬프트를 Codex CLI로 재현하는 키워드 추출 runner.

`legacy_v1_prompts.py`의 system+user 결합 프롬프트를 codex에 1회 호출 → 한·영 키워드를
한꺼번에 받아 `v1_categorize`로 언어별 분리 → `v1_generate_search_terms`로 단순 OR 2개
검색어 생성. **공백/하이픈 split, rotation, AND-operator 모두 미사용** (v1 동작 그대로).

산출물은 `pipeline.step1_search --frozen-queries`가 그대로 소비할 수 있는 JSONL.

CLI 예:
    python -m experiments.target_sweep.extract_keywords_codex_v1 \
      --questions data/test.csv --gold data/gold/scienceon_gold.json \
      --output experiments/outputs/frozen/queries_codex_gold41_v1.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Set

from src.codex_client import CodexClient

from experiments.target_sweep.legacy_v1_prompts import (
    v1_combined_prompt,
    v1_parse_keywords,
    v1_categorize,
    v1_generate_search_terms,
)

logger = logging.getLogger(__name__)


def _load_questions(path: Path, gold_qids: Optional[Set[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = str(obj.get("id") or obj.get("question_id") or idx)
                question = (obj.get("question") or obj.get("query") or obj.get("text") or "").strip()
                if not question:
                    continue
                if gold_qids is None or qid in gold_qids:
                    rows.append({"question_id": qid, "query": question})
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, item in enumerate(reader):
            qid = str(idx)
            question = (item.get("Question") or item.get("question") or "").strip()
            if not question:
                continue
            if gold_qids is None or qid in gold_qids:
                rows.append({"question_id": qid, "query": question})
    return rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="v1 legacy prompt + Codex CLI keyword extractor")
    parser.add_argument("--questions", required=True, help="Questions CSV or JSONL")
    parser.add_argument("--gold", default=None, help="Optional gold JSON (filter to qids)")
    parser.add_argument("--output", required=True, help="Output frozen-queries JSONL path")
    parser.add_argument("--codex-model", default="gpt-5.4")
    parser.add_argument("--codex-timeout-sec", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=1500)
    args = parser.parse_args()

    gold_qids: Optional[Set[str]] = None
    if args.gold:
        gold_obj = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        gold_qids = {str(k) for k in gold_obj.keys()}
        logger.info("Filtering to %d gold qids", len(gold_qids))

    rows = _load_questions(Path(args.questions), gold_qids)
    if not rows:
        logger.error("No questions loaded")
        return 1
    logger.info("Loaded %d questions for v1-prompt extraction (single call per question)", len(rows))

    client = CodexClient(model=args.codex_model, timeout_sec=args.codex_timeout_sec)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failures: list[dict[str, Any]] = []
    start = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, 1):
            qid = row["question_id"]
            question = row["query"]
            prompt = v1_combined_prompt(question)
            keywords: dict[str, list[str]] = {"korean": [], "english": []}
            search_terms: list[str] = [question]
            try:
                raw = client.generate_answer_with_prompt(prompt, max_tokens=args.max_tokens)
                parsed = v1_parse_keywords(raw)
                keywords = v1_categorize(parsed)
                search_terms = v1_generate_search_terms(keywords) or [question]
            except Exception as exc:  # noqa: BLE001
                logger.error("qid=%s extraction failed: %s", qid, exc)
                failures.append({"qid": qid, "error": f"{type(exc).__name__}: {exc}"})

            entry = {
                "question_id": qid,
                "query": question,
                "keywords": keywords,
                "search_terms": search_terms,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            elapsed = time.time() - start
            logger.info("[%d/%d] qid=%s kor=%d eng=%d terms=%d elapsed=%.1fs",
                        idx, len(rows), qid,
                        len(keywords["korean"]),
                        len(keywords["english"]),
                        len(search_terms), elapsed)

    summary = {
        "total": len(rows),
        "written": written,
        "failures": failures,
        "model": args.codex_model,
        "prompt": "v1_legacy (single call, simple-OR search_terms)",
        "elapsed_sec": time.time() - start,
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d entries to %s (failures=%d)", written, out_path, len(failures))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
