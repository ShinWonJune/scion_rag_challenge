"""Codex CLI 기반 키워드 추출 → frozen-queries JSONL 생성기.

본 스크립트는 `pipeline.step1_search --frozen-queries`가 소비하는 JSONL을 생성한다.
키워드 추출 로직은 `src.search_pipeline.core.codex_keyword_extractor.CodexKeywordExtractor`
(공유 base `LLMKeywordExtractor`)에 위임하므로, 사용되는 한국어/영어 프롬프트와
search_terms 빌드 로직이 ChatGPT/Gemini/vLLM extractor와 완전히 동일하다.

CLI 예:
    python -m experiments.target_sweep.extract_keywords_codex \
      --questions data/test.csv --gold data/gold/scienceon_gold.json \
      --output experiments/outputs/frozen/queries_codex_gold41.jsonl
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

from src.search_pipeline.core.extractor_factory import create_keyword_extractor

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
    parser = argparse.ArgumentParser(description="Codex CLI keyword extractor → frozen-queries JSONL")
    parser.add_argument("--questions", required=True, help="Questions CSV or JSONL")
    parser.add_argument("--gold", default=None, help="Optional gold JSON (filter to its qids only)")
    parser.add_argument("--output", required=True, help="Output frozen-queries JSONL path")
    parser.add_argument("--codex-model", default="gpt-5.4")
    parser.add_argument("--codex-timeout-sec", type=int, default=300)
    parser.add_argument("--language", choices=["all", "korean", "english"], default="all")
    parser.add_argument("--max-tokens", type=int, default=1500,
                        help="Max output tokens per codex call (per language).")
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
    logger.info("Loaded %d questions", len(rows))

    extractor = create_keyword_extractor(
        "codex",
        {
            "model": args.codex_model,
            "timeout_sec": args.codex_timeout_sec,
            "language": args.language,
            "max_tokens": args.max_tokens,
        },
    )
    logger.info("Extractor: %s", extractor.get_extractor_info())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failures: list[dict[str, Any]] = []
    start = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows, 1):
            qid = row["question_id"]
            question = row["query"]
            try:
                keywords = extractor.extract_keywords(question)
                search_terms = extractor.generate_search_terms(keywords)
            except Exception as exc:  # noqa: BLE001
                logger.error("qid=%s extraction failed: %s", qid, exc)
                failures.append({"qid": qid, "error": f"{type(exc).__name__}: {exc}"})
                keywords = {"korean": [], "english": []}
                search_terms = [question]

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
                        len(keywords.get("korean", [])),
                        len(keywords.get("english", [])),
                        len(search_terms), elapsed)

    summary = {
        "total": len(rows),
        "written": written,
        "failures": failures,
        "model": args.codex_model,
        "language": args.language,
        "elapsed_sec": time.time() - start,
        "extractor_info": extractor.get_extractor_info(),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d entries to %s (failures=%d)", written, out_path, len(failures))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
