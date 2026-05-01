#!/usr/bin/env python3
"""
extract_questions.py (SIMPLE)

주어진 데이터셋에서 지정된 질문 컬럼(또는 자동 탐지된 컬럼)의 모든 값을 JSONL 포맷으로 올바르게 작성합니다.
각 라인은 하나의 JSON 객체이며, 반드시 한 줄에 하나의 JSON만 포함되도록 합니다.

예시
----
python extract_questions.py \
  --input /workspace/data/rag_test_data/scion.csv \
  --output /workspace/data/rag_test_data/questions.jsonl \
  --question-col Question
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

try:
    import pandas as pd
except Exception:
    pd = None

CANDIDATE_QUESTION_COLS = [
    "question",
    "Question",
    "QUESTION",
    "질문",
    "query",
    "Query",
    "text",
    "Text",
    "prompt",
    "Prompt",
    "q",
    "Q",
]


def _load_df(path: str):
    if pd is None:
        raise RuntimeError(
            "pandas가 필요합니다. `pip install pandas openpyxl` 후 다시 시도하세요."
        )
    ext = os.path.splitext(path)[1].lower()
    # utf-8-sig 로 읽어서 BOM 제거
    if ext in (".csv", ".tsv", ".txt"):
        sep = "	" if ext == ".tsv" else None
        return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext in (".jsonl", ".jsonlines"):
        return pd.read_json(path, lines=True, encoding="utf-8-sig")
    if ext == ".json":
        data = json.load(open(path, "r", encoding="utf-8-sig"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError("JSON 파일은 list 형태여야 합니다.")
    raise ValueError(f"지원하지 않는 형식: {ext}")


def _pick_question_col(df, user_col: Optional[str]) -> str:
    if user_col:
        if user_col in df.columns:
            return user_col
        raise ValueError(
            f"지정한 컬럼이 없습니다: {user_col}. 실제 컬럼: {list(df.columns)}"
        )
    for c in CANDIDATE_QUESTION_COLS:
        if c in df.columns:
            return c
    raise ValueError(
        "질문 컬럼을 찾을 수 없습니다. --question-col 로 지정하시거나, "
        f"후보 중 하나로 컬럼명을 맞춰주세요: {CANDIDATE_QUESTION_COLS} (실제 컬럼: {list(df.columns)})"
    )


def main():
    ap = argparse.ArgumentParser(description="질문 컬럼을 JSONL로 추출")
    ap.add_argument("--input", required=True, help="입력 파일 경로")
    ap.add_argument("--output", required=True, help="출력 JSONL 경로")
    ap.add_argument(
        "--question-col", default=None, help="질문 컬럼명 (없으면 자동 탐지)"
    )
    args = ap.parse_args()

    df = _load_df(args.input)
    # 컬럼명 BOM/공백 정리
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    qcol = _pick_question_col(df, args.question_col)

    out_path = args.output
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, val in enumerate(df[qcol].astype(str).fillna("")):
            # 셀 내부 BOM/제어문자 정리
            text = val.replace("﻿", "").replace("", "").strip()
            if not text or text.lower() == "nan":
                continue
            obj = {"id": f"row_{i + 1:06d}", "question": text}
            # 한 줄에 정확히 하나의 JSON만 기록 (출력 인코딩은 utf-8: BOM 없음)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} questions → {out_path}")


if __name__ == "__main__":
    main()
