#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieve_singlehop_contexts.py

입력(JSONL; decompose/chain/rewrite 결과)에서 single-hop 질문을 꺼내
각 질문마다 VectorDB에서 top-5 문서를 검색해 context로 붙여 JSONL로 저장합니다.

- 입력: singlehop_decompose.jsonl (예: {"single_hop_questions": [...]} 형태)
- 출력: enriched_singlehop.jsonl (각 질문별 top-5 문서 포함)

사용 예시
--------
python retrieve_singlehop_contexts.py \
  --input /workspace/src/data/output/singlehop_decompose.jsonl \
  --output /workspace/src/data/output/enriched_singlehop.jsonl \
  --config ../configs/query_encoder/config_1.json \
  --topk 5 \
  --limit 1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# -------------------------------
# VectorDB / Retriever (사용자 예시 기반)
# -------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vdb(output_file: str) -> pd.DataFrame:
    """CSV 또는 JSONL에서 VectorDB 불러오기. embedding은 np.array로."""
    if output_file.endswith(".csv"):
        df = pd.read_csv(output_file, encoding="utf-8")
        df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)))
    elif output_file.endswith(".jsonl"):
        import jsonlines

        records = []
        with jsonlines.open(output_file, "r") as reader:
            for row in reader:
                row["embedding"] = np.array(row["embedding"])
                records.append(row)
        df = pd.DataFrame(records)
    else:
        raise ValueError("Unsupported format. Use CSV or JSONL.")
    return df


def build_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> np.ndarray:
    # cosine_similarity(query, mat)
    # = (q·x) / (||q||*||x||) ; sklearn 없이 직접 계산
    eps = 1e-12
    q = query_vec / (np.linalg.norm(query_vec) + eps)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + eps)
    sims = (m @ q.reshape(-1, 1)).ravel()
    idx = np.argsort(sims)[::-1][:k]
    return idx, sims[idx]


def retrieve_topk(query: str, config_path: str, top_k: int = 5) -> pd.DataFrame:
    # 1) 설정 불러오기
    config = load_config(config_path)
    output_file = config.get("output_file")
    if not output_file or not os.path.exists(output_file):
        raise FileNotFoundError("❌ VectorDB not found. 먼저 main.py로 생성하세요.")

    # 2) DB 로드
    df = load_vdb(output_file)

    # 3) 모델 로드 & 쿼리 임베딩
    model_name = config["model_name"]
    model = build_encoder(model_name)
    query_emb = model.encode([query], convert_to_numpy=True)[0]  # (D,)

    # 4) 유사도 Top-K
    embeddings = np.stack(df["embedding"].to_numpy())  # (N, D)
    top_idx, top_sims = cosine_topk(query_emb, embeddings, k=top_k)

    # 5) 결과 정리
    out = df.iloc[top_idx].copy().reset_index(drop=True)
    out.insert(0, "similarity", top_sims)  # 맨 앞 열로 similarity 추가
    # text/id 필드는 VDB 스키마에 맞춰 조정
    cols = [c for c in ["similarity", "id", "text", "title", "url"] if c in out.columns]
    if "similarity" not in cols:
        cols = ["similarity"] + cols
    return out[cols]


# -------------------------------
# JSONL IO
# -------------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="\n") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------------
# Main
# -------------------------------


def extract_singlehop_questions(rec: Dict[str, Any]) -> List[str]:
    """
    decompose: rec["single_hop_questions"] -> list
    rewrite:   rec["single_hop_question"]  -> list of 1
    chain:     rec["steps"] -> [{"index": i, "question": "..."}]
    """
    if "single_hop_questions" in rec and isinstance(rec["single_hop_questions"], list):
        return [str(q) for q in rec["single_hop_questions"] if q]
    if "single_hop_question" in rec and isinstance(rec["single_hop_question"], str):
        return [rec["single_hop_question"]]
    if "steps" in rec and isinstance(rec["steps"], list):
        qs = []
        for s in rec["steps"]:
            q = s.get("question")
            if q:
                qs.append(str(q))
        return qs
    # fallback: 없으면 빈 리스트
    return []


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    """JSON 배열로 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="single-hop 질문별 top-k context retrieval"
    )
    ap.add_argument("--input", required=True, help="singlehop_* JSONL 경로")
    ap.add_argument("--output", required=True, help="결과 JSONL 경로")
    ap.add_argument("--config", default="../configs/query_encoder/config_1.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--limit", type=int, default=None, help="처리할 레코드 수 제한(예시용)"
    )
    args = ap.parse_args()

    # 입력 파일 fallback: /workspace 경로가 없으면 /mnt/data 시도
    in_path = args.input
    if not os.path.exists(in_path) and in_path.startswith("/workspace/"):
        alt = in_path.replace("/workspace/src/data/output", "/mnt/data")
        if os.path.exists(alt):
            in_path = alt

    records = read_jsonl(in_path)
    if args.limit:
        records = records[: args.limit]

    enriched: List[Dict[str, Any]] = []
    for rec in records:
        single_hops = extract_singlehop_questions(rec)
        retr_all = []
        for q in single_hops:
            try:
                df = retrieve_topk(q, config_path=args.config, top_k=args.topk)
                docs = df.to_dict(orient="records")
            except Exception as e:
                docs = [{"error": str(e)}]
            retr_all.append({"question": q, "docs": docs})
        hope = {
            "id": rec.get("id"),
            "mode": rec.get("mode"),
            "single_hop_questions": single_hops,
            "retrievals": retr_all,  # 각 질문별 top-k 문서 리스트
        }
        enriched.append(
            {
                "id": rec.get("id"),
                "mode": rec.get("mode"),
                "single_hop_questions": single_hops,
                "retrievals": retr_all,  # 각 질문별 top-k 문서 리스트
            }
        )
    write_json(args.output, enriched)
    # 예시 1개만 화면에 출력
    if enriched:
        print("=== SAMPLE (first record) ===")
        print(json.dumps(enriched[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
