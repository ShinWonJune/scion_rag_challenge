"""1-question (or N-sample) timing harness.

Measures step1 + step3 + step4 latency for representative queries instead of 41Q
aggregate. Useful for "live demo" feasibility (t≤5min) decisions and per-query
cost without batch optimization noise.

Important: largest cell first to prime cache. Subsequent cells re-use embeddings
from outputs/_embedding_cache.db (configured via cache_db_path in encoder config).

Usage:
  python -m experiments.exp1_kmn.timing_1q --qids 1 --k 10 --n 1 --m 30 \
      --encoder configs/query_encoder/config_gte-multilingual-base.json \
      --output experiments/outputs/_timing/cell_x.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path("/mnt/c/Users/wonjune/workspace/RAG/scion_rag_challenge")
PYTHON = "/home/wonjune/miniconda3/envs/shrag/bin/python"


def filter_questions(src_jsonl: Path, qids: list[str], dst_jsonl: Path) -> int:
    qid_set = {str(q) for q in qids}
    n = 0
    with src_jsonl.open() as f, dst_jsonl.open("w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("id") or row.get("question_id")) in qid_set:
                g.write(line + "\n")
                n += 1
    return n


def filter_frozen(src_jsonl: Path, qids: list[str], dst_jsonl: Path) -> int:
    qid_set = {str(q) for q in qids}
    n = 0
    with src_jsonl.open() as f, dst_jsonl.open("w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("question_id")) in qid_set:
                g.write(line + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", nargs="+", default=["1"], help="One or more question_ids to time")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--frozen-full", default=str(PROJECT_ROOT / "experiments/outputs/exp1_kmn/_frozen/queries_v3_chatgpt_gold41.jsonl"))
    ap.add_argument("--questions-full", default=str(PROJECT_ROOT / "experiments/outputs/exp1_kmn/questions_gold41.jsonl"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default=None, help="Optional label for the cell")
    args = ap.parse_args()

    qids = list(args.qids)
    label = args.label or f"k{args.k}_n{args.n}_m{args.m}_qids{'-'.join(qids)}"
    work = Path(f"/tmp/timing_1q/{label}_{int(time.time())}")
    work.mkdir(parents=True, exist_ok=True)

    questions_path = work / "questions.jsonl"
    n_q = filter_questions(Path(args.questions_full), qids, questions_path)
    if n_q != len(qids):
        sys.exit(f"FATAL: filtered {n_q} questions (expected {len(qids)}); check qids")

    # Slice frozen-queries to top-N AND filter to qids
    frozen_full_filtered = work / "frozen_full_filtered.jsonl"
    if filter_frozen(Path(args.frozen_full), qids, frozen_full_filtered) == 0:
        sys.exit("FATAL: no frozen entries match qids")
    frozen_sliced = work / f"frozen_n{args.n}.jsonl"
    subprocess.run(
        [PYTHON, "-m", "experiments.exp1_kmn.slice_frozen_queries",
         "--input", str(frozen_full_filtered),
         "--output", str(frozen_sliced),
         "--n", str(args.n)],
        cwd=str(PROJECT_ROOT), check=True,
    )

    max_pages = max(1, args.k // 10)
    out = {
        "label": label,
        "qids": qids,
        "k": args.k, "n": args.n, "m": args.m, "max_pages": max_pages,
        "encoder": args.encoder,
        "started_at": datetime.now().isoformat(),
    }

    # Step1
    search_dir = work / "search"
    t = time.perf_counter()
    rc = subprocess.run(
        [PYTHON, "-m", "shrag.pipeline.steps.step1_search",
         "--questions", str(questions_path),
         "--sources", "scienceon",
         "--extractor", "vllm",
         "--vllm-url", os.environ.get("GPT_OSS_VLLM_URL", "http://10.38.38.40:8004/v1"),
         "--vllm-model", os.environ.get("GPT_OSS_VLLM_MODEL", "openai/gpt-oss-20b"),
         "--extractor-temperature", "0",
         "--frozen-queries", str(frozen_sliced),
         "--target-documents", str(args.m),
         "--scienceon-max-pages", str(max_pages),
         "--scienceon-max-concurrency", "2",
         "--scienceon-min-interval-sec", "0.5",
         "--output-dir", str(search_dir)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    out["step1_sec"] = round(time.perf_counter() - t, 3)
    if rc.returncode != 0:
        out["step1_error"] = rc.stderr[-2000:]
    # Find search_documents.jsonl
    docs_jsonl = next(search_dir.rglob("search_documents.jsonl"), None)
    if not docs_jsonl:
        out["error"] = "no search_documents.jsonl"
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(2)
    out["docs_count"] = sum(1 for _ in docs_jsonl.open())

    # Per-cell encoder cfg with output paths in work/
    encoder_cfg = work / "encoder_cfg.json"
    cfg = json.loads(Path(args.encoder).read_text())
    cfg["jsonl_path"] = str(docs_jsonl.resolve())
    cfg["output_dir"] = str(work.resolve())
    encoder_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    # Step3 (cache will short-circuit per-doc)
    t = time.perf_counter()
    rc = subprocess.run(
        [PYTHON, "-m", "shrag.pipeline.steps.step3_build_vectordb",
         "--encoder", str(encoder_cfg),
         "--docs", str(docs_jsonl),
         "--schema", "configs/csv_schema/test_2.json"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    out["step3_sec"] = round(time.perf_counter() - t, 3)
    out["step3_log_tail"] = rc.stdout[-1500:]
    if rc.returncode != 0:
        out["step3_error"] = rc.stderr[-2000:]

    # Find vectordb csv
    vectordb = next(work.rglob("vector_db_*.csv"), None)
    if not vectordb:
        out["error"] = "no vector_db_*.csv"
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(2)

    # Step4 retrieval
    retrieval_root = work / "retrieval"
    t = time.perf_counter()
    rc = subprocess.run(
        [PYTHON, "-m", "shrag.pipeline.steps.step4_retrieve",
         "--encoder", str(encoder_cfg),
         "--questions", str(questions_path),
         "--schema", "configs/csv_schema/test_2.json",
         "--vectordb", str(vectordb),
         "--top-k", "50",
         "--output-root", str(retrieval_root)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    out["step4_sec"] = round(time.perf_counter() - t, 3)
    if rc.returncode != 0:
        out["step4_error"] = rc.stderr[-2000:]

    out["total_sec"] = round(out.get("step1_sec", 0) + out.get("step3_sec", 0) + out.get("step4_sec", 0), 3)
    out["finished_at"] = datetime.now().isoformat()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"step1={out.get('step1_sec')}s step3={out.get('step3_sec')}s step4={out.get('step4_sec')}s total={out.get('total_sec')}s docs={out.get('docs_count')}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
