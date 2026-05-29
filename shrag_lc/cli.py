"""CLI entrypoint:  python -m shrag_lc.cli run --questions data/test.csv ...

Loads questions (.csv or .jsonl), runs the LangChain SHRAG pipeline, and writes
answers + a run manifest to ``outputs_lc/run_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from .config import PipelineConfig, Settings
from .reporting import build_manifest, write_json


def load_questions(path: str | Path, limit: int | None = None) -> list[dict]:
    """Load [{'id', 'question'}] from .csv or .jsonl (flexible column detection)."""
    path = Path(path)
    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("question") or item.get("query") or item.get("text")
                if q:
                    rows.append({"id": str(item.get("id", idx)), "question": str(q)})
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            lower = {name.lower(): name for name in (reader.fieldnames or [])}
            qcol = next(
                (lower[c] for c in ["question", "query", "text", "translated_question"] if c in lower),
                None,
            )
            if not qcol:
                raise ValueError(f"Question column not found in CSV: {path}")
            idcol = lower.get("id") or lower.get("question_id")
            for idx, item in enumerate(reader):
                q = (item.get(qcol) or "").strip()
                if not q:
                    continue
                qid = (item.get(idcol) or str(idx)).strip() if idcol else str(idx)
                rows.append({"id": qid, "question": q})
    else:
        raise ValueError(f"Unsupported questions file: {path}")
    return rows[:limit] if limit else rows


def build_config(args: argparse.Namespace) -> PipelineConfig:
    overrides = dict(
        source=args.source,
        target_documents=args.target_documents,
        dense_top_k=args.top_k,
        rerank_top_n=args.max_rank,
        use_reranker=not args.no_rerank,
        llm_backend=args.llm,
        extractor_backend=args.extractor,
        keyword_lang=args.keyword_lang,
        scienceon_max_pages=args.scienceon_max_pages,
        scienceon_max_concurrency=args.scienceon_max_concurrency,
        scienceon_min_interval_sec=args.scienceon_min_interval_sec,
        scienceon_fixed_concurrency=args.scienceon_fixed_concurrency,
        scienceon_max_retries=args.scienceon_max_retries,
        scienceon_retry_base_sleep_sec=args.scienceon_retry_base_sleep_sec,
        scienceon_retry_max_sleep_sec=args.scienceon_retry_max_sleep_sec,
        cache_root=args.cache_root,
        disable_cache=args.no_cache,
    )
    if args.encoder:
        return PipelineConfig.from_encoder_json(args.encoder, **overrides)
    return PipelineConfig(**overrides)


def cmd_run(args: argparse.Namespace) -> None:
    from .pipeline import SHRAGPipeline  # heavy imports deferred

    cfg = build_config(args)
    settings = Settings()
    questions = load_questions(args.questions, limit=args.limit)
    print(f"Loaded {len(questions)} questions; source={cfg.source}, llm={cfg.llm_backend}")

    pipeline = SHRAGPipeline(cfg, settings)
    out_dir = Path(args.output or f"outputs_lc/run_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")

    if args.corpus_first:
        results = pipeline.run_corpus_first(
            questions,
            artifact_dir=out_dir,
            index_dir=args.index_dir or (out_dir / "faiss_index"),
            reuse_index=args.reuse_index,
        )
        pipeline_mode = "corpus_first"
    else:
        results = pipeline.run(questions)
        pipeline_mode = "per_query"

    predictions_path = out_dir / "predictions.json"
    write_json(predictions_path, results)
    artifacts = {"predictions": predictions_path.name}
    if args.corpus_first:
        artifacts["corpus"] = "corpus.jsonl"
        artifacts["index_dir"] = str(args.index_dir or (out_dir / "faiss_index"))

    manifest = build_manifest(
        config=cfg,
        questions=questions,
        results=results,
        pipeline_mode=pipeline_mode,
        generated_at=generated_at,
        artifacts=artifacts,
        corpus=pipeline.last_run_context.get("corpus"),
        index=pipeline.last_run_context.get("index"),
    )
    write_json(out_dir / "manifest.json", manifest)
    print(f"Pipeline completed: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="shrag_lc — LangChain SHRAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the end-to-end pipeline")
    run.add_argument("--questions", required=True, help="Questions file (.csv or .jsonl)")
    run.add_argument("--source", default="wikipedia", choices=["scienceon", "pubmed", "wikipedia"])
    run.add_argument("--encoder", default=None, help="Optional configs/query_encoder/*.json")
    run.add_argument("--llm", default="vllm", choices=["vllm", "openai", "gemini"])
    run.add_argument("--extractor", default="vllm", choices=["vllm", "openai", "gemini"])
    run.add_argument("--keyword-lang", default="all", choices=["all", "korean", "english"])
    run.add_argument("--target-documents", type=int, default=50)
    run.add_argument("--scienceon-max-pages", type=int, default=5)
    run.add_argument("--scienceon-max-concurrency", type=int, default=2)
    run.add_argument("--scienceon-min-interval-sec", type=float, default=0.5)
    run.add_argument("--scienceon-fixed-concurrency", action="store_true")
    run.add_argument("--scienceon-max-retries", type=int, default=5)
    run.add_argument("--scienceon-retry-base-sleep-sec", type=float, default=2.0)
    run.add_argument("--scienceon-retry-max-sleep-sec", type=float, default=60.0)
    run.add_argument("--cache-root", default="outputs/_shared_cache")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--max-rank", type=int, default=3)
    run.add_argument("--no-rerank", action="store_true")
    run.add_argument("--corpus-first", action="store_true", help="Acquire all documents, then build one shared index")
    run.add_argument("--index-dir", default=None, help="Directory for a reusable FAISS index")
    run.add_argument("--reuse-index", action="store_true", help="Reuse --index-dir when metadata matches")
    run.add_argument("--limit", type=int, default=None, help="Only process the first N questions")
    run.add_argument("--output", default=None, help="Output directory")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
