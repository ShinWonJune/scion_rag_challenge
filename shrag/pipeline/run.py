from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from shrag.pipeline.steps.step1_search import run_step1
from shrag.pipeline.steps.step2_decompose import run_step2
from shrag.pipeline.steps.step3_build_vectordb import run_step3
from shrag.pipeline.steps.step4_retrieve import run_step4
from shrag.pipeline.steps.step5_generate import run_step5
from shrag.search.factories.search_client_factory import create_search_client
from shrag.search.extractors.extractor_factory import create_keyword_extractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified 5-step RAG pipeline entrypoint")
    parser.add_argument("--questions", required=True, help="Questions file (.jsonl or .csv)")
    parser.add_argument("--encoder", required=True, help="Query encoder config JSON")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "chatgpt", "vllm"])
    parser.add_argument("--sources", default="scienceon", help="Single source: scienceon|pubmed|wikipedia")
    parser.add_argument("--extractor", default="gemini", help="Keyword extractor backend")
    parser.add_argument("--chatgpt-model", default=None, help="Deprecated alias for --extractor-model.")
    parser.add_argument("--extractor-model", default="gpt-4o-mini", help="Model name for --extractor chatgpt")
    parser.add_argument("--llm-model", default="gpt-5.4", help="Model name for --llm chatgpt")
    parser.add_argument(
        "--extractor-temperature",
        default="0",
        help="Temperature for ChatGPT keyword extraction. Use 'none' to omit the parameter.",
    )
    parser.add_argument("--keyword-lang", choices=["all", "korean", "english"], default="all")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM endpoint URL")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-20b", help="vLLM model name")
    parser.add_argument("--openai-base-url", default=None, help="Optional OpenAI-compatible base URL for --llm chatgpt")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning effort for OpenAI answer generation",
    )
    parser.add_argument("--max-answer-tokens", type=int, default=4000, help="Maximum answer output tokens, including reasoning tokens for OpenAI Responses")
    parser.add_argument("--scienceon-credentials", default="configs/credentials/scienceon_api_credentials.json")
    parser.add_argument("--scienceon-max-pages", type=int, default=5)
    parser.add_argument("--scienceon-max-concurrency", type=int, default=2)
    parser.add_argument("--scienceon-min-interval-sec", type=float, default=0.5)
    parser.add_argument("--scienceon-fixed-concurrency", action="store_true")
    parser.add_argument("--scienceon-max-retries", type=int, default=5)
    parser.add_argument("--scienceon-retry-base-sleep-sec", type=float, default=2.0)
    parser.add_argument("--scienceon-retry-max-sleep-sec", type=float, default=60.0)
    parser.add_argument("--cache-root", default="outputs/_shared_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--frozen-queries", default=None, help="Optional JSONL with precomputed keywords/search terms")
    parser.add_argument("--pubmed-credentials", default="configs/credentials/pubmed_api_credentials.json")
    parser.add_argument("--schema", default="configs/csv_schema/test_2.json", help="VectorDB schema JSON")
    parser.add_argument("--output", default=None, help="Output root directory")
    parser.add_argument("--decompose", action="store_true", help="Run step2 decomposition")
    parser.add_argument("--decompose-model", default="gemini-2.5-flash")
    parser.add_argument("--target-documents", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-rank", type=int, default=5)
    return parser.parse_args()


def read_vectordb_from_encoder(encoder_config: str) -> str:
    with open(encoder_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    output_file = config.get("output_file")
    if not output_file:
        raise ValueError(f"output_file not found in config: {encoder_config}")
    return output_file


def _count_jsonl_lines(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


def _load_questions_for_jsonl(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                question = item.get("question") or item.get("query") or item.get("text")
                if question:
                    rows.append({"id": str(item.get("id", idx)), "question": str(question)})
        return rows

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported questions file format: {path}")

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        lower_map = {name.lower(): name for name in reader.fieldnames}
        question_col = None
        for candidate in ["question", "query", "text", "translated_question"]:
            if candidate in lower_map:
                question_col = lower_map[candidate]
                break
        if not question_col:
            raise ValueError(f"Question column not found in CSV: {path}")
        id_col = lower_map.get("id") or lower_map.get("question_id")
        for idx, item in enumerate(reader):
            question = (item.get(question_col) or "").strip()
            if not question:
                continue
            qid = (item.get(id_col) or str(idx)).strip() if id_col else str(idx)
            rows.append({"id": qid, "question": question})
    return rows


def _ensure_retrieval_questions_jsonl(questions_path: str, output_path: Path) -> str:
    source = Path(questions_path)
    if source.suffix.lower() == ".jsonl":
        return str(source)

    rows = _load_questions_for_jsonl(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(output_path)


def _sources_to_list(sources: str) -> list[str]:
    return [s.strip().lower() for s in sources.split(",") if s.strip()]


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return float(value)


def _to_wiki_lang(keyword_lang: str) -> str:
    return "ko" if keyword_lang == "korean" else "en"


def build_components(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    keyword_extractor = create_keyword_extractor(
        args.extractor,
        {
            "language": args.keyword_lang,
            "vllm_base_url": args.vllm_url,
            "vllm_model": args.vllm_model,
            "model": args.chatgpt_model or args.extractor_model,
            "temperature": _parse_optional_float(args.extractor_temperature),
        },
    )
    search_clients: dict[str, Any] = {}
    for source in _sources_to_list(args.sources):
        search_clients[source] = create_search_client(
            source,
            {
                "lang": _to_wiki_lang(args.keyword_lang),
                "scienceon_credentials_path": args.scienceon_credentials,
                "scienceon_max_pages": args.scienceon_max_pages,
                "scienceon_max_concurrency": args.scienceon_max_concurrency,
                "scienceon_min_interval_sec": args.scienceon_min_interval_sec,
                "scienceon_fixed_concurrency": args.scienceon_fixed_concurrency,
                "scienceon_max_retries": args.scienceon_max_retries,
                "scienceon_retry_base_sleep_sec": args.scienceon_retry_base_sleep_sec,
                "scienceon_retry_max_sleep_sec": args.scienceon_retry_max_sleep_sec,
                "cache_root": args.cache_root,
                "disable_cache": args.no_cache,
                "pubmed_credentials_path": args.pubmed_credentials,
            },
        )
    return keyword_extractor, search_clients


def _file_sha1(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _credential_fingerprint(path: str) -> str:
    return f"sha1:{_file_sha1(Path(path))[:12]}"


def _git_sha_and_dirty() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return "unknown", False


def _write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    keyword_extractor, search_clients = build_components(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output or f"outputs/run_{timestamp}")
    search_dir = output_root / "search"
    decompose_dir = output_root / "decompose"
    retrieval_dir = output_root / "retrieval"
    final_dir = output_root / "final"
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    git_sha, dirty = _git_sha_and_dirty()
    manifest: dict[str, Any] = {
        "run_id": output_root.name,
        "git_sha": git_sha,
        "git_dirty": dirty,
        "requirements_sha1": _file_sha1(Path("requirements.txt")),
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "cli_args": vars(args),
        "vllm": {"url": args.vllm_url, "model": args.vllm_model, "version": None},
        "openai": {
            "model": args.llm_model if args.llm == "chatgpt" else None,
            "base_url": args.openai_base_url,
            "reasoning_effort": args.openai_reasoning_effort,
            "max_answer_tokens": args.max_answer_tokens,
        } if args.llm == "chatgpt" else None,
        "credentials_fingerprint": {
            "scienceon": _credential_fingerprint(args.scienceon_credentials),
            "pubmed": _credential_fingerprint(args.pubmed_credentials),
        },
        "cache": {
            "enabled": not args.no_cache,
            "root": args.cache_root,
            "hit_count": 0,
            "miss_count": 0,
            "write_count": 0,
        },
        "runtime": {},
        "steps_completed": [],
    }
    _write_run_manifest(output_root / "run_manifest.json", manifest)

    step1_started = time.perf_counter()
    search_meta, search_docs = run_step1(
        questions_path=args.questions,
        output_dir=str(search_dir),
        keyword_extractor=keyword_extractor,
        search_clients=search_clients,
        target_documents=args.target_documents,
        use_timestamp_subdir=False,
        frozen_queries_path=args.frozen_queries,
    )
    manifest["runtime"]["step1_search_sec"] = round(time.perf_counter() - step1_started, 4)
    manifest["steps_completed"].append("step1")
    try:
        search_payload = json.loads(Path(search_meta).read_text(encoding="utf-8"))
        request_stats = search_payload.get("request_stats", {})
        cache_stats = request_stats.get("cache", {})
        manifest["cache"].update(
            {
                "hit_count": cache_stats.get("hit_count", 0),
                "miss_count": cache_stats.get("miss_count", 0),
                "write_count": cache_stats.get("write_count", 0),
            }
        )
        manifest["search_request_stats"] = request_stats
    except Exception:
        pass
    _write_run_manifest(output_root / "run_manifest.json", manifest)
    if _count_jsonl_lines(search_docs) == 0:
        raise RuntimeError(
            f"Step1 produced no documents: {search_docs}. "
            "Check API credentials/network/source settings before continuing."
        )

    retrieval_questions = _ensure_retrieval_questions_jsonl(
        args.questions,
        output_root / "questions.jsonl",
    )
    if args.decompose:
        step2_started = time.perf_counter()
        retrieval_questions = run_step2(
            input_path=retrieval_questions,
            output_path=str(decompose_dir / "singlehop_decompose.jsonl"),
            model=args.decompose_model,
            mode="decompose",
            use_timestamp_subdir=False,
        )
        manifest["runtime"]["step2_decompose_sec"] = round(time.perf_counter() - step2_started, 4)
        manifest["steps_completed"].append("step2")
        _write_run_manifest(output_root / "run_manifest.json", manifest)

    step3_started = time.perf_counter()
    run_step3(args.encoder, search_docs, args.schema)
    manifest["runtime"]["step3_build_vectordb_sec"] = round(time.perf_counter() - step3_started, 4)
    manifest["steps_completed"].append("step3")
    _write_run_manifest(output_root / "run_manifest.json", manifest)
    vectordb_csv = read_vectordb_from_encoder(args.encoder)

    step4_started = time.perf_counter()
    retrieval_output_dir = run_step4(
        encoder=args.encoder,
        questions=retrieval_questions,
        schema=args.schema,
        vectordb=vectordb_csv,
        top_k=args.top_k,
        output_dir=str(retrieval_dir),
        output_subdir="",
    )
    manifest["runtime"]["step4_retrieve_sec"] = round(time.perf_counter() - step4_started, 4)
    manifest["steps_completed"].append("step4")
    _write_run_manifest(output_root / "run_manifest.json", manifest)

    step5_started = time.perf_counter()
    run_step5(
        input_dir=str(retrieval_output_dir),
        output_dir=str(final_dir),
        max_rank=args.max_rank,
        llm=args.llm,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        openai_model=args.llm_model,
        openai_base_url=args.openai_base_url,
        openai_reasoning_effort=args.openai_reasoning_effort,
        max_answer_tokens=args.max_answer_tokens,
        use_timestamp_subdir=False,
    )
    manifest["runtime"]["step5_generate_sec"] = round(time.perf_counter() - step5_started, 4)
    manifest["steps_completed"].append("step5")
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["runtime"]["total_sec"] = round(
        sum(value for value in manifest["runtime"].values() if isinstance(value, (int, float))),
        4,
    )
    _write_run_manifest(output_root / "run_manifest.json", manifest)

    print(f"Pipeline completed: {output_root}")


if __name__ == "__main__":
    main()
