from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.search_pipeline.core.extractor_factory import create_keyword_extractor
    from src.search.factories.search_client_factory import create_search_client
except ImportError:
    from search_pipeline.core.extractor_factory import create_keyword_extractor
    from src.search.factories.search_client_factory import create_search_client

try:
    from src.llm_client.llm_factory import create_llm_client
except ImportError:
    def create_llm_client(backend: str, config: dict) -> object:
        if backend not in {"gemini", "chatgpt", "vllm"}:
            raise ValueError(f"Unsupported llm backend: {backend}")
        return {"backend": backend, "config": config}

try:
    from pipeline.step1_search import run_step1
    from pipeline.step2_decompose import run_step2
    from pipeline.step3_build_vectordb import run_step3
    from pipeline.step4_retrieve import run_step4
    from pipeline.step5_generate import run_step5
except ImportError:
    from step1_search import run_step1
    from step2_decompose import run_step2
    from step3_build_vectordb import run_step3
    from step4_retrieve import run_step4
    from step5_generate import run_step5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified 5-step RAG pipeline entrypoint")
    parser.add_argument("--questions", required=True, help="Questions file (.jsonl or .csv)")
    parser.add_argument("--encoder", required=True, help="Query encoder config JSON")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "chatgpt", "vllm"])
    parser.add_argument("--sources", default="scienceon", help="Single source: scienceon|pubmed|wikipedia")
    parser.add_argument("--extractor", default="gemini", help="Keyword extractor backend")
    parser.add_argument("--keyword-lang", choices=["all", "korean", "english"], default="all")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM endpoint URL")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-20b", help="vLLM model name")
    parser.add_argument("--scienceon-credentials", default="configs/credentials/scienceon_api_credentials.json")
    parser.add_argument("--scienceon-max-pages", type=int, default=5)
    parser.add_argument("--pubmed-credentials", default="configs/credentials/pubmed_api_credentials.json")
    parser.add_argument("--schema", default="configs/csv_schema/test_2.json", help="VectorDB schema JSON")
    parser.add_argument("--output", default=None, help="Output root directory")
    parser.add_argument("--decompose", action="store_true", help="Run step2 decomposition")
    parser.add_argument("--decompose-model", default="gemini-2.5-flash")
    parser.add_argument("--target-documents", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-rank", type=int, default=1)
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


def _sources_to_list(sources: str) -> list[str]:
    return [s.strip().lower() for s in sources.split(",") if s.strip()]


def _to_wiki_lang(keyword_lang: str) -> str:
    return "ko" if keyword_lang == "korean" else "en"


def build_components(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    llm_client = create_llm_client(args.llm, {"dry_run": False})
    keyword_extractor = create_keyword_extractor(
        args.extractor,
        {
            "language": args.keyword_lang,
            "vllm_base_url": args.vllm_url,
            "vllm_model": args.vllm_model,
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
                "pubmed_credentials_path": args.pubmed_credentials,
            },
        )
    return llm_client, keyword_extractor, search_clients


def main() -> None:
    args = parse_args()
    llm_client, keyword_extractor, search_clients = build_components(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output or f"outputs/run_{timestamp}")
    search_dir = output_root / "search"
    decompose_dir = output_root / "decompose"
    retrieval_dir = output_root / "retrieval"
    final_dir = output_root / "final"
    output_root.mkdir(parents=True, exist_ok=True)

    _, search_docs = run_step1(
        questions_path=args.questions,
        output_dir=str(search_dir),
        keyword_extractor=keyword_extractor,
        search_clients=search_clients,
        target_documents=args.target_documents,
        use_timestamp_subdir=False,
    )
    if _count_jsonl_lines(search_docs) == 0:
        raise RuntimeError(
            f"Step1 produced no documents: {search_docs}. "
            "Check API credentials/network/source settings before continuing."
        )

    retrieval_questions = args.questions
    if args.decompose:
        retrieval_questions = run_step2(
            input_path=args.questions,
            output_path=str(decompose_dir / "singlehop_decompose.jsonl"),
            model=args.decompose_model,
            mode="decompose",
            use_timestamp_subdir=False,
        )

    run_step3(args.encoder, search_docs, args.schema)
    vectordb_csv = read_vectordb_from_encoder(args.encoder)

    retrieval_output_dir = run_step4(
        encoder=args.encoder,
        questions=retrieval_questions,
        schema=args.schema,
        vectordb=vectordb_csv,
        top_k=args.top_k,
        output_dir=str(retrieval_dir),
        output_subdir="",
    )

    run_step5(
        input_dir=str(retrieval_output_dir),
        output_dir=str(final_dir),
        max_rank=args.max_rank,
        llm="vllm" if args.llm == "vllm" else "gemini",
        llm_client=llm_client,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        use_timestamp_subdir=False,
    )

    print(f"Pipeline completed: {output_root}")


if __name__ == "__main__":
    main()
