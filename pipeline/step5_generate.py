from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step5: answer generation")
    parser.add_argument("--input-dir", required=True, help="Retrieval JSON directory")
    parser.add_argument("--output-dir", default="outputs/final", help="Final answer directory")
    parser.add_argument("--max-rank", type=int, default=1, help="Max rank cutoff from retrieval hits")
    parser.add_argument("--llm", choices=["gemini", "vllm"], default="gemini")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM endpoint")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-20b", help="vLLM model name")
    parser.add_argument("--scifact", action="store_true", help="Use SciFact prompt")
    return parser.parse_args()


def run_step5(
    input_dir: str,
    output_dir: str,
    max_rank: int,
    llm: str = "gemini",
    llm_client: object | None = None,
    vllm_url: str = "http://localhost:8000/v1",
    vllm_model: str = "openai/gpt-oss-20b",
    scifact: bool = False,
    use_timestamp_subdir: bool = True,
) -> None:
    target_dir = Path(output_dir)
    if use_timestamp_subdir:
        target_dir = target_dir / datetime.now().strftime("%y%m%d_%H%M%S")
    target_dir.mkdir(parents=True, exist_ok=True)

    if llm_client is not None and hasattr(llm_client, "backend"):
        llm = getattr(llm_client, "backend", llm)
    cmd = [
        sys.executable,
        "src/preprocess_and_generate_answer.py",
        "--input_dir",
        input_dir,
        "--max_rank",
        str(max_rank),
        "--output_dir",
        str(target_dir),
    ]
    if llm == "vllm":
        cmd.extend(["--use-vllm", "--vllm-url", vllm_url, "--vllm-model", vllm_model])
    if scifact:
        cmd.append("--scifact")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    run_step5(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_rank=args.max_rank,
        llm=args.llm,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        scifact=args.scifact,
    )


if __name__ == "__main__":
    main()
