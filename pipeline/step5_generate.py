from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step5: answer generation")
    parser.add_argument("--input-dir", required=True, help="Retrieval JSON directory")
    parser.add_argument("--output-dir", default="outputs/final", help="Final answer directory")
    parser.add_argument("--max-rank", type=int, default=5, help="Max rank cutoff from retrieval hits (top-N docs fed to the answerer)")
    parser.add_argument("--llm", choices=["gemini", "vllm", "chatgpt"], default="gemini")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM endpoint")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-20b", help="vLLM model name")
    parser.add_argument("--openai-model", default="gpt-5.4", help="OpenAI model name for --llm chatgpt")
    parser.add_argument("--openai-base-url", default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--max-answer-tokens", type=int, default=4000, help="Maximum answer output tokens, including reasoning tokens for OpenAI Responses.")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning effort for OpenAI reasoning models",
    )
    parser.add_argument("--scifact", action="store_true", help="Use SciFact prompt")
    return parser.parse_args()


def run_step5(
    input_dir: str,
    output_dir: str,
    max_rank: int,
    llm: str = "gemini",
    vllm_url: str = "http://localhost:8000/v1",
    vllm_model: str = "openai/gpt-oss-20b",
    openai_model: str = "gpt-5.4",
    openai_base_url: str | None = None,
    openai_reasoning_effort: str | None = None,
    max_answer_tokens: int = 4000,
    scifact: bool = False,
    use_timestamp_subdir: bool = True,
) -> None:
    target_dir = Path(output_dir)
    if use_timestamp_subdir:
        target_dir = target_dir / datetime.now().strftime("%y%m%d_%H%M%S")
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.preprocess_and_generate_answer",
        "--input_dir",
        input_dir,
        "--max_rank",
        str(max_rank),
        "--output_dir",
        str(target_dir),
        "--max_answer_tokens",
        str(max_answer_tokens),
    ]
    if llm == "vllm":
        cmd.extend(["--use-vllm", "--vllm-url", vllm_url, "--vllm-model", vllm_model])
    elif llm == "chatgpt":
        cmd.extend(["--use-openai", "--openai-model", openai_model])
        if openai_base_url:
            cmd.extend(["--openai-base-url", openai_base_url])
        if openai_reasoning_effort:
            cmd.extend(["--openai-reasoning-effort", openai_reasoning_effort])
    if scifact:
        cmd.append("--scifact")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    run_step5(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_rank=args.max_rank,
        llm=args.llm,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        openai_reasoning_effort=args.openai_reasoning_effort,
        max_answer_tokens=args.max_answer_tokens,
        scifact=args.scifact,
    )


if __name__ == "__main__":
    main()
