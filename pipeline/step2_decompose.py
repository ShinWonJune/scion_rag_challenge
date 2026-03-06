from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2: multi-hop decomposition")
    parser.add_argument("--input", required=True, help="Input questions JSONL")
    parser.add_argument("--output", required=True, help="Output single-hop JSONL")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model for decomposition")
    parser.add_argument("--mode", default="decompose", choices=["decompose", "chain", "rewrite"])
    return parser.parse_args()


def run_step2(input_path: str, output_path: str, model: str, mode: str = "decompose") -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "src/multi_hop_to_single_hop.py",
        "--input",
        input_path,
        "--output",
        output_path,
        "--question_field",
        "question",
        "--context_field",
        "context",
        "--mode",
        mode,
        "--model",
        model,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def main() -> None:
    args = parse_args()
    print(run_step2(args.input, args.output, args.model, args.mode))


if __name__ == "__main__":
    main()
