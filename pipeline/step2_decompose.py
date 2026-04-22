from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2: multi-hop decomposition")
    parser.add_argument("--input", required=True, help="Input questions JSONL")
    parser.add_argument(
        "--output",
        default="outputs/decompose",
        help="Output file path or output directory (legacy option)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (preferred). If set, this value is used over --output.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model for decomposition")
    parser.add_argument("--mode", default="decompose", choices=["decompose", "chain", "rewrite"])
    return parser.parse_args()


def run_step2(
    input_path: str,
    output_path: str,
    model: str,
    mode: str = "decompose",
    use_timestamp_subdir: bool = True,
) -> str:
    out_path = Path(output_path)
    if out_path.suffix.lower() != ".jsonl":
        out_dir = out_path
        if use_timestamp_subdir:
            out_dir = out_dir / datetime.now().strftime("%y%m%d_%H%M%S")
        out_path = out_dir / "singlehop_decompose.jsonl"
    elif use_timestamp_subdir:
        out_path = out_path.parent / datetime.now().strftime("%y%m%d_%H%M%S") / out_path.name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.multi_hop_to_single_hop",
        "--input",
        input_path,
        "--output",
        str(out_path),
        "--question_field",
        "question",
        "--context_field",
        "context",
        "--mode",
        mode,
        "--model",
        model,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, check=True, env=env)
    return str(out_path)


def main() -> None:
    args = parse_args()
    out = args.output_dir or args.output
    print(run_step2(args.input, out, args.model, args.mode))


if __name__ == "__main__":
    main()
