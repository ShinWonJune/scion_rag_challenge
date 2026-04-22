from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step4: dense retrieval")
    parser.add_argument("--encoder", required=True, help="Query encoder config JSON")
    parser.add_argument("--questions", required=True, help="Questions JSONL")
    parser.add_argument("--schema", required=True, help="VectorDB schema JSON")
    parser.add_argument("--vectordb", default=None, help="VectorDB CSV path (optional)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k retrieval size")
    parser.add_argument("--output-root", default=None, help="Retrieval output directory (legacy option)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Retrieval output directory (preferred). If set, this value is used over --output-root.",
    )
    parser.add_argument("--output-subdir", default=None, help="Optional retrieval subdirectory name")
    parser.add_argument("--device", default="auto", help="Encoder device")
    return parser.parse_args()


def run_step4(
    encoder: str,
    questions: str,
    schema: str,
    vectordb: str | None,
    top_k: int,
    output_dir: str,
    device: str = "auto",
    output_subdir: str | None = None,
) -> str:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in root.iterdir() if p.is_dir()}
    cmd = [
        sys.executable,
        "-m",
        "src.retrieval_system.main",
        "--config_json",
        encoder,
        "--questions_jsonl",
        questions,
        "--schema_json",
        schema,
        "--top_k",
        str(top_k),
        "--output_root",
        output_dir,
        "--device",
        device,
    ]
    if output_subdir is not None:
        cmd.extend(["--output_subdir", output_subdir])
    if vectordb:
        cmd.extend(["--vectordb_csv", vectordb])

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, check=True, env=env)
    if output_subdir == "":
        return output_dir

    after_dirs = [p for p in root.iterdir() if p.is_dir() and p.name not in before]
    if after_dirs:
        after_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(after_dirs[0])
    all_dirs = [p for p in root.iterdir() if p.is_dir()]
    if all_dirs:
        all_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(all_dirs[0])
    return output_dir


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.output_root or "outputs/retrieval"
    retrieval_path = run_step4(
        encoder=args.encoder,
        questions=args.questions,
        schema=args.schema,
        vectordb=args.vectordb,
        top_k=args.top_k,
        output_dir=out_dir,
        device=args.device,
        output_subdir=args.output_subdir,
    )
    print(retrieval_path)


if __name__ == "__main__":
    main()
