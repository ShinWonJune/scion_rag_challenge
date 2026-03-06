from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step3: build vector DB")
    parser.add_argument("--encoder", required=True, help="Query encoder config JSON")
    parser.add_argument("--docs", required=True, help="Search documents JSONL")
    parser.add_argument(
        "--schema",
        default="configs/csv_schema/test_2.json",
        help="CSV schema JSON path",
    )
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id for embedding model")
    return parser.parse_args()


def run_step3(encoder: str, docs: str, schema: str, gpu_id: int | None = None) -> None:
    cmd = [
        sys.executable,
        "src/build_vectordb_search.py",
        "--config_path",
        encoder,
        "--data_schema",
        schema,
        "--docs_jsonl_path",
        docs,
    ]
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    run_step3(args.encoder, args.docs, args.schema, args.gpu_id)


if __name__ == "__main__":
    main()
