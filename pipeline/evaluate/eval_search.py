from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_evaluate_search():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.evaluate.eval_search import evaluate_search

    return evaluate_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1 통합 검색 평가")
    parser.add_argument("--results", required=True, help="검색 결과 JSON 파일")
    parser.add_argument("--ground_truth", required=True, help="정답 JSON 파일")
    parser.add_argument(
        "--source",
        required=True,
        choices=["scienceon", "miracl_ko", "miracl_en"],
        help="평가 소스",
    )
    parser.add_argument("--output", default=None, help="평가 결과 JSON 저장 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_search = _load_evaluate_search()
    report = evaluate_search(args.results, args.ground_truth, args.source)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
