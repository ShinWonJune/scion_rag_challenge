from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_evaluate_answers():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.evaluate.eval_answers import evaluate_answers

    return evaluate_answers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step5 통합 답변 평가")
    parser.add_argument("--predictions", required=True, help="예측 CSV 파일")
    parser.add_argument("--ground_truth", required=True, help="정답 CSV 파일")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["pubmedqa", "scifact", "scienceon"],
        help="평가 데이터셋",
    )
    parser.add_argument("--output", default=None, help="평가 결과 JSON 저장 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_answers = _load_evaluate_answers()
    report = evaluate_answers(args.predictions, args.ground_truth, args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
