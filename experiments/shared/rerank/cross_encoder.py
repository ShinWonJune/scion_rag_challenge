from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        pairs = [
            [query, f"{cand.get('title', '')}\n{cand.get('abstract', '')}".strip()]
            for cand in candidates
        ]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        reranked = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            reranked.append(item)
        reranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        for idx, item in enumerate(reranked[:top_k], start=1):
            item["rank"] = idx
        return reranked[:top_k]


def _iter_json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.json") if path.name != "predictions.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank retrieval JSON files with a cross-encoder")
    parser.add_argument("--input", required=True, help="Input retrieval directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    reranker = CrossEncoderReranker(model_name=args.model)
    for path in _iter_json_files(input_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("retrieval_results", []):
            result["hits"] = reranker.rerank(result.get("query", ""), result.get("hits", []), top_k=args.top_k)
        (output_dir / path.name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
