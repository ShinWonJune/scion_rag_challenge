from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

from shrag.utils.load_jsonl_and_make_text_for_embedding import (
    load_jsonl_and_make_text_for_embedding,
)


def _dtype(name: str | None) -> Any | None:
    if not name:
        return None
    value = str(name).lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def _percentiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.int64)
    return {
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def _token_lengths(model: SentenceTransformer, texts: list[str]) -> list[int]:
    tokenizer = model.tokenizer
    lengths: list[int] = []
    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.append(len(encoded["input_ids"]))
    return lengths


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exp15. Embedding Speed Tuning Sweep",
        "",
        "Measures the embedding path on the fixed Q41 corpus with stage timers.",
        "",
        f"- corpus: `{summary['corpus']}`",
        f"- docs: `{summary['doc_count']}`",
        f"- device: `{summary['device']}`",
        f"- torch: `{summary['torch_version']}`",
        "",
        "| case | model | dim | batch | dtype | max_len | bucket | load sec | encode sec | save sec | total sec | docs/sec | status |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["cases"]:
        lines.append(
            f"| {row['case_id']}"
            f" | `{row['model_name']}`"
            f" | {row.get('embedding_dim') or '-'}"
            f" | {row.get('batch_size') or '-'}"
            f" | {row.get('torch_dtype') or 'default'}"
            f" | {row.get('max_seq_length') or '-'}"
            f" | {row.get('length_bucket', False)}"
            f" | {row.get('load_sec') or '-'}"
            f" | {row.get('encode_sec') or '-'}"
            f" | {row.get('save_sec') or '-'}"
            f" | {row.get('total_sec') or '-'}"
            f" | {row.get('docs_per_sec') or '-'}"
            f" | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `encode_precision` is intentionally left at SentenceTransformers float32 output precision unless a case explicitly sets it.",
            "- `torch_dtype` controls model load/compute dtype; it is separate from output embedding quantization.",
            "- `length_bucket=true` sorts texts by token length before encoding and restores the original order before saving.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    docs = load_jsonl_and_make_text_for_embedding(args.corpus, embedding_mode=args.embedding_mode)
    texts = [str(doc.get("embedding_text", "")) for doc in docs]
    doc_count = len(texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary: dict[str, Any] = {
        "experiment": "exp15_embedding_speed_tuning",
        "corpus": args.corpus,
        "doc_count": doc_count,
        "embedding_mode": args.embedding_mode,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cases": [],
    }

    for case in _load_cases(Path(args.cases)):
        case_id = str(case["case_id"])
        case_dir = output / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "case_id": case_id,
            "model_name": case["model_name"],
            "embedding_dim": int(case["embedding_dim"]),
            "batch_size": int(case.get("batch_size", 32)),
            "torch_dtype": case.get("torch_dtype"),
            "max_seq_length": case.get("max_seq_length"),
            "attn_implementation": case.get("attn_implementation"),
            "length_bucket": bool(case.get("length_bucket", False)),
            "status": "started",
        }
        case_start = time.perf_counter()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            model_kwargs: dict[str, Any] = {}
            dtype_value = _dtype(case.get("torch_dtype"))
            if dtype_value is not None:
                model_kwargs["torch_dtype"] = dtype_value
            if case.get("attn_implementation"):
                model_kwargs["attn_implementation"] = str(case["attn_implementation"])

            load_start = time.perf_counter()
            model = SentenceTransformer(
                case["model_name"],
                trust_remote_code=True,
                local_files_only=True,
                truncate_dim=int(case["embedding_dim"]),
                device=device,
                model_kwargs=model_kwargs or None,
            )
            try:
                row["model_param_dtype"] = str(next(model.parameters()).dtype)
            except StopIteration:
                row["model_param_dtype"] = None
            if case.get("max_seq_length"):
                model.max_seq_length = int(case["max_seq_length"])
            row["effective_max_seq_length"] = int(getattr(model, "max_seq_length", 0) or 0)
            row["load_sec"] = round(time.perf_counter() - load_start, 4)

            lengths = _token_lengths(model, texts)
            row["token_lengths"] = _percentiles(lengths)
            order = list(range(doc_count))
            encode_texts = texts
            if row["length_bucket"]:
                order = sorted(order, key=lambda idx: lengths[idx])
                encode_texts = [texts[idx] for idx in order]

            encode_start = time.perf_counter()
            embeddings = model.encode(
                encode_texts,
                batch_size=row["batch_size"],
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            if row["length_bucket"]:
                restored = np.empty_like(embeddings)
                for encoded_idx, original_idx in enumerate(order):
                    restored[original_idx] = embeddings[encoded_idx]
                embeddings = restored
            row["encode_sec"] = round(time.perf_counter() - encode_start, 4)
            row["actual_dim"] = int(embeddings.shape[1])
            if row["actual_dim"] != row["embedding_dim"]:
                raise ValueError(f"dimension mismatch: expected {row['embedding_dim']} got {row['actual_dim']}")

            save_start = time.perf_counter()
            np.save(case_dir / "embeddings.npy", embeddings.astype(np.float32, copy=False))
            row["save_sec"] = round(time.perf_counter() - save_start, 4)
            row["total_sec"] = round(time.perf_counter() - case_start, 4)
            row["docs_per_sec"] = round(doc_count / row["total_sec"], 4)
            if torch.cuda.is_available():
                row["max_cuda_memory_mib"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            row["status"] = "completed"
        except Exception as exc:  # keep later cases running
            row["status"] = "failed"
            row["error"] = repr(exc)
            row["total_sec"] = round(time.perf_counter() - case_start, 4)
        finally:
            summary["cases"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_report(summary, output / "report.md")
            try:
                del model  # type: ignore[name-defined]
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding speed tuning sweep.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-mode", default="3T+A")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
