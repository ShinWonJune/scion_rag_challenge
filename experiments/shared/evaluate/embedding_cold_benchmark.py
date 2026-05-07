from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from shrag.pipeline._impl.build_vectordb import build_vectordb_search


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _prepare_config(original: Path, case_dir: Path) -> Path:
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["output_dir"] = str(case_dir)
    payload["output_file"] = str(case_dir / "vectordb.csv")
    payload.pop("cache_db_path", None)
    temp = case_dir / "encoder_config.cold.json"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return temp


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _find_vectordb(case_dir: Path) -> Path | None:
    direct = case_dir / "vectordb.csv"
    if direct.exists():
        return direct
    candidates = sorted(case_dir.rglob("vector_db_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _write_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exp12. Q41 Corpus Cold Embedding Benchmark",
        "",
        "Measures document embedding + vector DB build only. Search/acquisition, query encoding, retrieval, reranking, and generation are excluded.",
        "",
        f"- corpus: `{report['corpus']}`",
        f"- docs: `{report['doc_count']}`",
        "- embedding cache: disabled by removing `cache_db_path` from temporary configs",
        "",
        "| case | model | dim | docs | cold build sec | docs/sec | output |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['case_id']}"
            f" | `{row['model_name']}`"
            f" | {row['embedding_dim']}"
            f" | {row['doc_count']}"
            f" | {row['build_sec']}"
            f" | {row['docs_per_sec']}"
            f" | `{row.get('vectordb') or '-'}` |"
        )
    if len(report["cases"]) >= 2:
        base = report["cases"][0]
        for row in report["cases"][1:]:
            if base["build_sec"]:
                lines.extend(
                    [
                        "",
                        f"`{row['case_id']}` / `{base['case_id']}` build time ratio: `{round(row['build_sec'] / base['build_sec'], 4)}x`",
                    ]
                )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_yaml(Path(args.cases))
    corpus = Path(args.corpus)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    doc_count = _count_jsonl(corpus)
    report: dict[str, Any] = {
        "corpus": str(corpus),
        "doc_count": doc_count,
        "schema": args.schema,
        "cases": [],
    }
    for case in payload.get("cases", []):
        case_id = str(case["case_id"])
        case_dir = output / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        temp_config = _prepare_config(Path(case["encoder_config"]), case_dir)
        config_payload = json.loads(temp_config.read_text(encoding="utf-8"))
        started = time.perf_counter()
        build_vectordb_search(
            config_path=str(temp_config),
            data_schema=args.schema,
            docs_jsonl_path=str(corpus),
            auto_data_load=False,
            gpu_id=None,
        )
        build_sec = round(time.perf_counter() - started, 4)
        vectordb = _find_vectordb(case_dir)
        report["cases"].append(
            {
                "case_id": case_id,
                "model_name": config_payload["model_name"],
                "embedding_dim": config_payload["embedding_dim"],
                "doc_count": doc_count,
                "build_sec": build_sec,
                "docs_per_sec": round(doc_count / build_sec, 4) if build_sec else None,
                "temp_config": str(temp_config),
                "vectordb": str(vectordb) if vectordb else None,
            }
        )
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report, output / "report.md")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold document embedding/vector DB benchmark.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--schema", default="configs/csv_schema/test_2.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    report = run(_parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
