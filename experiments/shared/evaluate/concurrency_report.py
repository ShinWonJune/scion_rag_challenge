from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_meta(path: Path) -> Path | None:
    if path.is_file() and path.name == "search_meta_results.json":
        return path
    candidates = sorted(path.rglob("search_meta_results.json"))
    return candidates[0] if candidates else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize concurrency sweep outputs")
    parser.add_argument("--input", required=True, help="Sweep output root")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--output", required=True, help="Markdown report path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input)
    rows = []
    for run_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        meta_path = _find_meta(run_dir)
        if not meta_path:
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        stats = payload.get("request_stats", {})
        request_count = int(stats.get("request_count", 0))
        rate_limit_count = int(stats.get("rate_limit_count", 0))
        rate_limit_rate = rate_limit_count / max(1, request_count)
        rows.append(
            {
                "name": run_dir.name,
                "request_count": request_count,
                "rate_limit_count": rate_limit_count,
                "rate_limit_rate": rate_limit_rate,
                "runtime_sec": payload.get("runtime_sec"),
            }
        )
    rows.sort(key=lambda item: item["rate_limit_rate"])
    selected = next((row for row in rows if row["rate_limit_rate"] <= args.epsilon), None)
    lines = [
        "# Concurrency Sweep Report",
        "",
        "| run | requests | 429 | 429 rate | runtime sec |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['request_count']} | {row['rate_limit_count']} | {row['rate_limit_rate']:.4f} | {row['runtime_sec']} |"
        )
    lines.append("")
    lines.append(f"Selected: `{selected['name']}`" if selected else "Selected: none")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
