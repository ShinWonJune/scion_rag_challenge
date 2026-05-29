"""Manifest and stage-summary helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import PipelineError


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def summarize_results(results: list[dict]) -> dict[str, dict[str, Any]]:
    summary: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        stages = result.get("stages") or {}
        for stage_name, report in stages.items():
            status = report.get("status", "unknown")
            summary[stage_name][status] += 1
    return {stage: dict(counter) for stage, counter in sorted(summary.items())}


def collect_failures(results: list[dict]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for result in results:
        for error in result.get("errors") or []:
            failures.append(error)
        for report in (result.get("stages") or {}).values():
            for error in report.get("errors") or []:
                if error not in failures:
                    failures.append(error)
    return failures


def build_manifest(
    *,
    config: Any,
    questions: list[dict],
    results: list[dict],
    pipeline_mode: str,
    generated_at: str,
    artifacts: dict[str, str] | None = None,
    corpus: dict[str, Any] | None = None,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": generated_at.replace(":", "").replace("-", ""),
        "generated_at": generated_at,
        "completed_at": _now_iso(),
        "config": asdict(config),
        "n_questions": len(questions),
        "pipeline_mode": pipeline_mode,
        "artifacts": artifacts or {},
        "stage_summary": summarize_results(results),
        "failures": collect_failures(results),
    }
    if corpus is not None:
        manifest["corpus"] = corpus
    if index is not None:
        manifest["index"] = index
    return manifest


def write_json(path: str | Path, data: Any) -> None:
    import json

    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
