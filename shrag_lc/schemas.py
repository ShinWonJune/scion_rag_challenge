"""Structured result schemas for the LangChain SHRAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_KEYWORD_SPLIT_RE = re.compile(r"[\s\-]+")


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def normalize_keyword_values(values: list[str]) -> list[str]:
    """Normalize LLM keywords like the original SHRAG parser.

    LLMs often return phrase-like keywords. The legacy pipeline splits those
    phrases by whitespace or hyphen before deduplication, so the LC structured
    path must apply the same rule.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        for keyword in _KEYWORD_SPLIT_RE.split(value.strip()):
            keyword = keyword.strip()
            if len(keyword) <= 1:
                continue
            key = keyword.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(keyword)
    return out[:10]


class PipelineStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_DOCUMENTS = "no_documents"
    FAILED = "failed"


class StageStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_DOCUMENTS = "no_documents"
    SKIPPED = "skipped"
    FAILED = "failed"


class StageName(StrEnum):
    KEYWORD_EXTRACTION = "keyword_extraction"
    SEARCH = "search"
    CORPUS_BUILD = "corpus_build"
    EMBEDDING = "embedding"
    INDEX_BUILD = "index_build"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    GENERATION = "generation"


class KeywordList(BaseModel):
    """Structured keyword output for a single language."""

    keywords: list[str] = Field(default_factory=list)

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        return normalize_keyword_values(values)


@dataclass
class PipelineError:
    stage: str
    message: str
    error_type: str = "Error"
    question_id: str | None = None
    question: str | None = None
    source: str | None = None
    term: str | None = None
    language: str | None = None

    @classmethod
    def from_exception(
        cls,
        stage: StageName | str,
        exc: Exception,
        *,
        question_id: str | None = None,
        question: str | None = None,
        source: str | None = None,
        term: str | None = None,
        language: str | None = None,
    ) -> "PipelineError":
        return cls(
            stage=str(stage),
            error_type=type(exc).__name__,
            message=str(exc),
            question_id=question_id,
            question=question,
            source=source,
            term=term,
            language=language,
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StageReport:
    stage: str
    status: str = StageStatus.SUCCESS
    started_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None
    counts: dict[str, int | float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[PipelineError] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: StageStatus | str | None = None) -> "StageReport":
        if status is not None:
            self.status = str(status)
        self.completed_at = utc_now_iso()
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["errors"] = [e.to_dict() for e in self.errors]
        return {k: v for k, v in data.items() if v not in (None, [], {})}


def stage_report(
    stage: StageName | str,
    status: StageStatus | str = StageStatus.SUCCESS,
    *,
    counts: dict[str, int | float] | None = None,
    warnings: list[str] | None = None,
    errors: list[PipelineError] | None = None,
    metadata: dict[str, Any] | None = None,
) -> StageReport:
    return StageReport(
        stage=str(stage),
        status=str(status),
        counts=counts or {},
        warnings=warnings or [],
        errors=errors or [],
        metadata=metadata or {},
    ).finish(status)


def reports_to_dict(reports: dict[str, StageReport]) -> dict[str, dict[str, Any]]:
    return {name: report.to_dict() for name, report in reports.items()}


def errors_to_dict(errors: list[PipelineError]) -> list[dict[str, Any]]:
    return [error.to_dict() for error in errors]
