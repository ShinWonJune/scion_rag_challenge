from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TAG_RE = re.compile(r"<[^>]+>")
CSV_DOC_RE = re.compile(
    r"Title:\s*(?P<title>.*?),\s*Abstract:\s*(?P<abstract>.*?),\s*Source:\s*(?P<url>.*)$",
    re.S,
)
CN_RE = re.compile(r"[?&]cn=([^,&\s]+)")


MANUAL_OVERRIDES: dict[str, str] = {
    # These questions are not reliably covered by scion_answer_docs.txt because
    # the tail of that file has X marks and shifted numbering. The IDs below
    # come from ScienceON fullrun search/retrieval outputs.
    "39": "ART001468056",
    "40": "NART138954953",
    "41": "NPAP11397415",
    "42": "DIKO0008088315",
    "43": "NPAP12842380",
    "44": "ART002777203",
    "45": "DIKO0012481742",
    "46": "DIKO0008088315",
    "47": "NPAP12842380",
    "48": "JAKO201608450941626",
    "49": "JAKO201909358629507",
}

JUDGMENT_OVERRIDES: dict[str, tuple[str, str]] = {
    "37": (
        "sufficient",
        "초록에 smart farm, low-carbon green industry policy, IT convergence technology의 적용과 효과 분석이 직접 포함된다.",
    ),
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "using",
    "used",
    "use",
    "can",
    "how",
    "what",
    "would",
    "could",
    "main",
    "key",
    "briefly",
    "concisely",
    "summarize",
    "summary",
    "provide",
    "outline",
    "describe",
    "through",
    "based",
    "study",
    "research",
    "method",
    "methods",
    "result",
    "results",
    "analysis",
    "model",
    "models",
    "approach",
    "approaches",
}


@dataclass
class Candidate:
    qid: str
    doc_id: str
    title: str
    abstract: str
    url: str
    source: str
    source_kind: str
    csv_rank: int | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    search_rank: int | None = None


ADDITIONAL_SCIENCEON_DOCS: dict[str, list[Candidate]] = {
    "39": [
        Candidate(
            qid="39",
            doc_id="ART001468056",
            title="AHP를 이용한 에너지-IT 융합기술 도출에 관한 연구",
            abstract=(
                "세계적으로 비효율적인 에너지 소비로 인한 에너지 및 환경 문제가 지속적으로 대두되고 있다. "
                "최근 이 같은 문제를 해결하기 위해서 에너지-IT(Energy-IT, EIT)융합기술이 효과적인 해결책으로 큰 관심을 받고 있다. "
                "하지만 국내에서는 스마트그리드 외 EIT 융합기술에 대한 정책적 연구개발 및 투자가 미흡한 실정이다. "
                "따라서 본 논문에서는 EIT 융합 기술의 효용성을 조사하고 AHP(Analytic Hierarchy Process) 기법을 이용하여 EIT 융합기술을 도출한다. "
                "본 연구를 통하여 정부의 국가 에너지 문제 해결과 경쟁력을 향상을 위한 정책 결정에 기여할 것으로 기대한다. "
                "연구결과 에너지 저감 분야 중에서는 에너지절약형건물(green building) 분야가 가장 효용성 있는 융합분야로 분석되었고, "
                "에너지절약형건물 분야 내의 융합기술 중에서 네트워크 기능을 활용한 건물 내 에너지소비기기가 기술성, 경제성 부문에서 높은 가중치를 받으면서 가장 높은 기술로 평가되었다."
            ),
            url="http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue=05787966&target=NART&cn=ART001468056",
            source="ScienceON",
            source_kind="external_scienceon_title_verification",
        )
    ]
}


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = html.unescape(TAG_RE.sub(" ", text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_doc_id(url: str, fallback: str = "") -> str:
    match = CN_RE.search(url or "")
    return clean_text(match.group(1)) if match else clean_text(fallback)


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip().lower().rstrip(".,")


def tokenize(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9_+\-]*|[가-힣]{2,}", clean_text(text).lower())
    return {t for t in raw if len(t) > 1 and t not in STOPWORDS}


def overlap(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


def parse_csv_candidate(qid: str, rank: int, text: str) -> Candidate | None:
    match = CSV_DOC_RE.search(text or "")
    if not match:
        return None
    url = clean_text(match.group("url"))
    doc_id = get_doc_id(url)
    if not doc_id:
        return None
    return Candidate(
        qid=qid,
        doc_id=doc_id,
        title=clean_text(match.group("title")),
        abstract=clean_text(match.group("abstract")),
        url=url,
        source="ScienceON",
        source_kind="test_csv_candidate",
        csv_rank=rank,
    )


def load_test_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_seed_titles(path: Path) -> dict[str, str]:
    seeds: dict[str, str] = {}
    if not path.exists():
        return seeds
    for idx, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?P<mark>[Oo])\s+(?P<num>\d+)\.\s*(?P<title>.+)$", line)
        if not match:
            continue
        # Use only entries whose embedded number agrees with line order.
        # The tail of this file contains shifted numbering, so accepting it
        # blindly creates false gold labels.
        if int(match.group("num")) == idx:
            seeds[str(idx)] = clean_text(match.group("title")).rstrip(",")
    return seeds


def add_candidate(pool: dict[tuple[str, str], Candidate], cand: Candidate) -> None:
    key = (cand.qid, cand.doc_id)
    existing = pool.get(key)
    if existing is None:
        pool[key] = cand
        return
    if len(cand.abstract) > len(existing.abstract):
        existing.abstract = cand.abstract
    if len(cand.title) > len(existing.title):
        existing.title = cand.title
    if cand.url and not existing.url:
        existing.url = cand.url
    if cand.csv_rank is not None:
        existing.csv_rank = min(existing.csv_rank or cand.csv_rank, cand.csv_rank)
    if cand.search_rank is not None:
        existing.search_rank = min(existing.search_rank or cand.search_rank, cand.search_rank)
    if cand.retrieval_rank is not None:
        existing.retrieval_rank = min(existing.retrieval_rank or cand.retrieval_rank, cand.retrieval_rank)
        existing.retrieval_score = max(existing.retrieval_score or 0.0, cand.retrieval_score or 0.0)
    if cand.source_kind not in existing.source_kind:
        existing.source_kind = f"{existing.source_kind}+{cand.source_kind}"


def load_search_meta(path: Path) -> dict[str, list[Candidate]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[Candidate]] = defaultdict(list)
    for row in payload.get("results", []):
        qid = str(row.get("question_id", ""))
        for rank, doc in enumerate(row.get("documents", []), 1):
            doc_id = clean_text(doc.get("doc_id") or doc.get("CN"))
            if not qid or not doc_id:
                continue
            out[qid].append(
                Candidate(
                    qid=qid,
                    doc_id=doc_id,
                    title=clean_text(doc.get("title")),
                    abstract=clean_text(doc.get("abstract")),
                    url=clean_text(doc.get("url")),
                    source="ScienceON",
                    source_kind="scienceon_search_meta",
                    search_rank=rank,
                )
            )
    return out


def load_retrieval_hits(path: Path) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = defaultdict(list)
    if not path.exists():
        return out
    for file in path.glob("*.json"):
        payload = json.loads(file.read_text(encoding="utf-8"))
        qid = str(payload.get("id", file.name.split("_", 1)[0]))
        for result in payload.get("retrieval_results", []):
            for hit in result.get("hits", []):
                doc_id = clean_text(hit.get("doc_id") or hit.get("CN") or hit.get("cn"))
                if not doc_id:
                    continue
                out[qid].append(
                    Candidate(
                        qid=qid,
                        doc_id=doc_id,
                        title=clean_text(hit.get("title")),
                        abstract=clean_text(hit.get("abstract")),
                        url=clean_text(hit.get("url")),
                        source="ScienceON",
                        source_kind="retrieval_hit",
                        retrieval_rank=int(hit.get("rank", 9999)),
                        retrieval_score=float(hit.get("score", 0.0)),
                    )
                )
    return out


def score_candidate(
    row: dict[str, str], cand: Candidate, seed_title: str | None, override_doc_id: str | None
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    question = row.get("Question", "")
    answer = row.get("SAI_Answer", "")
    translated_question = row.get("translated_question", "")
    translated_answer = row.get("translated_SAI_answer", "")
    text = f"{cand.title} {cand.abstract}"

    q_sim = overlap(question, text)
    a_sim = overlap(answer, text)
    tq_sim = 0.0 if translated_question == "없음" else overlap(translated_question, text)
    ta_sim = 0.0 if translated_answer == "없음" else overlap(translated_answer, text)
    score += 3.5 * q_sim + 2.5 * a_sim + 2.0 * tq_sim + 1.5 * ta_sim
    reasons.append(f"question_overlap={q_sim:.3f}")
    reasons.append(f"answer_overlap={a_sim:.3f}")

    if seed_title and normalize_title(seed_title) in normalize_title(cand.title):
        score += 8.0
        reasons.append("seed_title_match")
    if override_doc_id and cand.doc_id == override_doc_id:
        score += 10.0
        reasons.append("manual_scienceon_override")

    if cand.retrieval_rank:
        score += max(0.0, 3.0 - 0.25 * (cand.retrieval_rank - 1))
        reasons.append(f"retrieval_rank={cand.retrieval_rank}")
    if cand.retrieval_score:
        score += cand.retrieval_score
    if cand.search_rank:
        score += max(0.0, 1.5 - 0.05 * (cand.search_rank - 1))
        reasons.append(f"search_rank={cand.search_rank}")
    if cand.csv_rank:
        score += max(0.0, 1.0 - 0.02 * (cand.csv_rank - 1))
        reasons.append(f"csv_rank={cand.csv_rank}")

    if cand.abstract and cand.abstract not in {"없음", "?놁쓬"}:
        score += 0.75
    else:
        score -= 2.0
        reasons.append("missing_abstract")

    return score, reasons


def judge_sufficiency(row: dict[str, str], cand: Candidate, score_reasons: list[str]) -> tuple[str, str]:
    question = row.get("Question", "")
    answer = row.get("SAI_Answer", "")
    text = f"{cand.title} {cand.abstract}"
    q_terms = tokenize(question)
    a_terms = tokenize(answer)
    doc_terms = tokenize(text)
    q_hits = sorted(q_terms & doc_terms)
    a_hits = sorted(a_terms & doc_terms)
    has_override = "manual_scienceon_override" in score_reasons
    has_seed = "seed_title_match" in score_reasons
    abstract_len = len(cand.abstract)

    if abstract_len < 40:
        return "insufficient", "초록이 없거나 너무 짧아 질문 답변 근거로 쓰기 어렵다."
    if has_seed or has_override:
        if len(q_hits) >= 2 or len(a_hits) >= 4 or cand.retrieval_rank == 1:
            return (
                "sufficient",
                "ScienceON 문서 제목/초록이 질문의 핵심 대상과 맞고, seed 또는 보정 규칙으로 정답 문서 후보가 확인된다.",
            )
        return (
            "partial",
            "정답 문서로 식별되지만 초록만으로는 질문의 세부 요구를 모두 확인하기 어렵다.",
        )
    if len(q_hits) >= 4 and len(a_hits) >= 5:
        return "sufficient", "질문 및 예시 답변의 핵심 용어가 초록에 충분히 나타난다."
    if len(q_hits) >= 2 or len(a_hits) >= 4:
        return "partial", "질문과 관련성은 높지만 초록만으로 모든 세부 내용을 보장하기는 어렵다."
    return "insufficient", "제목 또는 초록의 핵심 용어 매칭이 부족하다."


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--seed-docs", default="data/scion_answer_docs.txt")
    parser.add_argument(
        "--search-meta",
        default="outputs/e2e_vllm20b_testcsv_scienceon_fullrun/search/search_meta_results.json",
    )
    parser.add_argument(
        "--retrieval-dir",
        default="outputs/e2e_vllm20b_testcsv_scienceon_fullrun/retrieval",
    )
    parser.add_argument("--output-dir", default="experiments/gold_scienceon/artifacts")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_test_rows(Path(args.test_csv))
    seeds = load_seed_titles(Path(args.seed_docs))
    search_meta = load_search_meta(Path(args.search_meta))
    retrieval_hits = load_retrieval_hits(Path(args.retrieval_dir))

    pool: dict[tuple[str, str], Candidate] = {}
    candidate_cols = [c for c in rows[0] if c.startswith("retrieved_article_name_")]
    for row_idx, row in enumerate(rows):
        qid = str(row.get("id") or row_idx)
        for rank, col in enumerate(candidate_cols, 1):
            cand = parse_csv_candidate(qid, rank, row.get(col, ""))
            if cand:
                add_candidate(pool, cand)
        for cand in search_meta.get(qid, []):
            add_candidate(pool, cand)
        for cand in retrieval_hits.get(qid, []):
            add_candidate(pool, cand)
        for cand in ADDITIONAL_SCIENCEON_DOCS.get(qid, []):
            add_candidate(pool, cand)

    scored_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        qid = str(row.get("id") or row_idx)
        per_q = [cand for (cand_qid, _), cand in pool.items() if cand_qid == qid]
        ranked: list[tuple[float, Candidate, list[str]]] = []
        for cand in per_q:
            score, reasons = score_candidate(row, cand, seeds.get(qid), MANUAL_OVERRIDES.get(qid))
            ranked.append((score, cand, reasons))
        ranked.sort(key=lambda item: item[0], reverse=True)

        for rank, (score, cand, reasons) in enumerate(ranked[:10], 1):
            judgment, judgment_reason = judge_sufficiency(row, cand, reasons)
            scored_rows.append(
                {
                    "qid": qid,
                    "candidate_rank": rank,
                    "score": round(score, 4),
                    "judgment": judgment,
                    "doc": asdict(cand),
                    "score_reasons": reasons,
                    "judgment_reason": judgment_reason,
                }
            )

        if not ranked:
            gold_rows.append(
                {
                    "qid": qid,
                    "question": row.get("Question", ""),
                    "status": "missing",
                    "judgment": "insufficient",
                    "judgment_reason": "ScienceON 후보 문서를 확보하지 못했다.",
                }
            )
            continue

        score, cand, reasons = ranked[0]
        judgment, judgment_reason = judge_sufficiency(row, cand, reasons)
        if qid in JUDGMENT_OVERRIDES:
            judgment, judgment_reason = JUDGMENT_OVERRIDES[qid]
        gold_rows.append(
            {
                "qid": qid,
                "question": row.get("Question", ""),
                "translated_question": row.get("translated_question", ""),
                "gold_doc_id": cand.doc_id,
                "gold_title": cand.title,
                "gold_abstract": cand.abstract,
                "gold_url": cand.url,
                "source": cand.source,
                "source_kind": cand.source_kind,
                "csv_rank": cand.csv_rank,
                "search_rank": cand.search_rank,
                "retrieval_rank": cand.retrieval_rank,
                "retrieval_score": cand.retrieval_score,
                "score": round(score, 4),
                "judgment": judgment,
                "judgment_reason": judgment_reason,
                "score_reasons": reasons,
                "seed_title": seeds.get(qid),
                "manual_override_doc_id": MANUAL_OVERRIDES.get(qid),
            }
        )

    write_jsonl(out_dir / "scienceon_gold_docs.jsonl", gold_rows)
    write_jsonl(out_dir / "scienceon_candidate_judgments_top10.jsonl", scored_rows)

    with (out_dir / "scienceon_gold_docs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "qid",
            "judgment",
            "gold_doc_id",
            "gold_title",
            "gold_url",
            "source_kind",
            "csv_rank",
            "search_rank",
            "retrieval_rank",
            "score",
            "judgment_reason",
            "question",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in gold_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    counts: dict[str, int] = defaultdict(int)
    for row in gold_rows:
        counts[row.get("judgment", "unknown")] += 1
    report = {
        "total_questions": len(rows),
        "candidate_count": len(pool),
        "judgment_counts": dict(sorted(counts.items())),
        "outputs": [
            str(out_dir / "scienceon_gold_docs.jsonl"),
            str(out_dir / "scienceon_gold_docs.csv"),
            str(out_dir / "scienceon_candidate_judgments_top10.jsonl"),
        ],
        "notes": [
            "ScienceON-only candidates were drawn from test.csv, fullrun search_meta_results.json, and fullrun retrieval hits.",
            "scion_answer_docs.txt was used only where entries are marked O/o and embedded numbering matches row order.",
            "Manual overrides are recorded in each row for tail questions where the seed file is shifted or marked X.",
        ],
    }
    (out_dir / "README.md").write_text(
        "# ScienceON Gold Document Artifacts\n\n"
        f"- total_questions: {report['total_questions']}\n"
        f"- candidate_count: {report['candidate_count']}\n"
        f"- judgment_counts: {report['judgment_counts']}\n\n"
        "Files:\n"
        "- `scienceon_gold_docs.jsonl`: one selected ScienceON gold document per question.\n"
        "- `scienceon_gold_docs.csv`: compact review table.\n"
        "- `scienceon_candidate_judgments_top10.jsonl`: top-10 judged candidates per question.\n\n"
        "Selection uses local ScienceON evidence from `data/test.csv` and previous fullrun ScienceON outputs.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
