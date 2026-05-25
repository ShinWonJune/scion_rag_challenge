#!/usr/bin/env python3
"""Aggregate LLM-as-judge scores + generation cost for the two arms and write
a report + append a section to docs/FOLLOWUP_EXPERIMENT_SUMMARY.md.

Quality:  per-arm mean overall + sub-axes, paired delta (B-A) with bootstrap CI,
          win/tie/loss on overall.
Cost:     per-arm prompt/completion/total tokens + generation latency.
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "experiments/outputs/exp_answer_quality_judge"
SUMMARY_DOC = REPO / "docs/FOLLOWUP_EXPERIMENT_SUMMARY.md"
ARMS = ["dense_top5", "rerank_top3"]
AXES = ["answer_relevance", "evidence_coverage", "faithfulness_to_gold",
        "faithfulness_to_retrieved", "specificity", "overall"]
BOOT_N = 2000
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_judge(arm: str) -> dict[str, dict]:
    rows = load_jsonl(OUT_ROOT / arm / "judge" / "judge_results.jsonl")
    return {str(r["qid"]): r for r in rows}


def load_gen_summary(arm: str) -> dict:
    return json.loads((OUT_ROOT / arm / "gen_summary.json").read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(deltas: list[float], n: int = BOOT_N) -> tuple[float, float]:
    rnd = random.Random(SEED)
    k = len(deltas)
    means = []
    for _ in range(n):
        sample = [deltas[rnd.randrange(k)] for _ in range(k)]
        means.append(mean(sample))
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return lo, hi


def main() -> None:
    ja = load_judge("dense_top5")
    jb = load_judge("rerank_top3")
    qids = sorted(set(ja) & set(jb), key=lambda x: int(x))
    assert len(ja) == len(jb) == len(qids), f"qid mismatch: A={len(ja)} B={len(jb)} common={len(qids)}"
    print(f"paired qids: {len(qids)}")

    # per-arm axis means
    axis_means = {arm: {} for arm in ARMS}
    for arm, jd in (("dense_top5", ja), ("rerank_top3", jb)):
        for ax in AXES:
            axis_means[arm][ax] = round(mean([float(jd[q].get(ax, 0) or 0) for q in qids]), 4)

    # paired delta on overall (B - A)
    deltas = [float(jb[q].get("overall", 0) or 0) - float(ja[q].get("overall", 0) or 0) for q in qids]
    mean_delta = mean(deltas)
    lo, hi = bootstrap_ci(deltas)
    wins = sum(1 for d in deltas if d > 0)   # rerank better
    losses = sum(1 for d in deltas if d < 0)  # dense better
    ties = sum(1 for d in deltas if d == 0)

    # label distribution (server-computed correctness label, if present)
    def label_counts(jd):
        c = {}
        for q in qids:
            lab = jd[q].get("label", "n/a")
            c[lab] = c.get(lab, 0) + 1
        return c

    # generation cost
    gen = {arm: load_gen_summary(arm) for arm in ARMS}

    report = {
        "n_paired": len(qids),
        "axis_means": axis_means,
        "overall_delta_B_minus_A": round(mean_delta, 4),
        "overall_delta_95ci": [round(lo, 4), round(hi, 4)],
        "win_tie_loss_rerank_vs_dense": {"rerank_better": wins, "tie": ties, "dense_better": losses},
        "label_counts": {"dense_top5": label_counts(ja), "rerank_top3": label_counts(jb)},
        "generation_cost": gen,
    }
    (OUT_ROOT / "compare_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # ---- report.md ----
    A, B = axis_means["dense_top5"], axis_means["rerank_top3"]
    ga, gb = gen["dense_top5"], gen["rerank_top3"]
    sig = "유의" if (lo > 0 or hi < 0) else "비유의(0 포함)"
    md = f"""# 응답 품질 비교: dense-top5 vs rerank-top3 (LLM-as-judge)

- 생성/judge 모델: `{ga['model']}` (vLLM), judge 절대 채점 rubric `gold_answer_judge_v2`
- 동일 41Q gold, 동일 encoder(gte), 유일 변수 = rerank 단계(+context docs 5→3)
- judge = 생성 모델과 동일(gpt-oss) → self-judge bias는 **두 arm 대칭**이라 delta에 교란 작음

## 품질 (judge 평균, N={len(qids)})

| Axis (만점) | dense-top5 | rerank-top3 | Δ(B−A) |
|---|---:|---:|---:|
| answer_relevance (2) | {A['answer_relevance']} | {B['answer_relevance']} | {round(B['answer_relevance']-A['answer_relevance'],4)} |
| evidence_coverage (3) | {A['evidence_coverage']} | {B['evidence_coverage']} | {round(B['evidence_coverage']-A['evidence_coverage'],4)} |
| faithfulness_to_gold (3) | {A['faithfulness_to_gold']} | {B['faithfulness_to_gold']} | {round(B['faithfulness_to_gold']-A['faithfulness_to_gold'],4)} |
| faithfulness_to_retrieved (3) | {A['faithfulness_to_retrieved']} | {B['faithfulness_to_retrieved']} | {round(B['faithfulness_to_retrieved']-A['faithfulness_to_retrieved'],4)} |
| specificity (2) | {A['specificity']} | {B['specificity']} | {round(B['specificity']-A['specificity'],4)} |
| **overall (10)** | **{A['overall']}** | **{B['overall']}** | **{round(mean_delta,4)}** |

- **overall paired delta (rerank − dense) = {round(mean_delta,3)}**, 95% bootstrap CI [{round(lo,3)}, {round(hi,3)}] (n={BOOT_N}) → {sig}
- **win/tie/loss (rerank 기준)**: {wins} / {ties} / {losses}

## 생성 비용 (gpt-oss-20b, max_tokens={ga['max_answer_tokens']})

| 항목 | dense-top5 | rerank-top3 | Δ(B−A) |
|---|---:|---:|---:|
| context docs (mean) | {ga['n_context_docs_mean']} | {gb['n_context_docs_mean']} | {round(gb['n_context_docs_mean']-ga['n_context_docs_mean'],3)} |
| prompt tokens (mean) | {ga['prompt_tokens_mean']} | {gb['prompt_tokens_mean']} | {round(gb['prompt_tokens_mean']-ga['prompt_tokens_mean'],1)} |
| completion tokens (mean) | {ga['completion_tokens_mean']} | {gb['completion_tokens_mean']} | {round(gb['completion_tokens_mean']-ga['completion_tokens_mean'],1)} |
| total tokens (mean) | {ga['total_tokens_mean']} | {gb['total_tokens_mean']} | {round(gb['total_tokens_mean']-ga['total_tokens_mean'],1)} |
| gen latency sec (mean/q) | {ga['gen_latency_sec_mean']} | {gb['gen_latency_sec_mean']} | {round(gb['gen_latency_sec_mean']-ga['gen_latency_sec_mean'],4)} |
| total prompt tokens (41Q) | {ga['prompt_tokens_total']} | {gb['prompt_tokens_total']} | {gb['prompt_tokens_total']-ga['prompt_tokens_total']} |

## 한계
- N={len(qids)} 단일 dataset. self-judge(gpt-oss 생성+채점) — delta는 대칭 상쇄지만 절대 점수는 호의적일 수 있음.
- 절대 점수는 ceiling 부근 압축 가능 → paired delta·win/tie/loss 병행 해석.
- gen latency(mean/q)는 동시 요청 환경의 평균 처리시간으로, vLLM 서버 부하·동시성에 따라 변동.
"""
    (OUT_ROOT / "report.md").write_text(md, encoding="utf-8")
    print("WROTE report.md")

    # ---- append to FOLLOWUP_EXPERIMENT_SUMMARY.md ----
    doc = SUMMARY_DOC.read_text(encoding="utf-8")
    marker = "## 응답 품질 비교: dense-top5 vs rerank-top3 (LLM-as-judge)"
    if marker not in doc:
        section = "\n\n---\n\n# Part J. 응답 품질 (LLM-as-judge)\n\n" + md.replace(
            "# 응답 품질 비교: dense-top5 vs rerank-top3 (LLM-as-judge)",
            marker, 1)
        SUMMARY_DOC.write_text(doc.rstrip() + section + "\n", encoding="utf-8")
        print("APPENDED to FOLLOWUP_EXPERIMENT_SUMMARY.md")
    else:
        print("section already present; skipped append")


if __name__ == "__main__":
    main()
