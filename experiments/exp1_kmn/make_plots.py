"""Generate Phase A heatmap, Phase B curve, and Pareto plot from _aggregate.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _f(v):
    try:
        return float(v) if v else None
    except Exception:
        return None


def heatmap_phaseA(rows: list[dict], out: Path) -> None:
    a = [r for r in rows if r["phase"] == "phaseA" and _f(r["hit_at_5"]) is not None]
    if not a:
        print("No phaseA rows for heatmap")
        return
    ks = sorted({int(r["k"]) for r in a})
    ns = sorted({int(r["n"]) for r in a})
    mat = np.full((len(ns), len(ks)), np.nan)
    for r in a:
        i = ns.index(int(r["n"]))
        j = ks.index(int(r["k"]))
        mat[i, j] = _f(r["hit_at_5"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f"n={n}" for n in ns])
    ax.set_xlabel("k (docs per term)")
    ax.set_ylabel("n (search terms per language)")
    ax.set_title("Phase A — Hit@5 (m=70)")
    for i in range(len(ns)):
        for j in range(len(ks)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Hit@5")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved heatmap → {out}")


def curve_phaseB(rows: list[dict], out: Path) -> None:
    b = [r for r in rows if r["phase"] == "phaseB" and _f(r["hit_at_5"]) is not None]
    if not b:
        print("No phaseB rows for curve")
        return
    b.sort(key=lambda r: int(r["m"]))
    ms = [int(r["m"]) for r in b]
    h5 = [_f(r["hit_at_5"]) for r in b]
    cov = [_f(r["gold_found_rate"]) for r in b]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ms, h5, marker="o", label="Hit@5")
    ax.plot(ms, cov, marker="s", label="gold_found_rate")
    ax.set_xlabel("m (target_documents)")
    ax.set_ylabel("score")
    ax.set_title("Phase B — m sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved curve → {out}")


def pareto(rows: list[dict], out: Path) -> None:
    rs = [r for r in rows if _f(r["hit_at_5"]) is not None and _f(r["total_sec"]) is not None]
    if not rs:
        print("No rows for pareto")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"phaseA": "tab:blue", "phaseB": "tab:orange", "phaseC": "tab:green"}
    for ph in colors:
        sub = [r for r in rs if r["phase"] == ph]
        if not sub:
            continue
        xs = [_f(r["total_sec"]) / 60.0 for r in sub]
        ys = [_f(r["hit_at_5"]) for r in sub]
        ax.scatter(xs, ys, label=ph, c=colors[ph], alpha=0.7, s=42)
        for r, x, y in zip(sub, xs, ys):
            ax.annotate(r["cell"].replace("_m70", "").replace("k", "").replace("n", ""), (x, y), fontsize=7, alpha=0.6)
    ax.set_xlabel("total_time (min) — step1+step3+step4")
    ax.set_ylabel("Hit@5")
    ax.set_title("Pareto: Hit@5 vs total_time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved pareto → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/outputs/exp1_kmn/_aggregate.csv")
    ap.add_argument("--outdir", default="experiments/outputs/exp1_kmn/_plots")
    args = ap.parse_args()
    rows = _load(Path(args.csv))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    heatmap_phaseA(rows, out / "phaseA_heatmap.png")
    curve_phaseB(rows, out / "phaseB_curve.png")
    pareto(rows, out / "pareto.png")


if __name__ == "__main__":
    main()
