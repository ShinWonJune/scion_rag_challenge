"""Matplotlib-only plotting utilities for experiment result visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def pareto_plot(
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    xlabel: str = "Runtime (s)",
    ylabel: str = "Hit@5",
    title: str = "Pareto: Hit@5 vs Runtime",
    output_path: Path | str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, zorder=3)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def heatmap(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str = "Heatmap",
    xlabel: str = "",
    ylabel: str = "",
    fmt: str = ".3f",
    output_path: Path | str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(5, len(col_labels)), max(4, len(row_labels))))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def rank_shift_histogram(
    before_ranks: Sequence[int],
    after_ranks: Sequence[int],
    title: str = "Rank Shift Distribution",
    output_path: Path | str | None = None,
) -> None:
    """Plot distribution of (before_rank - after_rank) per query; positive = improved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas = [b - a for b, a in zip(before_ranks, after_ranks)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deltas, bins=range(min(deltas) - 1, max(deltas) + 2), edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Rank improvement (positive = reranker moved gold up)")
    ax.set_ylabel("Query count")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
