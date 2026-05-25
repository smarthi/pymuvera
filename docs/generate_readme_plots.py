"""Generate the illustrative README plots.

These plots are explanatory diagrams, not benchmark results. They encode the
configuration tradeoffs described in README.md: SimHash variance, partition
sparsity, fill cost, low-rank convergence, two-stage reranking recovery, and
the experimental CALIBRATED_EIGENBASIS spectral-bias tradeoff.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#4E79A7"
GREEN = "#59A14F"
ORANGE = "#F06A32"
RED = "#E15759"
DARK_ORANGE = "#D45A25"
GRAY = "#6B7280"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "images"


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "legend.frameon": True,
        }
    )


def _save(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)
    return output


def plot1_variance_vs_repetitions(output_dir: Path) -> Path:
    reps = np.arange(1, 21)
    variance = 1.0 / reps

    fig, ax = plt.subplots(figsize=(11.8, 7.3))
    ax.plot(
        reps,
        variance,
        color="#2F5F8F",
        marker="o",
        linewidth=2.8,
        label="FDE approximation variance proportional to 1/repetitions",
    )
    for y in [0.25, 0.125, 0.0625]:
        ax.axhline(y, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.annotate(
        "reps=4 -> 25% baseline variance",
        xy=(4, 0.25),
        xytext=(7.0, 0.32),
        arrowprops={"arrowstyle": "->", "color": "#8C8C8C"},
        color="#8C8C8C",
        fontsize=11,
    )
    ax.annotate(
        "reps=8 -> 12.5% variance",
        xy=(8, 0.125),
        xytext=(11.0, 0.18),
        arrowprops={"arrowstyle": "->", "color": "#8C8C8C"},
        color="#8C8C8C",
        fontsize=11,
    )
    ax.annotate(
        "reps=16 -> 6.25% variance",
        xy=(16, 0.0625),
        xytext=(11.0, 0.035),
        arrowprops={"arrowstyle": "->", "color": "#8C8C8C"},
        color="#8C8C8C",
        fontsize=11,
    )
    ax.set_title(
        "Error Source 1: SimHash partitioning variance decreases with repetitions\n"
        "More repetitions = more independent partition chances = lower variance",
        fontsize=18,
        pad=10,
    )
    ax.set_xlabel("num_repetitions", fontsize=13)
    ax.set_ylabel("Relative approximation variance", fontsize=13)
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    return _save(fig, output_dir / "plot1_variance_vs_repetitions.png")


def plot2_tokens_per_partition(output_dir: Path) -> Path:
    k_values = np.arange(1, 12)
    doc_lengths = [64, 256, 512, 1024]
    colors = ["#3BA17D", "#2F5F8F", ORANGE, "#D33F00"]

    fig, ax = plt.subplots(figsize=(11.74, 7.3))
    ax.axhspan(0, 2, color=RED, alpha=0.10)
    for doc_len, color in zip(doc_lengths, colors, strict=True):
        tokens_per_partition = doc_len / (2**k_values)
        ax.plot(
            k_values,
            tokens_per_partition,
            marker="o",
            linewidth=2.5,
            color=color,
            label=f"{doc_len} tokens/doc",
        )
    ax.axhline(2, color="#FF2E3B", linestyle="--", linewidth=1.8, label="fill threshold (< 2 tokens/partition)")
    ax.axhline(4, color="#FFA07A", linestyle=":", linewidth=1.4, label="ideal minimum (4 tokens/partition)")
    ax.text(8.0, 0.6, "fill_empty_partitions\nrecommended", color="#FF2E3B", fontsize=11)
    ax.set_title(
        "Partition occupancy tradeoff as k increases\n"
        "Too many partitions -> sparse slots -> suppressed recall",
        fontsize=18,
        pad=10,
    )
    ax.set_xlabel("num_simhash_projections (k)", fontsize=13)
    ax.set_ylabel("Average tokens per partition", fontsize=13)
    ax.set_ylim(0, 35)
    ax.set_xticks(k_values)
    ax.set_xticklabels([f"k={k}\n({2**k} parts)" for k in k_values])
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return _save(fig, output_dir / "plot2_tokens_per_partition.png")


def plot3_fill_cost_comparison(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(17.8, 7.7))

    k_values = np.arange(4, 12)
    empty_slots = 0.70 * (2**k_values)
    hamming_cost = 512 * k_values * empty_slots
    densifying_cost = empty_slots

    ax = axes[0]
    ax.plot(
        k_values,
        hamming_cost,
        color=ORANGE,
        marker="s",
        linewidth=2.5,
        label="Hamming NN fill\nO(num_tokens x k x num_empty)",
    )
    ax.plot(
        k_values,
        densifying_cost,
        color="#3BA17D",
        marker="o",
        linewidth=2.5,
        label="Densifying LSH fill\nO(num_empty)",
    )
    ax.set_yscale("log")
    ax.set_title("Fill cost: Hamming NN vs Densifying LSH\n(512-token doc, 70% empty slots)", fontsize=15)
    ax.set_xlabel("k (num_simhash_projections)", fontsize=13)
    ax.set_ylabel("Approximate operations (log scale)", fontsize=13)
    ax.legend(loc="upper left", fontsize=11)

    doc_counts = np.array([32, 64, 128, 256, 512, 1024, 2048])
    k = 8
    empty = 0.60 * (2**k)
    hamming_by_doc_len = doc_counts * k * empty
    densifying_by_doc_len = np.full_like(doc_counts, empty, dtype=float)

    ax = axes[1]
    ax.plot(doc_counts, hamming_by_doc_len, color=ORANGE, marker="s", linewidth=2.5, label="Hamming NN fill")
    ax.plot(doc_counts, densifying_by_doc_len, color="#3BA17D", marker="o", linewidth=2.5, label="Densifying LSH fill")
    ax.set_yscale("log")
    ax.set_title(
        "Fill cost vs document length (k=8, 60% empty)\n"
        "Densifying is constant; Hamming scales with doc length",
        fontsize=15,
    )
    ax.set_xlabel("Document token count", fontsize=13)
    ax.set_ylabel("Approximate operations (log scale)", fontsize=13)
    ax.legend(loc="upper left", fontsize=11)

    fig.suptitle("Error Source 7: Fill strategy cost comparison", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.0)
    return _save(fig, output_dir / "plot3_fill_cost_comparison.png")


def plot4_eggroll_vs_clt(output_dir: Path) -> Path:
    ranks = np.arange(1, 21)
    clt = 1 / np.sqrt(ranks)
    eggroll = 1 / ranks

    fig, ax = plt.subplots(figsize=(11.8, 7.3))
    ax.plot(ranks, clt, color=ORANGE, marker="s", linewidth=2.8, label="CLT rate O(r^-1/2) - standard random approximation")
    ax.plot(ranks, eggroll, color="#3BA17D", marker="o", linewidth=2.8, label="EGGROLL rate O(r^-1) - LOW_RANK_GAUSSIAN")
    ax.fill_between(ranks, eggroll, clt, color="#3BA17D", alpha=0.15, label="EGGROLL advantage")
    ax.annotate(
        "r=4: CLT ~= 0.50",
        xy=(4, 0.5),
        xytext=(7.0, 0.62),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
        fontsize=11,
    )
    ax.annotate(
        "r=4: EGGROLL ~= 0.25",
        xy=(4, 0.25),
        xytext=(7.0, 0.34),
        arrowprops={"arrowstyle": "->", "color": "#3BA17D"},
        color="#3BA17D",
        fontsize=11,
    )
    ax.set_title(
        "Error Source 5: LOW_RANK_GAUSSIAN convergence\n"
        "EGGROLL beats the CLT rate because symmetry kills all odd cumulants",
        fontsize=18,
        pad=10,
    )
    ax.set_xlabel("simhash_rank (r)", fontsize=13)
    ax.set_ylabel("Approximation error (relative)", fontsize=13)
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    return _save(fig, output_dir / "plot4_eggroll_vs_clt.png")


def plot5_two_stage_recall(output_dir: Path) -> Path:
    labels = ["Exact MaxSim\n(no FDE)", "FDE only\n(no rerank)", "FDE + MaxSim\nrerank top-100"]
    recall_at_1 = np.array([0.89, 0.61, 0.86])
    recall_at_5 = np.array([0.99, 0.90, 0.98])
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13.3, 7.3))
    bars1 = ax.bar(x - width / 2, recall_at_1, width, color=BLUE, label="R@1", alpha=0.95)
    bars5 = ax.bar(x + width / 2, recall_at_5, width, color=GREEN, label="R@5", alpha=0.95)
    ax.axhline(recall_at_1[0], color=BLUE, linestyle=":", linewidth=1.3, alpha=0.65)
    for bars in [bars1, bars5]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
    ax.annotate(
        "-28 pts\nSimHash\nerror",
        xy=(1 - width / 2, recall_at_1[1]),
        xytext=(0.5, 0.60),
        arrowprops={"arrowstyle": "<->", "color": "#D33F00", "lw": 1.8},
        color="#D33F00",
        fontsize=11,
        ha="center",
    )
    ax.annotate(
        "+25 pts\nreranking\nrecovery",
        xy=(2 - width / 2, recall_at_1[2]),
        xytext=(1.55, 0.76),
        arrowprops={"arrowstyle": "->", "color": "#3BA17D", "lw": 1.8},
        color="#2F8E67",
        fontsize=11,
        ha="center",
    )
    ax.set_title(
        "Reconstruction error in the two-stage retrieval pipeline\n"
        "FDE error is largely recovered by MaxSim reranking",
        fontsize=18,
        pad=10,
    )
    ax.set_ylabel("Recall", fontsize=13)
    ax.set_ylim(0.4, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    return _save(fig, output_dir / "plot5_two_stage_recall.png")


def plot6_error_breakdown(output_dir: Path) -> Path:
    labels = [
        "DEFAULT_IDENTITY\nk=4, reps=4",
        "DEFAULT_IDENTITY\nk=6, reps=8\n(recommended)",
        "DEFAULT_IDENTITY\nk=8, reps=8\n+ fill",
        "LOW_RANK_GAUSSIAN\nr=4, k=8, reps=8",
        "SRHT\nk=8, reps=8",
    ]
    simhash = np.array([0.30, 0.14, 0.10, 0.16, 0.10])
    aggregation = np.array([0.06, 0.04, 0.02, 0.04, 0.02])
    empty = np.array([0.08, 0.04, 0.01, 0.00, 0.01])
    low_rank = np.array([0.00, 0.00, 0.00, 0.05, 0.00])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(14.8, 7.3))
    ax.bar(x, simhash, color=BLUE, width=0.55, label="SimHash partitioning error")
    ax.bar(x, aggregation, bottom=simhash, color=GREEN, width=0.55, label="Aggregation (centroid) error")
    ax.bar(x, empty, bottom=simhash + aggregation, color="#F58657", width=0.55, label="Empty slot error")
    ax.bar(
        x,
        low_rank,
        bottom=simhash + aggregation + empty,
        color=DARK_ORANGE,
        width=0.55,
        label="LOW_RANK_GAUSSIAN extra error",
    )
    totals = simhash + aggregation + empty + low_rank
    for xi, total in zip(x, totals, strict=True):
        ax.text(xi, total + 0.005, f"{total:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title(
        "Reconstruction error breakdown by source across common configs\n"
        "(illustrative - relative magnitudes; all errors recovered by MaxSim reranking)",
        fontsize=17,
        pad=10,
    )
    ax.set_ylabel("Relative FDE approximation error", fontsize=13)
    ax.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    return _save(fig, output_dir / "plot6_error_breakdown.png")


def plot7_eigenbasis_spectral_bias(output_dir: Path) -> Path:
    dims = np.arange(1, 129)
    eigenvalues = np.exp(-(dims - 1) / 8.0) + 0.035 * np.exp(-(dims - 1) / 80.0)
    eigenvalues /= eigenvalues[0]

    head_signal = np.linspace(0.0, 1.0, 101)
    default_error = np.ones_like(head_signal)
    rotation_only = np.ones_like(head_signal)
    weighted_error = 0.62 + 0.78 * np.power(1.0 - head_signal, 1.15)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 7.3),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )

    ax = axes[0]
    ax.semilogy(dims, eigenvalues, color=BLUE, linewidth=3)
    ax.axvspan(1, 16, color=BLUE, alpha=0.12, label="principal subspace")
    ax.axvspan(80, 128, color=RED, alpha=0.10, label="low-variance tail")
    ax.axhline(1.0 / 128.0, color=GRAY, linewidth=2, linestyle="--", label="uniform reference")
    ax.set_title("Example calibrated eigenspectrum", fontsize=16, pad=14)
    ax.set_xlabel("Eigenbasis coordinate, sorted by variance", fontsize=12)
    ax.set_ylabel("Normalized eigenvalue (log scale)", fontsize=12)
    ax.legend(loc="upper right")
    ax.text(18, 0.26, "Eigenvalue weighting spends\nmore partition influence here", color="#2F5F8F", fontsize=11)
    ax.text(72, 0.017, "Tail details can be\nunder-partitioned", color="#B33A3D", fontsize=11)

    ax = axes[1]
    ax.plot(
        head_signal * 100,
        default_error,
        color=BLUE,
        linewidth=3,
        linestyle="--",
        label="DEFAULT_IDENTITY / rotation-only baseline",
    )
    ax.plot(
        head_signal * 100,
        rotation_only,
        color=GRAY,
        linewidth=2,
        linestyle=":",
        label="CALIBRATED_EIGENBASIS, weighting off",
    )
    ax.plot(
        head_signal * 100,
        weighted_error,
        color=RED,
        linewidth=4,
        label="CALIBRATED_EIGENBASIS, eigenvalue weighting on",
    )
    ax.fill_between(head_signal * 100, weighted_error, default_error, where=weighted_error > default_error, color=RED, alpha=0.14, interpolate=True)
    ax.fill_between(head_signal * 100, weighted_error, default_error, where=weighted_error < default_error, color=GREEN, alpha=0.16, interpolate=True)
    ax.set_title("Error Source 6: spectral-bias tradeoff", fontsize=16, pad=14)
    ax.set_xlabel("Share of retrieval signal in high-variance eigendirections", fontsize=12)
    ax.set_ylabel("Relative FDE reconstruction error", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.5, 1.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%\nall tail", "25%", "50%", "75%", "100%\nall head"])
    ax.legend(loc="upper right")
    ax.annotate(
        "Tail-heavy corpora:\nsmall text, forms,\nrare visual details",
        xy=(10, 1.30),
        xytext=(6, 1.43),
        arrowprops={"arrowstyle": "->", "color": "#B33A3D"},
        fontsize=11,
        color="#B33A3D",
    )
    ax.annotate(
        "Head-heavy corpora:\nstable semantic axes",
        xy=(86, 0.73),
        xytext=(66, 0.58),
        arrowprops={"arrowstyle": "->", "color": "#2F7D4F"},
        fontsize=11,
        color="#2F7D4F",
    )
    ax.text(
        0.02,
        0.02,
        "Synthetic intuition, not a benchmark. Measure against exact MaxSim.",
        transform=ax.transAxes,
        fontsize=10,
        color="#374151",
    )

    fig.suptitle(
        "CALIBRATED_EIGENBASIS can trade principal-subspace error for tail-detail error",
        fontsize=18,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3.0)
    return _save(fig, output_dir / "plot7_eigenbasis_spectral_bias.png")


PLOTS: dict[str, Callable[[Path], Path]] = {
    "plot1": plot1_variance_vs_repetitions,
    "plot2": plot2_tokens_per_partition,
    "plot3": plot3_fill_cost_comparison,
    "plot4": plot4_eggroll_vs_clt,
    "plot5": plot5_two_stage_recall,
    "plot6": plot6_error_breakdown,
    "plot7": plot7_eigenbasis_spectral_bias,
}


def generate_plots(output_dir: Path = DEFAULT_OUTPUT_DIR, only: list[str] | None = None) -> list[Path]:
    _style()
    selected = only or list(PLOTS)
    return [PLOTS[name](output_dir) for name in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plot PNGs should be written. Defaults to docs/images.",
    )
    parser.add_argument(
        "--only",
        choices=list(PLOTS),
        nargs="+",
        help="Generate only the selected plot ids, for example: --only plot6 plot7.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in generate_plots(args.output_dir, args.only):
        print(path)


if __name__ == "__main__":
    main()
