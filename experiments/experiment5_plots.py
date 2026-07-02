"""
Experiment 5 — Storage Plots (v2)
===================================
Fixes applied per supervisor feedback:
  - Consistent FONT = 11 (matches experiment1_plots_v5.py)
  - Minor grid lines on Y axis (log scale)
  - Y axis label includes "(log scale)"
  - Legend moved so it never overlaps bar tips
  - "Workflow Run" x-axis label italicised on figB
  - Normal/Failure x-tick labels coloured green/red on figB and figC
  - figC legend placed above plot in 2 rows
"""

import json
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.ticker import LogLocator, AutoMinorLocator
from pathlib import Path

PLOTS_DIR = Path("results/storage/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Single source of truth — must match experiment1_plots_v5.py ───────────
FONT = 12.5

matplotlib.rcParams["hatch.linewidth"] = 1.5

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          FONT,
    "font.weight":        "bold",
    "axes.titlesize":     FONT,
    "axes.labelsize":     FONT,
    "axes.titleweight":   "bold",
    "axes.labelweight":   "bold",
    "xtick.labelsize":    FONT,
    "ytick.labelsize":    FONT,
    "legend.fontsize":    FONT,
    "figure.dpi":         200,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "gray",
})


def _apply_log_gridlines(ax, axis="y"):
    """Add major + minor gridlines for a log-scale axis."""
    if axis == "y":
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(0.1, 1.0, 0.1), numticks=10))
        ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.15)
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.25)
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.15)
        ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.25)


def _save(fig, name):
    for ext in ["pdf", "png"]:
        path = PLOTS_DIR / f"{name}.{ext}"
        try:
            fig.savefig(path, bbox_inches="tight",
                        dpi=200 if ext == "png" else None)
            print(f"  Saved  -> {path.name}")
        except PermissionError:
            print(f"  SKIPPED -> {path.name}  (close the file and rerun)")
    plt.close(fig)


# ── Fig B: IPFS trace size vs blockchain anchor ───────────────────────────
def figB_storage_growth():
    fig, ax = plt.subplots(figsize=(8, 4))

    runs = ["W1\nNormal", "W1\nFailure",
            "W2\nNormal", "W2\nFailure",
            "W3\nNormal", "W3\nFailure",
            "W4\nNormal", "W4\nFailure"]

    ipfs_vals  = [4.497, 7.385, 4.023, 4.231,
                  7.474, 7.480, 4.604, 10.119]
    onchain_kb = [115 / 1024] * 8

    x  = np.arange(len(runs))
    bw = 0.35

    ax.bar(x - bw/2, ipfs_vals, bw,
           color="#3A7EBF", alpha=0.85,
           hatch="///", edgecolor="black",
           label="IPFS (full trace)")
    ax.bar(x + bw/2, onchain_kb, bw,
           color="#E07B3A", alpha=0.85,
           hatch="xxx", edgecolor="black",
           label="Blockchain anchor (Merkle root)")

    ax.set_xticks(x)
    ax.set_xticklabels(runs)

    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color("#2E7D32" if i % 2 == 0 else "#C62828")

    ax.set_xlabel("Workflow Run", style="italic")
    ax.set_ylabel("Storage size (KB)\n(log scale)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)

    _apply_log_gridlines(ax, axis="y")

    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
                    ncol=2, borderaxespad=0)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    _save(fig, "figB_storage_growth")


# ── Fig C: Intelligent anchoring — naive vs tiered policy ─────────────────
def figC_intelligent_anchoring():
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = ["W1\nNormal", "W1\nFailure",
              "W2\nNormal", "W2\nFailure",
              "W3\nNormal", "W3\nFailure",
              "W4\nNormal", "W4\nFailure"]

    naive_kb = [4.497, 7.385, 4.023, 4.231,
                7.474, 7.480, 4.604, 10.119]

    anchor_bytes = [35, 231, 35, 35, 35, 35, 35, 57]
    anchor_kb    = [b / 1024 for b in anchor_bytes]
    policies     = ["MINIMAL", "FULL",    "MINIMAL", "MINIMAL",
                    "MINIMAL", "MINIMAL", "MINIMAL", "STANDARD"]

    policy_colors = {
        "MINIMAL":  "#4CAF50",
        "STANDARD": "#FF9800",
        "FULL":     "#E53935",
    }
    bar_colors = [policy_colors[p] for p in policies]

    x  = np.arange(len(labels))
    bw = 0.35

    ax.bar(x - bw/2, naive_kb, bw,
           color="#6A1B9A", alpha=0.80,
           hatch="xxx", edgecolor="black",
           label="Naive (baseline)")
    ax.bar(x + bw/2, anchor_kb, bw,
           color=bar_colors, alpha=0.85,
           hatch="///", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color("#2E7D32" if i % 2 == 0 else "#C62828")

    ax.set_ylabel("On-chain storage (KB) [log]")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)

    _apply_log_gridlines(ax, axis="y")

    p_nai = mpatches.Patch(facecolor="#6A1B9A", alpha=0.80,
                            hatch="xxx", edgecolor="black",
                            linewidth=1, label="Naive (baseline)")
    p_min = mpatches.Patch(facecolor="#4CAF50", alpha=0.85,
                            hatch="///", edgecolor="black",
                            linewidth=1, label="MINIMAL (score=0)")
    p_std = mpatches.Patch(facecolor="#FF9800", alpha=0.85,
                            hatch="///", edgecolor="black",
                            linewidth=1, label="STANDARD (score 1-3)")
    p_ful = mpatches.Patch(facecolor="#E53935", alpha=0.85,
                            hatch="///", edgecolor="black",
                            linewidth=1, label="FULL (score 4+)")

    leg = ax.legend(handles=[p_nai, p_min, p_std, p_ful],
                    loc="lower left", bbox_to_anchor=(0.0, 1.02),
                    ncol=2, borderaxespad=0)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    _save(fig, "figC_intelligent_anchoring")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Storage Plots v2")
    print("="*55 + "\n")
    figB_storage_growth()
    figC_intelligent_anchoring()
    print(f"\n  Saved to: {PLOTS_DIR}/")
    print("="*55)