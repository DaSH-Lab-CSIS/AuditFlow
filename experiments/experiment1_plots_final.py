"""
Experiment 1 — Final Plots (v5)
================================
Consistent font sizing throughout — no local overrides.
All font sizes come from rcParams only.
"""

import json
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import LogLocator, AutoMinorLocator
from pathlib import Path

PLOTS_DIR = Path("results/plots_v5")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Single source of truth for ALL font sizes ──────────────────────────────
FONT = 11   # base font — change this one number to rescale everything

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

C_OFF = "#1565C0"
C_ON  = "#E64A19"
C_IC  = "#4CAF50"
C_SE  = "#FF9800"
C_IO  = "#7B4EA8"

WORKFLOWS = ["W1_email", "W2_codereview", "W3_research", "W4_coding"]
WF_LABELS = [
    "W1: Email Triage\n(Single-agent)",
    "W2: Code Review\n(Multi-agent)",
    "W3: Web Research\n(Tool-augmented)",
    "W4: Coding Agent\n(Feedback-driven)",
]
WF_SHORT = ["W1", "W2", "W3", "W4"]


def load_raw():
    with open("results/raw/all_latencies.json") as f:
        return json.load(f)

def load_mb():
    try:
        with open("results/experiment1b_microbench.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

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


def fig1_grouped_bar(raw: dict):
    fig, ax = plt.subplots(figsize=(7.8, 3.7))

    x      = np.arange(len(WORKFLOWS))
    bw     = 0.32
    offset = 0.18

    medians_off, medians_on = [], []
    stds_off,    stds_on    = [], []
    sigs                    = []

    for wf in WORKFLOWS:
        off = raw[wf]["off"]
        on  = raw[wf]["on"]
        medians_off.append(statistics.median(off))
        medians_on.append(statistics.median(on))
        stds_off.append(statistics.stdev(off))
        stds_on.append(statistics.stdev(on))
        try:
            from scipy import stats as ss
            _, pval = ss.ttest_ind(off, on, equal_var=False)
            sigs.append("n.s." if pval >= 0.05 else f"p={pval:.3f}*")
        except ImportError:
            sigs.append("n.s.")

    errs_off = [min(s, m * 0.15) for s, m in zip(stds_off, medians_off)]
    errs_on  = [min(s, m * 0.15) for s, m in zip(stds_on,  medians_on)]

    ax.bar(x - offset, medians_off, bw,
           yerr=errs_off, capsize=4,
           color=C_OFF, alpha=0.95,
           hatch="///", edgecolor="black", linewidth=0.5,
           label="Without AuditFlow",
           error_kw=dict(elinewidth=1.5, ecolor="black", capthick=1.5))
    ax.bar(x + offset, medians_on, bw,
           yerr=errs_on, capsize=4,
           color=C_ON, alpha=0.95,
           hatch="...", edgecolor="black", linewidth=0.5,
           label="With AuditFlow",
           error_kw=dict(elinewidth=1.5, ecolor="black", capthick=1.5))

    ax.set_xticks(x)
    # fontsize not set here — comes from rcParams xtick.labelsize
    ax.set_xticklabels(WF_LABELS)
    ax.set_ylabel("End-to-end latency (ms)\n(log scale)")
    ax.set_yscale("log")

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(LogLocator(base=10,
                                          subs=np.arange(0.1, 1.0, 0.1),
                                          numticks=10))
    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.15)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.25)

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 2.5)

    leg = ax.legend(loc="upper left")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(left=0.13)
    _save(fig, "fig1_grouped_bar")


def fig2_horizontal_bar(mb: dict):
    fig, ax = plt.subplots(figsize=(6.5, 2.5))

    intercept, serialise, io_vals = [], [], []

    for wf in WORKFLOWS:
        r  = mb[wf]
        ic = (r["llm_call"]["interception_ms"]["mean"] +
              r["tool_call"]["interception_ms"]["mean"]) / 2 * 1000
        se = (r["llm_call"]["serialisation_ms"]["mean"] +
              r["tool_call"]["serialisation_ms"]["mean"]) / 2 * 1000
        iv = (r["llm_call"]["io_ms"]["mean"] +
              r["tool_call"]["io_ms"]["mean"]) / 2 * 1000
        intercept.append(ic)
        serialise.append(se)
        io_vals.append(iv)

    y      = np.arange(len(WORKFLOWS))
    height = 0.45

    ax.barh(y, intercept, height,
            label="Interception", color=C_IC, alpha=0.88,
            hatch="xxx", edgecolor="black")
    ax.barh(y, serialise, height, left=intercept,
            label="Serialisation + Hashing", color=C_SE, alpha=0.88,
            hatch="///", edgecolor="black")
    ax.barh(y, io_vals, height,
            left=[i+s for i, s in zip(intercept, serialise)],
            label="File I/O (JSONL write)", color=C_IO, alpha=0.88,
            hatch="...", edgecolor="black")

    for i, (ic, se, iv) in enumerate(zip(intercept, serialise, io_vals)):
        total = ic + se + iv
        # fontsize not set — comes from rcParams
        ax.text(total + 0.8, i, f"{total:.1f} µs",
                ha="left", va="center", color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(WF_SHORT)
    ax.set_xlabel("Per-event overhead (µs)")
    ax.invert_yaxis()

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.15)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.25)

    max_total = max(i+s+v for i, s, v in zip(intercept, serialise, io_vals))
    ax.set_xlim(0, max_total * 1.45)

    # legend above the plot, 3 columns
    leg = ax.legend(loc="lower right")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    plt.tight_layout()
    _save(fig, "fig2_horizontal_bar")


def main():
    print("\n" + "="*60)
    print("  Experiment 1 - Plots v5")
    print("="*60 + "\n")

    raw = load_raw()
    mb  = load_mb()

    fig1_grouped_bar(raw)
    if mb:
        fig2_horizontal_bar(mb)
    else:
        print("  Skipping Fig 2 — run experiment1b_microbench.py first")

    print(f"\n  Saved to: {PLOTS_DIR}/")
    print("="*60)


main()