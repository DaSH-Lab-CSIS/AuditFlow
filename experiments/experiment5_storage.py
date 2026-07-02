"""
Experiment 5 — Storage Cost and Intelligent Anchoring (RQ5)
=============================================================
Measures:
  1. Trace size per workflow (bytes per event, total trace size)
  2. On-chain vs off-chain storage split (Merkle root vs full trace)
  3. Intelligent anchoring policy — how much on-chain storage is saved
     for low-risk runs vs naive full anchoring

Reads existing JSONL trace files from W1-W4.
No LLM needed — runs in seconds.

Usage:
    python experiment5_storage.py

Results saved to:
    results/storage/storage_results.json
    results/storage/storage_results.csv
    results/storage/plots/
"""

import json
import hashlib
import csv
import statistics
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Point these at your actual trace files
TRACE_FILES = {
    "W1_normal":  "../auditflow_w1/traces/trace_normal.jsonl",
    "W1_failure": "../auditflow_w1/traces/trace_failure.jsonl",
    "W2_normal":  "../auditflow_w2/traces/trace_normal.jsonl",
    "W2_failure": "../auditflow_w2/traces/trace_failure.jsonl",
    "W3_normal":  "../auditflow_w3/traces/trace_normal.jsonl",
    "W3_failure": "../auditflow_w3/traces/trace_failure.jsonl",
    "W4_normal":  "../auditflow_w4/traces/trace_normal.jsonl",
    "W4_failure": "../auditflow_w4/traces/trace_failure.jsonl",
}

RESULTS_DIR = Path("results/storage")
PLOTS_DIR   = Path("results/storage/plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

WF_DISPLAY = {
    "W1_normal":  "W1 Normal",
    "W1_failure": "W1 Failure",
    "W2_normal":  "W2 Normal",
    "W2_failure": "W2 Failure",
    "W3_normal":  "W3 Normal",
    "W3_failure": "W3 Failure",
    "W4_normal":  "W4 Normal",
    "W4_failure": "W4 Failure",
}


# ── Load trace ────────────────────────────────────────────────────────────────

def load_trace(path: str) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ── Storage measurements ──────────────────────────────────────────────────────

def measure_trace_storage(events: list[dict]) -> dict:
    """Measure storage costs for a single trace."""

    # per-event sizes
    event_sizes = []
    event_sizes_by_type = {}

    for e in events:
        size = len(json.dumps(e, default=str).encode("utf-8"))
        event_sizes.append(size)
        etype = e.get("type", "UNKNOWN")
        event_sizes_by_type.setdefault(etype, []).append(size)

    total_bytes = sum(event_sizes)

    # per-type averages
    per_type = {
        etype: round(statistics.mean(sizes), 1)
        for etype, sizes in event_sizes_by_type.items()
    }

    # Merkle root computation (on-chain)
    hashes = [e.get("hash", "").encode() or
              hashlib.sha256(json.dumps(e, default=str)
                             .encode()).hexdigest()[:16].encode()
              for e in events]

    # compute Merkle root
    merkle_root = _compute_merkle_root(hashes)
    merkle_bytes = len(merkle_root.encode("utf-8"))

    # on-chain record (Merkle root + run_id + timestamp)
    run_id    = events[0].get("run_id", "unknown") if events else "unknown"
    on_chain  = {
        "merkle_root": merkle_root,
        "run_id":      run_id,
        "timestamp":   events[0].get("timestamp", 0) if events else 0,
    }
    on_chain_bytes = len(json.dumps(on_chain).encode("utf-8"))

    # IPFS = full trace (off-chain)
    ipfs_bytes = total_bytes

    # savings ratio
    savings_pct = (1 - on_chain_bytes / ipfs_bytes) * 100 if ipfs_bytes else 0

    return {
        "n_events":        len(events),
        "total_bytes":     total_bytes,
        "total_kb":        round(total_bytes / 1024, 3),
        "bytes_per_event": round(statistics.mean(event_sizes), 1)
                           if event_sizes else 0,
        "per_type_bytes":  per_type,
        "ipfs_bytes":      ipfs_bytes,
        "ipfs_kb":         round(ipfs_bytes / 1024, 3),
        "on_chain_bytes":  on_chain_bytes,
        "merkle_bytes":    merkle_bytes,
        "savings_pct":     round(savings_pct, 2),
        "merkle_root":     merkle_root,
    }


def _compute_merkle_root(hashes: list[bytes]) -> str:
    """Compute Merkle root from list of hash bytes."""
    if not hashes:
        return ""
    layer = [h if isinstance(h, bytes) else h.encode() for h in hashes]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256(layer[i] + layer[i+1]).hexdigest()[:16].encode()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].decode() if layer else ""


# ── Intelligent anchoring policy ──────────────────────────────────────────────

def compute_risk_score(events: list[dict]) -> tuple[int, list[str]]:
    """
    Compute a risk score for the run based on audit flags.
    Returns (score, reasons).

    Scoring:
      IRREVERSIBLE_ACTION flag  → +3
      WARNING flag              → +2
      FAILED_MAX_ITER status    → +2
      Unexpected event count    → +1
    """
    score   = 0
    reasons = []

    for e in events:
        meta = e.get("metadata", {})
        out  = e.get("output", {})

        # check metadata flags
        if meta.get("AUDIT_FLAG") == "IRREVERSIBLE_ACTION":
            score += 3
            reasons.append("IRREVERSIBLE_ACTION flag in metadata")

        if meta.get("WARNING"):
            score += 2
            reasons.append(f"WARNING: {meta['WARNING'][:60]}")

        # check anywhere in the full event string for irreversible
        event_str = json.dumps(e, default=str).lower()
        if "irreversible" in event_str and e.get("type") == "TOOL_CALL":
            score += 3
            reasons.append("Irreversible action detected in tool output")

        if isinstance(out, dict):
            if "FAILED_MAX_ITER" in str(out):
                score += 2
                reasons.append("FAILED_MAX_ITER detected")

        # check output string directly
        if "irreversible" in str(out).lower():
            score += 3
            reasons.append("Irreversible warning in output payload")

    # unexpected event count (normal W1=7, W2=5, W3=8, W4=7)
    if len(events) > 10:
        score += 1
        reasons.append(f"High event count ({len(events)})")

    return score, list(set(reasons))


def anchoring_policy(risk_score: int,
                     events: list[dict]) -> dict:
    """
    Intelligent anchoring policy based on risk score.

    Score 0   → Minimal: Merkle root only
    Score 1-3 → Standard: Merkle root + flagged event hashes
    Score 4+  → Full: Merkle root + all individual event hashes
    """
    hashes     = [e.get("hash", "") for e in events]
    merkle     = _compute_merkle_root(
        [h.encode() for h in hashes if h])
    merkle_b   = len(merkle.encode())

    flagged    = [e for e in events
                  if e.get("metadata", {}).get("AUDIT_FLAG") or
                     e.get("metadata", {}).get("WARNING") or
                     "IRREVERSIBLE" in str(e.get("output", {}))]

    if risk_score == 0:
        policy     = "MINIMAL"
        on_chain   = {"merkle_root": merkle}
        description = "Merkle root only"

    elif risk_score <= 3:
        policy     = "STANDARD"
        on_chain   = {
            "merkle_root":    merkle,
            "flagged_hashes": [e.get("hash","") for e in flagged],
        }
        description = f"Merkle root + {len(flagged)} flagged event hashes"

    else:
        policy     = "FULL"
        on_chain   = {
            "merkle_root":  merkle,
            "all_hashes":   hashes,
        }
        description = f"Merkle root + all {len(hashes)} event hashes"

    on_chain_bytes = len(json.dumps(on_chain).encode())

    return {
        "policy":          policy,
        "description":     description,
        "risk_score":      risk_score,
        "on_chain_bytes":  on_chain_bytes,
        "on_chain_kb":     round(on_chain_bytes / 1024, 4),
        "n_flagged":       len(flagged),
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment():
    print("\n" + "="*65)
    print("  Experiment 5 — Storage Cost and Intelligent Anchoring (RQ5)")
    print("="*65 + "\n")

    all_results = {}

    for name, path in TRACE_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {name} — trace not found at {path}")
            continue

        events = load_trace(path)
        if not events:
            print(f"  SKIP {name} — empty trace")
            continue

        storage   = measure_trace_storage(events)
        score, reasons = compute_risk_score(events)
        anchoring = anchoring_policy(score, events)

        result = {
            "name":      name,
            "display":   WF_DISPLAY.get(name, name),
            "storage":   storage,
            "risk":      {"score": score, "reasons": reasons},
            "anchoring": anchoring,
        }
        all_results[name] = result

        print(f"  {name}")
        print(f"    Events        : {storage['n_events']}")
        print(f"    IPFS (total)  : {storage['ipfs_kb']:.3f} KB")
        print(f"    Bytes/event   : {storage['bytes_per_event']:.0f} B")
        print(f"    On-chain      : {storage['on_chain_bytes']} B "
              f"(Merkle root)")
        print(f"    Savings       : {storage['savings_pct']:.1f}%")
        print(f"    Risk score    : {score} → policy={anchoring['policy']}")
        if reasons:
            for r in reasons:
                print(f"      ↳ {r}")
        print(f"    Smart anchor  : {anchoring['description']}")
        print(f"    Anchor size   : {anchoring['on_chain_bytes']} B")
        print()

    # save JSON
    json_path = RESULTS_DIR / "storage_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # save CSV
    csv_path = RESULTS_DIR / "storage_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Run", "Events", "IPFS_KB", "Bytes_per_event",
                    "OnChain_bytes", "Savings_pct", "Risk_score",
                    "Policy", "Anchor_bytes"])
        for name, r in all_results.items():
            w.writerow([
                name,
                r["storage"]["n_events"],
                r["storage"]["ipfs_kb"],
                r["storage"]["bytes_per_event"],
                r["storage"]["on_chain_bytes"],
                r["storage"]["savings_pct"],
                r["risk"]["score"],
                r["anchoring"]["policy"],
                r["anchoring"]["on_chain_bytes"],
            ])

    print(f"  Results saved → {json_path}")
    print(f"  Results saved → {csv_path}")

    # summary table
    print("\n" + "="*65)
    print("  STORAGE SUMMARY TABLE")
    print("="*65)
    print(f"  {'Run':<18} {'Events':>7} {'IPFS(KB)':>10} "
          f"{'OnChain(B)':>12} {'Saving%':>9} {'Policy':<10}")
    print(f"  {'-'*65}")
    for name, r in all_results.items():
        print(f"  {name:<18} "
              f"{r['storage']['n_events']:>7} "
              f"{r['storage']['ipfs_kb']:>10.3f} "
              f"{r['storage']['on_chain_bytes']:>12} "
              f"{r['storage']['savings_pct']:>9.1f}% "
              f"{r['anchoring']['policy']:<10}")

    print("\n  KEY FINDING:")
    if all_results:
        on_chain_sizes = [r["storage"]["on_chain_bytes"]
                          for r in all_results.values()]
        ipfs_sizes     = [r["storage"]["ipfs_kb"]
                          for r in all_results.values()]
        print(f"  On-chain size is constant: "
              f"{min(on_chain_sizes)}–{max(on_chain_sizes)} bytes "
              f"regardless of trace size")
        print(f"  IPFS size varies: "
              f"{min(ipfs_sizes):.3f}–{max(ipfs_sizes):.3f} KB "
              f"with workflow complexity")
        print(f"  Average on-chain savings: "
              f"{statistics.mean([r['storage']['savings_pct'] for r in all_results.values()]):.1f}%")

    return all_results


# ── Plots ─────────────────────────────────────────────────────────────────────

def generate_plots(all_results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       18,
        "axes.labelsize":  18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.grid":       True,
        "grid.alpha":      0.25,
        "grid.linestyle":  "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    names    = list(all_results.keys())
    labels   = [WF_DISPLAY.get(n, n).replace("\n", " ") for n in names]
    ipfs_kb  = [all_results[n]["storage"]["ipfs_kb"]        for n in names]
    on_chain = [all_results[n]["storage"]["on_chain_bytes"] for n in names]
    n_events = [all_results[n]["storage"]["n_events"]       for n in names]
    anchor_b = [all_results[n]["anchoring"]["on_chain_bytes"] for n in names]
    policies = [all_results[n]["anchoring"]["policy"]        for n in names]

    # colour by policy
    policy_colors = {"MINIMAL": "#4CAF50", "STANDARD": "#FF9800",
                     "FULL": "#E53935"}
    bar_colors = [policy_colors.get(p, "gray") for p in policies]

    # ── Fig A: IPFS trace size vs on-chain size ────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    x  = np.arange(len(names))
    bw = 0.35

    # IPFS in KB, on-chain in bytes converted to KB for comparison
    on_chain_kb = [b / 1024 for b in on_chain]

    b1 = ax.bar(x - bw/2, ipfs_kb, bw,
                color="#3A7EBF", alpha=0.85,
                hatch="///", edgecolor="black",
                label="IPFS (full trace)")
    b2 = ax.bar(x + bw/2, on_chain_kb, bw,
                color="#E07B3A", alpha=0.85,
                hatch="...", edgecolor="black",
                label="Blockchain (Merkle root only)")

    # value labels
    for i, (ik, ok) in enumerate(zip(ipfs_kb, on_chain_kb)):
        ax.text(i - bw/2, ik * 1.02, f"{ik:.2f}KB",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#3A7EBF")
        ax.text(i + bw/2, ok * 1.02 + 0.001,
                f"{on_chain[i]}B",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#E07B3A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold",
                       rotation=15, ha="right")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylabel("Storage size (KB)", fontsize=18, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)

    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    leg = ax.legend(fontsize=14, framealpha=0.9)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    # ax.set_title(
    #     "Figure A: IPFS Trace Size vs Blockchain Anchor Size\n"
    #     "(log scale; blockchain stores only Merkle root — O(1) storage)",
    #     fontsize=13, fontweight="bold", pad=10
    # )
    plt.tight_layout()
    _save_plot(fig, "figA_storage_comparison")

    # # ── Fig B: Storage growth vs event count ──────────────────────────────
    # fig, ax = plt.subplots(figsize=(8, 5))

    # # scatter: events vs IPFS size
    # ax.scatter(n_events, ipfs_kb,
    #            color="#3A7EBF", s=120, zorder=5,
    #            label="IPFS trace size")

    # # fit line
    # if len(n_events) > 2:
    #     coeffs = np.polyfit(n_events, ipfs_kb, 1)
    #     fit_x  = np.linspace(min(n_events)-1, max(n_events)+1, 100)
    #     fit_y  = np.polyval(coeffs, fit_x)
    #     ax.plot(fit_x, fit_y, color="#3A7EBF",
    #             linestyle="--", linewidth=2, alpha=0.7,
    #             label=f"Linear fit (slope={coeffs[0]:.3f} KB/event)")

    # # on-chain line (constant)
    # avg_on_chain_kb = statistics.mean(on_chain_kb)
    # ax.axhline(y=avg_on_chain_kb, color="#E07B3A",
    #            linewidth=2.5, linestyle="-",
    #            label=f"Blockchain anchor ({int(statistics.mean(on_chain))}B constant)")

    

    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_fontweight("bold")
    # ax.set_xlabel("Number of events in trace",
    #               fontsize=18, fontweight="bold")
    # ax.set_ylabel("Storage size (KB)", fontsize=18, fontweight="bold")

    # leg = ax.legend(fontsize=12, framealpha=0.9)
    # for t in leg.get_texts():
    #     t.set_fontweight("bold")

   
    # plt.tight_layout()
    # _save_plot(fig, "figB_storage_growth")

    # ── Fig B: Storage size per run — grouped bar ─────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    runs = ["W1\nNormal", "W1\nFailure", "W2\nNormal", "W2\nFailure",
            "W3\nNormal", "W3\nFailure", "W4\nNormal", "W4\nFailure"]

    ipfs_vals     = [4.497, 7.385, 4.023, 4.231,
                     7.474, 7.480, 4.604, 10.119]
    onchain_kb    = [115/1024] * 8   # constant 114-115 bytes in KB

    x  = np.arange(len(runs))
    bw = 0.35

    b1 = ax.bar(x - bw/2, ipfs_vals, bw,
                color="#3A7EBF", alpha=0.85,
                hatch="///", edgecolor="black",
                label="IPFS (full trace)")

    b2 = ax.bar(x + bw/2, onchain_kb, bw,
                color="#E07B3A", alpha=0.85,
                hatch="xxx", edgecolor="black",
                label="Blockchain anchor (Merkle root)")

    # # value labels on IPFS bars only
    # for i, v in enumerate(ipfs_vals):
    #     ax.text(i - bw/2, v + 0.1, f"{v:.2f}",
    #             ha="center", va="bottom",
    #             fontsize=9, fontweight="bold",
    #             color="#3A7EBF")

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=18, fontweight="bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(13)

    ax.set_ylabel("Storage size (KB) (log scale)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Workflow Run", fontsize=18, fontweight="bold")
    ax.set_ylim(0, 12)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)

    leg = ax.legend(fontsize=13, framealpha=0.9)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    # ax.set_title(
    #     "IPFS Trace Size vs Blockchain Anchor Size\n"
    #     "Blockchain stores only Merkle root — O(1) on-chain storage",
    #     fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    _save_plot(fig, "figB_storage_growth")

    # ── Fig C: Intelligent anchoring — naive vs smart ──────────────────────
    fig, ax = plt.subplots(figsize=(15, 6))

    naive_kb  = ipfs_kb          # naive: store everything on-chain
    smart_kb  = [b / 1024 for b in anchor_b]  # smart: tiered policy

    b1 = ax.bar(x - bw/2, naive_kb, bw,
                color="#6A1B9A", alpha=0.80,
                hatch="xxx", edgecolor="black",
                label="Naive (full trace on-chain)")
    b2 = ax.bar(x + bw/2, smart_kb, bw,
                color=bar_colors, alpha=0.85,
                hatch="///", edgecolor="black",
                label="Intelligent anchoring (tiered policy)")

    # for i, (nk, sk, p) in enumerate(zip(naive_kb, smart_kb, policies)):
    #     saving = (1 - sk/nk) * 100 if nk else 0
    #     ax.text(i + bw/2, sk * 1.05,
    #             f"{p}\n({saving:.0f}% saved)",
    #             ha="center", va="bottom",
    #             fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, fontweight="bold",
                       rotation=15, ha="right")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_ylabel("On-chain storage (KB)", fontsize=20, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    # policy legend patches
    import matplotlib.patches as mpatches
    p_min = mpatches.Patch(color="#4CAF50", alpha=0.85,
                            label="MINIMAL (score=0)")
    p_std = mpatches.Patch(color="#FF9800", alpha=0.85,
                            label="STANDARD (score 1-3)")
    p_ful = mpatches.Patch(color="#E53935", alpha=0.85,
                            label="FULL (score 4+)")
    p_nai = mpatches.Patch(color="#6A1B9A", alpha=0.80,
                            hatch="xxx", label="Naive (baseline)")
    leg = ax.legend(handles=[p_nai, p_min, p_std, p_ful],
                    fontsize=14, framealpha=0.9)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    # ax.set_title(
    #     "Figure C: Intelligent Anchoring vs Naive On-Chain Storage\n"
    #     "Tiered policy reduces on-chain cost based on run risk score",
    #     fontsize=13, fontweight="bold", pad=10
    # )
    plt.tight_layout()
    _save_plot(fig, "figC_intelligent_anchoring")

    print(f"\n  Plots saved to: {PLOTS_DIR}/")


def _save_plot(fig, name: str):
    import matplotlib.pyplot as plt
    for ext in ["pdf", "png"]:
        path = PLOTS_DIR / f"{name}.{ext}"
        try:
            fig.savefig(path, bbox_inches="tight",
                        dpi=200 if ext == "png" else None)
            print(f"  Saved  → {path.name}")
        except PermissionError:
            print(f"  SKIPPED → {path.name} (close the file first)")
    plt.close(fig)


if __name__ == "__main__":
    results = run_experiment()
    generate_plots(results)