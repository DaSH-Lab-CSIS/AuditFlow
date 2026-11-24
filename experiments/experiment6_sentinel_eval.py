"""
Experiment 6 — Sentinel Detection Effectiveness (EQ6)
========================================================
Runs each workflow's normal and failure scripts N times via subprocess,
reading the real traces/sentinel_alerts.jsonl written by the live
Sentinel (wired into log_events.py) after each run.

Reports, per workflow:
  - false-positive rate (alerts fired on normal runs — should be ~0)
  - true-positive rate  (alerts fired on failure runs — should be high)
  - detection latency   (seq number of the first alert)

Usage:
    python experiment6_sentinel_eval.py

Results saved to:
    results/sentinel/{workflow}_sentinel_n30.json
    results/sentinel/all_workflows_sentinel_n30.json
"""

import json
import subprocess
import sys
import time
import os   
from pathlib import Path

BASE = Path(__file__).parent.parent
N_RUNS = 30

WORKFLOWS = {
    "W1_email": {
        "dir": BASE / "auditflow_w1",
        "normal_script": "run_agent_interceptor.py",
        "failure_script": "run_agent_failure.py",
    },
    "W2_codereview": {
        "dir": BASE / "auditflow_w2",
        "normal_script": "run_agent_interceptor.py",
        "failure_script": "run_agent_failure.py",
    },
    "W3_research": {
        "dir": BASE / "auditflow_w3",
        "normal_script": "run_agent_original.py",
        "failure_script": "run_agent_failure.py",
    },
    "W4_coding": {
        "dir": BASE / "auditflow_w4",
        "normal_script": "run_agent_interceptor.py",
        "failure_script": "run_failure_interceptor.py",
    },
}


def run_once(workflow_dir: Path, script: str) -> dict:
    alerts_file = workflow_dir / "traces" / "sentinel_alerts.jsonl"
    if alerts_file.exists():
        alerts_file.unlink()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, script],
        cwd=str(workflow_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=300,
    )
    t1 = time.perf_counter()

    alerts = []
    if alerts_file.exists():
        with open(alerts_file) as f:
            alerts = [json.loads(line) for line in f if line.strip()]

    return {
        "returncode": result.returncode,
        "wall_time_s": round(t1 - t0, 2),
        "alerts": alerts,
        "stderr_tail": result.stderr[-500:] if result.returncode != 0 else "",
    }


def run_condition(wf_name: str, wf_cfg: dict, condition: str, script: str, n: int) -> list:
    print(f"\n  [{wf_name}] {condition} — {n} trials")
    runs = []
    for i in range(n):
        r = run_once(wf_cfg["dir"], script)
        runs.append(r)
        fired = [a["alert_type"] for a in r["alerts"]]
        status = ("FIRED: " + ",".join(fired)) if fired else "silent"
        err = "  [ERROR]" if r["returncode"] != 0 else ""
        print(f"    run {i+1:02d}/{n}  {status}  ({r['wall_time_s']}s){err}")
    return runs


def summarize(runs: list, label: str) -> dict:
    n = len(runs)
    fired_runs = [r for r in runs if r["alerts"]]
    errors = [r for r in runs if r["returncode"] != 0]
    latencies = [r["alerts"][0]["seq"] for r in fired_runs if r["alerts"][0].get("seq") is not None]

    return {
        "condition": label,
        "n": n,
        "n_fired": len(fired_runs),
        "fire_rate": round(len(fired_runs) / n, 3) if n else 0,
        "n_errors": len(errors),
        "alert_types_seen": sorted({a["alert_type"] for r in fired_runs for a in r["alerts"]}),
        "mean_detection_seq": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }


def main():
    out_dir = BASE / "experiments" / "results" / "sentinel"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for wf_name, wf_cfg in WORKFLOWS.items():
        print(f"\n{'='*65}\n  {wf_name}\n{'='*65}")

        normal_runs = run_condition(wf_name, wf_cfg, "normal", wf_cfg["normal_script"], N_RUNS)
        failure_runs = run_condition(wf_name, wf_cfg, "failure", wf_cfg["failure_script"], N_RUNS)

        normal_summary = summarize(normal_runs, "normal")
        failure_summary = summarize(failure_runs, "failure")

        print(f"\n  normal : {normal_summary['n_fired']}/{N_RUNS} fired "
              f"(false positives — want 0)")
        print(f"  failure: {failure_summary['n_fired']}/{N_RUNS} fired "
              f"(true positives), types={failure_summary['alert_types_seen']}")

        all_results[wf_name] = {
            "normal": {"summary": normal_summary, "runs": normal_runs},
            "failure": {"summary": failure_summary, "runs": failure_runs},
        }

        with open(out_dir / f"{wf_name}_sentinel_n{N_RUNS}.json", "w") as f:
            json.dump(all_results[wf_name], f, indent=2)

    with open(out_dir / f"all_workflows_sentinel_n{N_RUNS}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  All results saved → {out_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()