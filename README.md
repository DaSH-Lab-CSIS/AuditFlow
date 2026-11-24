# AuditFlow

AuditFlow is a lightweight, provenance-aware audit middleware for agentic workflows. It captures execution events at runtime, reconstructs each run as a causal Directed Acyclic Graph (DAG), and supports post-hoc failure attribution, cross-run comparison and tamper-evident integrity verification. An optional rule-based runtime monitor (the Sentinel) flags structurally visible failures during execution.

This repository accompanies the paper *AuditFlow: Auditability Framework for Distributed Agentic Workflows* (submitted to *Cluster Computing*).

## Repository layout

| Path | Contents |
|---|---|
| `auditflow_w1/` | W1 Email Triage (single-agent, context poisoning) |
| `auditflow_w2/` | W2 Code Review pipeline (multi-agent, cascading error) |
| `auditflow_w3/` | W3 Web Research agent (tool-augmented, prompt injection) |
| `auditflow_w4/` | W4 Autonomous Coding agent (feedback loop, hallucination loop) |
| `auditflow_w5/` | Section 6 use case: deterministic six-agent GitHub issue-resolution scenario (`run_scenario.py`) and its analysis (`analyse.py`) |
| `sentinel/` | Rule-based runtime monitor |
| `experiments/` | Evaluation scripts (overhead, reconstruction, storage, Sentinel) |
| `experiments/results/sentinel/` | Raw results of the n=30 Sentinel evaluation |

In each workflow folder, `run_agent_original.py` is the un-instrumented baseline, `run_agent_interceptor.py` runs the workflow with AuditFlow, `run_agent_failure.py` runs the failure scenario, `log_events.py` is the event logger and `build_dag.py` reconstructs and verifies the DAG. Recorded traces are in `traces/`.

## Setup

Requires Python 3.11.

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the folder you run from and add your keys. The workflows use GPT-4o-mini through the OpenAI API. W3 additionally needs a Tavily key. The overhead experiment (`experiment1_overhead.py`) uses a local Ollama `llama3.2` model.

## Running

Run workflow scripts from inside their folder, because traces are written to a relative `traces/` directory:

```bash
cd auditflow_w2
python run_agent_interceptor.py     # normal run, writes traces/
python run_agent_failure.py         # failure run
python build_dag.py                 # reconstruct DAG, verify hashes, compare runs
```

Section 6 use case (no LLM, fully deterministic):

```bash
python auditflow_w5/run_scenario.py
python auditflow_w5/analyse.py
```

Experiments (run from the `experiments/` folder):

```bash
cd experiments
python experiment6_sentinel_eval.py      # Sentinel detection, n=30 per condition
python experiment7_sentinel_overhead.py  # Sentinel overhead micro-benchmark
python trace_stats.py ../auditflow_w5/traces/trace_failure.jsonl
```

## Reproducibility notes

LLM outputs are non-deterministic, so re-running the workflows produces different traces and slightly different numbers. The traces and result files in this repository are the ones analysed in the paper. The audit pipeline itself (DAG reconstruction, hashing, verification, earliest-differing-event analysis, Sentinel rules) is deterministic for a given trace.


## License

This project is released under the MIT License.
