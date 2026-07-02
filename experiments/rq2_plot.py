"""Measure DAG reconstruction cost for RQ2."""
import json
import time
import hashlib
from pathlib import Path

TRACES = {
    "W1 Email Triage":  "../auditflow_w1/traces/trace_normal.jsonl",
    "W2 Code Review":   "../auditflow_w2/traces/trace_normal.jsonl",
    "W3 Web Research":  "../auditflow_w3/traces/trace_normal.jsonl",
    "W4 Coding Agent":  "../auditflow_w4/traces/trace_normal.jsonl",
    "W4 Failure":       "../auditflow_w4/traces/trace_failure.jsonl",
}

def load_trace(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events

def build_dag(events):
    nodes = {e["event_id"]: e for e in events}
    children = {}
    for e in events:
        if e.get("parent_id"):
            children.setdefault(e["parent_id"], []).append(e["event_id"])
    return nodes, children

def dag_depth(nodes, children):
    depths = {}
    def dfs(n):
        if n in depths: return depths[n]
        if n not in children:
            depths[n] = 0; return 0
        d = 1 + max(dfs(c) for c in children[n])
        depths[n] = d; return d
    for n in nodes: dfs(n)
    return max(depths.values()) if depths else 0

def causal_chain_query(nodes, children):
    """Root to leaf traversal."""
    root = next(e["event_id"] for e in nodes.values()
                if not e.get("parent_id"))
    path = []
    def walk(n):
        path.append(n)
        for c in children.get(n, []):
            walk(c)
    walk(root)
    return path

def agent_filter_query(nodes):
    """Filter LLM_CALL events."""
    return [n for n in nodes.values()
            if n.get("type") == "LLM_CALL"]

def tool_query(nodes):
    """Filter TOOL_CALL events."""
    return [n for n in nodes.values()
            if n.get("type") == "TOOL_CALL"]

N = 100  # runs per measurement

print(f"\n{'Workflow':<22} {'Events':>7} {'Nodes':>6} "
      f"{'Depth':>6} {'Build(ms)':>10} "
      f"{'Causal(ms)':>11} {'Filter(ms)':>11} {'Tool(ms)':>9}")
print("-" * 90)

for name, path in TRACES.items():
    if not Path(path).exists():
        print(f"  SKIP {name} — not found")
        continue

    events = load_trace(path)
    n_events = len(events)

    # DAG construction time
    build_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        nodes, children = build_dag(events)
        t1 = time.perf_counter()
        build_times.append((t1-t0)*1000)

    depth = dag_depth(nodes, children)

    # causal chain query time
    causal_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = causal_chain_query(nodes, children)
        t1 = time.perf_counter()
        causal_times.append((t1-t0)*1000)

    # agent filter query time
    filter_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = agent_filter_query(nodes)
        t1 = time.perf_counter()
        filter_times.append((t1-t0)*1000)

    # tool query time
    tool_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = tool_query(nodes)
        t1 = time.perf_counter()
        tool_times.append((t1-t0)*1000)

    def mean(vals): return round(sum(vals)/len(vals), 4)

    print(f"{name:<22} {n_events:>7} {len(nodes):>6} "
          f"{depth:>6} {mean(build_times):>10} "
          f"{mean(causal_times):>11} "
          f"{mean(filter_times):>11} "
          f"{mean(tool_times):>9}")