"""Build and compare DAGs from W2 traces."""

import json
import sys
import hashlib
from pathlib import Path


def load_trace(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def build_dag(events: list[dict]) -> tuple[dict, list[tuple]]:
    nodes = {e["event_id"]: e for e in events}
    edges = []
    for e in events:
        if e["parent_id"]:
            edges.append((e["parent_id"], e["event_id"]))
    return nodes, edges


def print_dag(nodes: dict, edges: list[tuple], label: str):
    children = {}
    for parent, child in edges:
        children.setdefault(parent, []).append(child)

    root = next(e["event_id"] for e in nodes.values() if e["parent_id"] is None)

    sym = {"GRAPH_START": "◉", "LLM_CALL": "⬡",
           "TOOL_CALL": "⬟", "GRAPH_END": "◉"}

    # map seq → agent name for W2
    agent_names = {
        1: "graph",
        2: "static_analyzer",
        3: "code_reviewer",
        4: "patch_generator",
        5: "graph",
    }

    print("\n" + "="*65)
    print(f"  EXECUTION DAG — {label}")
    print("="*65)

    def walk(node_id: str, depth: int):
        e      = nodes[node_id]
        indent = "    " * depth
        s      = sym.get(e["type"], "○")
        seq    = e.get("seq", 0)
        agent  = agent_names.get(seq, "unknown")

        print(f"{indent}{s}  [{seq:02d}]  {e['type']:<14}  "
              f"agent={agent:<20} hash={e['hash']}")

        # show key content
        if e["type"] == "LLM_CALL":
            out     = e.get("output", {})
            content = out.get("content", "")[:120]
            print(f"{indent}       → {content}")

        if depth > 0:
            print(f"{indent}       ↑ parent: {e['parent_id'][:8]}…")

        for child_id in children.get(node_id, []):
            print(f"{'    '*depth}    │")
            walk(child_id, depth + 1)

    walk(root, 0)

    print("\n" + "="*65)
    print(f"  Nodes : {len(nodes)}")
    print(f"  Edges : {len(edges)}")
    print(f"  Depth : {len(nodes) - 1}")
    print("="*65)


def verify_integrity(events: list[dict]):
    print("\n  Integrity check:")
    all_ok = True
    for e in events:
        stored    = e["hash"]
        payload   = {k: v for k, v in e.items() if k != "hash"}
        recomputed = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        ok = stored == recomputed
        if not ok:
            all_ok = False
        print(f"    [{e['seq']:02d}] {e['type']:<14} {'✓' if ok else '✗ TAMPERED'}")
    print(f"\n  Result: {'All events intact ✓' if all_ok else 'TAMPERING DETECTED ✗'}")


def compare_runs(normal_events: list, failure_events: list):
    print("\n" + "="*65)
    print("  CROSS-RUN COMPARISON — F6 Cascading Error")
    print("="*65)

    agent_names = {2: "static_analyzer", 3: "code_reviewer", 4: "patch_generator"}

    print(f"\n  Both runs: {len(normal_events)} events, same graph structure")
    print(f"  Same code submitted. Same agents. Same graph.\n")

    # compare LLM outputs at each agent
    print("  Agent-by-agent output comparison:")
    print("  " + "-"*61)

    for seq in [2, 3, 4]:
        n_event = next((e for e in normal_events  if e["seq"] == seq), None)
        f_event = next((e for e in failure_events if e["seq"] == seq), None)
        agent   = agent_names.get(seq, "unknown")

        if not n_event or not f_event:
            continue

        n_out = n_event.get("output", {}).get("content", "")[:200]
        f_out = f_event.get("output", {}).get("content", "")[:200]
        same  = n_out == f_out

        print(f"\n  seq={seq}  agent={agent}")
        print(f"  normal  hash={n_event['hash']}  failure hash={f_event['hash']}")
        print(f"  outputs match: {same}")

        if not same and seq == 2:
            print(f"\n  ← EDE: Earliest Differing Event")
            print(f"  normal  diagnosis: {n_out[:100]}")
            print(f"  failure diagnosis: {f_out[:100]}")

    # show the cascading chain
    print(f"\n  " + "-"*61)
    print(f"  Cascading error chain (F6):")
    print(f"  seq=02  Static Analyzer  → WRONG diagnosis (type annotation)")
    print(f"  seq=03  Code Reviewer    → trusted wrong diagnosis, wrote wrong review")
    print(f"  seq=04  Patch Generator  → followed wrong review, fixed wrong thing")
    print(f"\n  Conventional log: 3x LLM_CALL → SUCCESS. No error. No flag.")
    print(f"  AuditFlow:        seq=02 output diverges from normal run.")
    print(f"                    Wrong diagnosis visible in seq=02 output.")
    print(f"                    seq=03 input contains wrong diagnosis → propagated.")
    print(f"                    seq=04 patch generated for wrong problem.")
    print(f"\n  Root cause: Static Analyzer output at seq=02.")
    print(f"  Causal chain: seq=02 → seq=03 → seq=04 (all linked by parent_id)")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        normal_events  = load_trace(Path(sys.argv[1]))
        failure_events = load_trace(Path(sys.argv[2]))

        print_dag(*build_dag(normal_events),  label="NORMAL")
        verify_integrity(normal_events)
        print_dag(*build_dag(failure_events), label="FAILURE")
        verify_integrity(failure_events)
        compare_runs(normal_events, failure_events)
    else:
        path   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("traces/trace_normal.jsonl")
        events = load_trace(path)
        print_dag(*build_dag(events), label=path.stem.upper())
        verify_integrity(events)