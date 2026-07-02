"""Step 4: Read a real trace and build + visualise the DAG."""

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

    type_symbol = {
        "GRAPH_START": "◉",
        "LLM_CALL":    "⬡",
        "TOOL_CALL":   "⬟",
        "GRAPH_END":   "◉",
    }

    print("\n" + "="*65)
    print(f"  EXECUTION DAG — {label}")
    print("="*65)

    def walk(node_id: str, depth: int):
        e = nodes[node_id]
        indent = "    " * depth
        sym = type_symbol.get(e["type"], "○")

        # flag dangerous tool calls in red
        flag = ""
        if e["type"] == "TOOL_CALL":
            inp = e["input"]
            if inp.get("tool") == "delete_emails":
                flag = "  ◄ IRREVERSIBLE ACTION"

        print(f"{indent}{sym}  [{e['seq']:02d}]  {e['type']:<14}  hash={e['hash']}{flag}")

        if e["type"] == "LLM_CALL":
            out = e["output"]
            if out.get("has_tool_calls"):
                for tc in out.get("tool_calls", []):
                    print(f"{indent}       → tool call : {tc['name']}({tc['args']})")
            else:
                print(f"{indent}       → answer    : {out.get('content','')[:100]}")
            # show context size — key for poisoning story
            print(f"{indent}       → ctx messages: {e['input']['messages_count']}")

        elif e["type"] == "TOOL_CALL":
            inp = e["input"]
            print(f"{indent}       → tool   : {inp['tool']}")
            print(f"{indent}       → input  : {inp['input']}")
            out_preview = str(e["output"].get("output",""))[:120]
            print(f"{indent}       → output : {out_preview}")

        if depth > 0:
            print(f"{indent}       ↑ parent : {e['parent_id'][:8]}…")

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
        stored = e["hash"]
        payload = {k: v for k, v in e.items() if k != "hash"}
        recomputed = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        ok = stored == recomputed
        if not ok:
            all_ok = False
        flag = ""
        if e["type"] == "TOOL_CALL" and e["input"].get("tool") == "delete_emails":
            flag = "  ◄ delete action"
        print(f"    [{e['seq']:02d}] {e['type']:<14} {'✓' if ok else '✗ TAMPERED'}{flag}")
    print(f"\n  Result: {'All events intact ✓' if all_ok else 'TAMPERING DETECTED ✗'}")

def compare_runs(normal_events: list, failure_events: list):
    print("\n" + "="*65)
    print("  CROSS-RUN COMPARISON")
    print("="*65)

    n_tools = [e for e in normal_events  if e["type"] == "TOOL_CALL"]
    f_tools = [e for e in failure_events if e["type"] == "TOOL_CALL"]

    print(f"\n  Normal  run : {len(normal_events)} events, "
          f"{len(n_tools)} tool calls → "
          f"{[e['input']['tool'] for e in n_tools]}")
    print(f"  Failure run : {len(failure_events)} events, "
          f"{len(f_tools)} tool calls → "
          f"{[e['input']['tool'] for e in f_tools]}")

    # find earliest differing event (EDE)
    # find earliest differing event (EDE)
    print("\n  Earliest Differing Event (EDE):")
    found_ede = False
    for i, (n, f) in enumerate(zip(normal_events, failure_events)):
        n_tool = n["input"].get("tool","") if n["type"] == "TOOL_CALL" else ""
        f_tool = f["input"].get("tool","") if f["type"] == "TOOL_CALL" else ""
        n_ctx  = n["input"].get("messages_count", 0)
        f_ctx  = f["input"].get("messages_count", 0)

        if n["type"] != f["type"] or n_tool != f_tool or n_ctx != f_ctx:
            print(f"    seq={i+1}")
            print(f"    normal  → {n['type']}  ctx_messages={n_ctx}  tool={n_tool or 'none'}")
            print(f"    failure → {f['type']}  ctx_messages={f_ctx}  tool={f_tool or 'none'}")
            print(f"    Divergence: failure run has {f_ctx - n_ctx} extra messages in context")
            print(f"    (stale instructions from prior turns still present)")
            found_ede = True
            break
    if not found_ede:
        print("    Runs follow identical topology — divergence is in outputs only")
        
    # flag destructive actions
    deletes = [e for e in failure_events
               if e["type"] == "TOOL_CALL" and e["input"].get("tool") == "delete_emails"]
    if deletes:
        print(f"\n  Destructive actions in failure run: {len(deletes)}")
        for d in deletes:
            print(f"    seq={d['seq']}  deleted={d['input']['input']}")
        print(f"\n  Root cause visible in AuditFlow trace:")
        print(f"    LLM_CALL seq=02 context shows stale instruction in message[0]:")

        # pull the stale instruction from the failure trace
        llm2 = next((e for e in failure_events
                     if e["type"] == "LLM_CALL" and e["seq"] == 2), None)
        if llm2:
            for msg in llm2["input"]["messages"]:
                if "delete" in msg["content"].lower():
                    print(f"    [{msg['role'].upper()}]: {msg['content']}")

        print(f"\n  Conventional log shows only: delete_emails() → SUCCESS")
        print(f"  AuditFlow shows: stale instruction → LLM decision → delete tool → IRREVERSIBLE")

if __name__ == "__main__":
    # run on one file or compare both
    if len(sys.argv) == 3:
        normal_path  = Path(sys.argv[1])
        failure_path = Path(sys.argv[2])
        normal_events  = load_trace(normal_path)
        failure_events = load_trace(failure_path)

        print_dag(*build_dag(normal_events),  label="NORMAL")
        verify_integrity(normal_events)
        print_dag(*build_dag(failure_events), label="FAILURE")
        verify_integrity(failure_events)
        compare_runs(normal_events, failure_events)

    else:
        path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("traces/trace_normal.jsonl")
        events = load_trace(path)
        print(f"Loaded {len(events)} events from {path}")
        print_dag(*build_dag(events), label=path.stem.upper())
        verify_integrity(events)