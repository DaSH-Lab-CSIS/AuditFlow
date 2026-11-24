"""Step 3: Event logger — wraps nodes and writes every event to a JSONL file."""

import json
import uuid
import time
import hashlib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.sentinel import sentinel

LOG_FILE = Path("traces/trace.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)

_run_id = str(uuid.uuid4())
_seq = 0
_last_event_id = None

def _new_event(event_type: str, node: str, input_data: dict, output_data: dict) -> dict:
    global _seq, _last_event_id
    _seq += 1
    event_id = str(uuid.uuid4())

    event = {
        "event_id":  event_id,
        "run_id":    _run_id,
        "seq":       _seq,
        "type":      event_type,
        "node":      node,
        "timestamp": round(time.time(), 4),
        "parent_id": _last_event_id,
        "input":     input_data,
        "output":    output_data,
    }

    # hash the payload for integrity
    payload = {k: v for k, v in event.items()}
    event["hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    _last_event_id = event_id

    sentinel.on_event(event)
    # write to file
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")

    print(f"  [LOG] seq={_seq:02d} type={event_type:<12} node={node}")
    return event


def log_llm_call(messages_in: list, message_out) -> None:
    input_data = {
        "messages_count": len(messages_in),
        "messages": [
            {"role": getattr(m, "type", "unknown"), "content": str(m.content)[:300]}
            for m in messages_in
        ]
    }
    tool_calls = getattr(message_out, "tool_calls", [])
    output_data = {
        "content": str(message_out.content)[:500],
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls],
        "has_tool_calls": len(tool_calls) > 0,
    }
    _new_event("LLM_CALL", "call_model", input_data, output_data)


def log_tool_call(tool_name: str, tool_input: dict, tool_output: str) -> None:
    _new_event("TOOL_CALL", "tools", {"tool": tool_name, "input": tool_input},
               {"output": str(tool_output)[:500]})


def log_graph_start(query: str) -> None:
    _new_event("GRAPH_START", "graph", {"query": query}, {})


def log_graph_end(final_answer: str) -> None:
    _new_event("GRAPH_END", "graph", {}, {"final_answer": final_answer[:500]})