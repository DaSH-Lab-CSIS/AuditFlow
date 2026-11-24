"""Six-agent GitHub issue-resolution scenario (Section 6).
Deterministic: no LLM. Writes the same event schema as the other workflows."""

import json, uuid, time, hashlib
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "traces"
OUT_DIR.mkdir(exist_ok=True)


class Logger:
    def __init__(self, path):
        self.path = path
        self.path.unlink(missing_ok=True)
        self.run_id = str(uuid.uuid4())
        self.seq = 0

    def emit(self, etype, node, inp, out, parents=()):
        self.seq += 1
        parents = list(parents)
        ev = {
            "event_id":   str(uuid.uuid4()),
            "run_id":     self.run_id,
            "seq":        self.seq,
            "type":       etype,
            "node":       node,
            "timestamp":  round(time.time(), 4),
            "parent_id":  parents[0] if parents else None,
            "parent_ids": parents,
            "input":      inp,
            "output":     out,
        }
        ev["hash"] = hashlib.sha256(
            json.dumps(ev, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        print(f"  [LOG] seq={self.seq:02d} type={etype:<12} node={node}")
        return ev["event_id"]


def run(failure: bool):
    name = "trace_failure.jsonl" if failure else "trace_normal.jsonl"
    log = Logger(OUT_DIR / name)
    src  = "auth_utils.py" if failure else "auth_handler.py"
    test = "test_auth_utils.py" if failure else "test_auth_handler.py"
    issue = "Issue #4821: IndexError at line 142 in process_data()"

    e1 = log.emit("GRAPH_START", "orchestrator", {"issue": issue}, {})
    e2 = log.emit("LLM_CALL", "issue_analyser",
                  {"issue": issue},
                  {"content": "Likely off-by-one when indexing in process_data; "
                              "locate the module that defines it."}, [e1])
    e3 = log.emit("TOOL_CALL", "code_retriever",
                  {"tool": "find_file", "input": {"query": "process_data"}},
                  {"output": src}, [e2])
    e4 = log.emit("TOOL_CALL", "test_retriever",
                  {"tool": "find_tests", "input": {"source_file": src}},
                  {"output": test}, [e3])
    e5 = log.emit("LLM_CALL", "patch_generator",
                  {"source_file": src, "tests": test},
                  {"content": f"Patch: guard index in {src}, line 142"}, [e3, e4])
    e6 = log.emit("TOOL_CALL", "test_runner",
                  {"tool": "run_tests", "input": {"suite": test}},
                  {"output": "12 passed, 0 failed"}, [e5])
    e7 = log.emit("LLM_CALL", "pr_generator",
                  {"patch_file": src, "test_result": "12 passed"},
                  {"content": f"PR: fix IndexError in {src}"}, [e6])
    log.emit("GRAPH_END", "orchestrator", {},
             {"final_answer": "SUCCESS: PR prepared, all tests pass"}, [e7])


if __name__ == "__main__":
    print("== normal run");  run(failure=False)
    print("== failure run"); run(failure=True)