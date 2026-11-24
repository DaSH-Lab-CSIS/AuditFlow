import json
import sys
from collections import Counter, defaultdict


def parents_of(e):
    p = e.get("parent_ids") or e.get("parent_id", e.get("parents"))
    if p is None:
        return []
    return p if isinstance(p, list) else [p]


def stats(rows, name):
    events = [e for e, _ in rows]
    ids = {e["event_id"]: e for e in events}
    memo = {}

    def depth(eid):
        if eid in memo:
            return memo[eid]
        memo[eid] = 0  # guards against cycles
        ps = [p for p in parents_of(ids[eid]) if p in ids]
        memo[eid] = 1 + max((depth(p) for p in ps), default=0)
        return memo[eid]

    total = sum(n for _, n in rows)
    snap = [len(json.dumps({"i": e.get("input"), "o": e.get("output")},
                           default=str).encode()) for e in events]
    print(f"== {name}")
    print(f" events            : {len(events)}")
    print(f" by type           : {dict(Counter(e.get('type') for e in events))}")
    print(f" agents/nodes      : {len({e.get('agent') or e.get('node') for e in events})}")
    print(f" causal edges      : {sum(len([p for p in parents_of(e) if p in ids]) for e in events)}")
    print(f" DAG depth         : {max(depth(i) for i in ids)}")
    print(f" JSONL size        : {total} B ({total / 1024:.2f} KB, {total / len(rows):.0f} B/event)")
    print(f" state snapshots   : {sum(snap)} B total, largest event {max(snap)} B")


def main(path):
    runs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs[json.loads(line).get("run_id")].append(
                    (json.loads(line), len(line.encode("utf-8"))))
    for rid, rows in runs.items():
        stats(rows, f"run {rid}")


if __name__ == "__main__":
    main(sys.argv[1])