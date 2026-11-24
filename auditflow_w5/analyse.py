"""Section 6 analysis: verify, earliest differing event, propagation, tamper test."""

import json, time, hashlib, copy
from pathlib import Path

D = Path(__file__).resolve().parent / "traces"


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rehash(e):
    payload = {k: v for k, v in e.items() if k != "hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def verify(events):
    """Return list of seq numbers whose hash does not match."""
    return [e["seq"] for e in events if e["hash"] != rehash(e)]


def body(e):
    return json.dumps({"type": e["type"], "node": e["node"],
                       "input": e["input"], "output": e["output"]},
                      sort_keys=True)


def earliest_diff(a, b):
    for ea, eb in zip(a, b):
        if body(ea) != body(eb):
            return ea["seq"]
    return None


def descendants(events, root_id):
    kids = {}
    for e in events:
        for p in e.get("parent_ids") or []:
            kids.setdefault(p, []).append(e["event_id"])
    seen, stack = set(), [root_id]
    while stack:
        x = stack.pop()
        for c in kids.get(x, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def timed(fn, reps=2000):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000  # ms


normal  = load(D / "trace_normal.jsonl")
failure = load(D / "trace_failure.jsonl")
by_seq  = {e["seq"]: e for e in failure}

print("1. Integrity")
print("   normal  tampered seqs:", verify(normal)  or "none")
print("   failure tampered seqs:", verify(failure) or "none")
print(f"   verify time per run : {timed(lambda: verify(failure)):.3f} ms")

print("\n2. Earliest differing event (EDE)")
s = earliest_diff(normal, failure)
e = by_seq[s]
print(f"   seq={s} node={e['node']}")
print("   normal :", next(x for x in normal if x['seq'] == s)["output"])
print("   failure:", e["output"])
print(f"   EDE time           : {timed(lambda: earliest_diff(normal, failure)):.3f} ms")

print("\n3. Propagation")
desc = descendants(failure, e["event_id"])
differ = [x["seq"] for x in failure
          if body(x) != body(next(y for y in normal if y["seq"] == x["seq"]))]
print(f"   causal descendants of EDE : {len(desc)} of {len(failure)-1} later events")
print(f"   events with differing payload (seq): {differ}")
print(f"   final answers identical   : "
      f"{normal[-1]['output'] == failure[-1]['output']}")

print("\n4. Tamper test (alter one output in failure trace)")
bad = copy.deepcopy(failure)
bad[2]["output"]["output"] = "auth_handler.py"
print("   detected at seq:", verify(bad))