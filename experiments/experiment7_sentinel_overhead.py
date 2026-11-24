import sys
import tempfile
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.sentinel import Sentinel

N = 20000                      # events per measurement
REPEATS = 5
PAYLOAD = "x" * 1000           # ~1 KB, comparable to your average event size
ALERTS = Path(tempfile.gettempdir()) / "sentinel_bench_alerts.jsonl"


def make_event(i, tool):
    return {
        "event_id": f"e{i}",
        "run_id": f"r{i // 8}",            # 8 events per run
        "seq": i % 8,                      # consecutive within a run
        "type": "TOOL_CALL" if i % 2 == 0 else "TOOL_RESULT",
        "node": "agent",
        "input": {"tool": tool, "q": i},
        "output": {"text": PAYLOAD + str(i)},   # unique, so no non-convergence
    }


def bench(tool):
    if ALERTS.exists():
        ALERTS.unlink()
    s = Sentinel(alerts_file=ALERTS, verbose=False)
    events = [make_event(i, tool) for i in range(N)]
    t0 = time.perf_counter()
    for e in events:
        s.on_event(e)
    t1 = time.perf_counter()
    return (t1 - t0) / N * 1e6          # microseconds per event


no_alert = [bench("search_emails") for _ in range(REPEATS)]
with_alert = [bench("delete_emails") for _ in range(REPEATS)]

print(f"No-alert path   : median {statistics.median(no_alert):.1f} us/event "
      f"(min {min(no_alert):.1f}, max {max(no_alert):.1f})")
print(f"Alert-firing path: median {statistics.median(with_alert):.1f} us/event "
      f"(every TOOL_CALL fires, so this is a worst case)")