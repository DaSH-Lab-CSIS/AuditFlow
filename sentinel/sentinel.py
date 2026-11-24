"""
Sentinel: rule-based, online (real-time) monitor for AuditFlow traces.

Consumes the same normalized event dict produced by each workflow's
log_events.py (_new_event) and applies four deterministic rules:

  1. NON_CONVERGENCE     - repeated identical output within a run
  2. IRREVERSIBLE_ACTION - a TOOL_CALL matching a known destructive action
  3. BUDGET_EXCEEDED     - event count for a run exceeds a configured limit
  4. EVENT_GAP           - a gap in the per-run monotonic `seq` counter
"""

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


ALERTS_FILE = Path("traces/sentinel_alerts.jsonl")

# Tool names treated as irreversible/destructive regardless of payload content.
DEFAULT_HIGH_RISK_TOOLS = {"delete_emails"}


class Sentinel:
    def __init__(
        self,
        non_convergence_threshold: int = 3,
        budget_limit: int = 20,
        high_risk_tools: Optional[set] = None,
        alerts_file: Path = ALERTS_FILE,
        verbose: bool = True,
    ):
        self.non_convergence_threshold = non_convergence_threshold
        self.budget_limit = budget_limit
        self.high_risk_tools = high_risk_tools or set(DEFAULT_HIGH_RISK_TOOLS)
        self.alerts_file = alerts_file
        self.verbose = verbose

        self._hash_counts: Dict[tuple, int] = defaultdict(int)
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._last_seq: Dict[str, int] = {}

        self.alerts_file.parent.mkdir(exist_ok=True, parents=True)

    def on_event(self, event: dict) -> List[dict]:
        alerts = []
        try:
            alerts.extend(self._check_non_convergence(event))
            alerts.extend(self._check_irreversible_action(event))
            alerts.extend(self._check_budget(event))
            alerts.extend(self._check_event_gap(event))
        except Exception as exc:  # pragma: no cover - defensive
            alerts.append(
                self._make_alert(
                    event, "SENTINEL_INTERNAL_ERROR", f"{type(exc).__name__}: {exc}"
                )
            )

        for alert in alerts:
            self._emit(alert)

        return alerts

    def _check_non_convergence(self, event: dict) -> List[dict]:
        # NOTE: event["hash"] is an integrity hash over the *whole* record,
        # including event_id (UUID), seq, and timestamp. all unique per
        # event by construction, so it can never repeat and cannot be used
        # to detect repeated payloads. We compute our own content hash below.
        run_id = event.get("run_id")
        content_hash = self._content_hash(event)
        key = (run_id, event.get("type"), event.get("node"), content_hash)
        self._hash_counts[key] += 1
        count = self._hash_counts[key]

        if count == self.non_convergence_threshold:
            return [
                self._make_alert(
                    event,
                    "NON_CONVERGENCE",
                    f"Identical (type={event.get('type')}, node={event.get('node')}) "
                    f"payload repeated {count} times without new state",
                )
            ]
        return []

    def _check_irreversible_action(self, event: dict) -> List[dict]:
        if event.get("type") != "TOOL_CALL":
            return []

        tool_name = event.get("input", {}).get("tool")
        event_str = json.dumps(event, default=str).lower()

        is_high_risk_tool = tool_name in self.high_risk_tools
        mentions_irreversible = "irreversible" in event_str

        if is_high_risk_tool or mentions_irreversible:
            reason = []
            if is_high_risk_tool:
                reason.append(f"tool '{tool_name}' is in high-risk tool set")
            if mentions_irreversible:
                reason.append("payload contains 'irreversible'")
            return [self._make_alert(event, "IRREVERSIBLE_ACTION", "; ".join(reason))]
        return []

    def _check_budget(self, event: dict) -> List[dict]:
        run_id = event.get("run_id")
        self._event_counts[run_id] += 1
        count = self._event_counts[run_id]

        if count == self.budget_limit:
            return [
                self._make_alert(
                    event,
                    "BUDGET_EXCEEDED",
                    f"Run has emitted {count} events, exceeding the "
                    f"configured budget of {self.budget_limit}",
                )
            ]
        return []

    def _check_event_gap(self, event: dict) -> List[dict]:
        run_id = event.get("run_id")
        seq = event.get("seq")
        if seq is None:
            return []

        last = self._last_seq.get(run_id)
        self._last_seq[run_id] = seq

        if last is not None and seq != last + 1:
            return [
                self._make_alert(
                    event,
                    "EVENT_GAP",
                    f"Sequence discontinuity: expected seq={last + 1}, got seq={seq}",
                )
            ]
        return []

    @staticmethod
    def _content_hash(event: dict) -> str:
        # Hash only the OUTPUT (plus type/node), so repeated results are
        # detected regardless of incidental input differences — e.g. an
        # iteration counter like {"iteration": N} that legitimately changes
        # on every call of a retry/feedback loop but doesn't reflect progress.
        payload = {
            "type": event.get("type"),
            "node": event.get("node"),
            "output": event.get("output"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def _make_alert(self, event: dict, alert_type: str, reason: str) -> dict:
        return {
            "alert_type": alert_type,
            "event_id": event.get("event_id"),
            "run_id": event.get("run_id"),
            "seq": event.get("seq"),
            "node": event.get("node"),
            "reason": reason,
            "timestamp": round(time.time(), 4),
        }

    def _emit(self, alert: dict) -> None:
        with open(self.alerts_file, "a") as f:
            f.write(json.dumps(alert, default=str) + "\n")

        if self.verbose:
            print(
                f"  [SENTINEL] {alert['alert_type']:<20} "
                f"seq={alert['seq']} node={alert['node']} :: {alert['reason']}"
            )

    def reset(self) -> None:
        self._hash_counts.clear()
        self._event_counts.clear()
        self._last_seq.clear()


sentinel = Sentinel()  # module-level singleton