"""
W4 — Autonomous Coding Agent (Feedback-driven)
===============================================
Architecture:

    START
      │
      ▼
    Coder          ← generates/fixes code
      │
      ▼
    Test Runner    ← runs tests, reports result
      │
      ▼
    Loop Controller ──── PASSED ──────────────► END
          │
          ├── RUNNING → Critic → Coder (next iteration)
          │
          └── FAILED_MAX_ITER ─────────────────► END

Failure class: F16 — Hallucination loop
The Test Runner returns a misleading error.
The Critic misdiagnoses. The Coder applies the wrong fix.
The loop repeats identically until max_iterations is hit.
"""

from langgraph.graph import StateGraph, END
from state import CodingState
from agents.coder import coder_agent
from agents.test_runner import test_runner
from agents.critic import critic_agent
from agents.loop_controller import loop_controller


def route_after_controller(state: dict) -> str:
    status = state.get("status", "RUNNING")
    if status == "PASSED":
        return "end"
    if status == "FAILED_MAX_ITER":
        return "end"
    return "critic"   # RUNNING → critic → coder → test_runner → loop_controller


def build_graph():
    workflow = StateGraph(CodingState)

    workflow.add_node("coder",           coder_agent)
    workflow.add_node("test_runner",     test_runner)
    workflow.add_node("loop_controller", loop_controller)
    workflow.add_node("critic",          critic_agent)

    workflow.set_entry_point("coder")

    workflow.add_edge("coder",           "test_runner")
    workflow.add_edge("test_runner",     "loop_controller")

    workflow.add_conditional_edges(
        "loop_controller",
        route_after_controller,
        {
            "end":    END,
            "critic": "critic",
        },
    )

    workflow.add_edge("critic", "coder")

    return workflow.compile()


graph = build_graph()
