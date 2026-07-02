"""
W2 — Code Review Pipeline (Multi-Agent Orchestrated)
=====================================================
Architecture:

    User submits buggy code
            │
            ▼
      Orchestrator
            │
            ▼
    Static Analyzer        ← reads code, identifies issues
            │
            ▼
      Orchestrator
            │
            ▼
    Code Reviewer          ← reads analyzer output, writes review
            │
            ▼
      Orchestrator
            │
            ▼
    Patch Generator        ← reads review, generates fixed code
            │
            ▼
      Orchestrator → END → Final report

Failure class: F6 — Cascading error propagation
The Static Analyzer misdiagnoses the bug. The Reviewer
and Patch Generator both behave correctly — but on wrong input.
The final patch does not fix the actual bug.
"""

from langgraph.graph import StateGraph, END
from state import WorkflowState
from agents.static_analyzer import static_analyzer_agent
from agents.code_reviewer import code_reviewer_agent
from agents.patch_generator import patch_generator_agent
from agents.orchestrator import orchestrator_agent


def _route_from_orchestrator(state: dict) -> str:
    return state.get("next_agent", "END")


def build_graph():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("orchestrator",    orchestrator_agent)
    workflow.add_node("static_analyzer", static_analyzer_agent)
    workflow.add_node("code_reviewer",   code_reviewer_agent)
    workflow.add_node("patch_generator", patch_generator_agent)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {
            "static_analyzer": "static_analyzer",
            "code_reviewer":   "code_reviewer",
            "patch_generator": "patch_generator",
            "END":             END,
        },
    )

    workflow.add_edge("static_analyzer", "orchestrator")
    workflow.add_edge("code_reviewer",   "orchestrator")
    workflow.add_edge("patch_generator", "orchestrator")

    return workflow.compile()


graph = build_graph()
