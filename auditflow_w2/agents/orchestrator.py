"""
Orchestrator — rule-based routing hub.
No LLM needed here. Routing is deterministic based on state fields.
"""

from langchain_core.messages import AIMessage


def orchestrator_agent(state: dict) -> dict:
    """
    Route to the next agent based on what has been populated in state.

    Routing logic:
      - no analyzer_output  → static_analyzer
      - no reviewer_output  → code_reviewer
      - no patch_output     → patch_generator
      - all populated       → END (compile final report)
    """
    analyzer_output = state.get("analyzer_output", "")
    reviewer_output = state.get("reviewer_output", "")
    patch_output    = state.get("patch_output", "")

    if not analyzer_output:
        print("\n[Orchestrator] No analysis yet → routing to Static Analyzer")
        return {
            "next_agent": "static_analyzer",
            "messages": [AIMessage(
                content="[Orchestrator] Routing to Static Analyzer.",
                name="orchestrator"
            )],
        }

    if not reviewer_output:
        print("\n[Orchestrator] Analysis ready → routing to Code Reviewer")
        return {
            "next_agent": "code_reviewer",
            "messages": [AIMessage(
                content="[Orchestrator] Routing to Code Reviewer.",
                name="orchestrator"
            )],
        }

    if not patch_output:
        print("\n[Orchestrator] Review ready → routing to Patch Generator")
        return {
            "next_agent": "patch_generator",
            "messages": [AIMessage(
                content="[Orchestrator] Routing to Patch Generator.",
                name="orchestrator"
            )],
        }

    # All agents done — compile final report
    print("\n[Orchestrator] All agents complete → compiling final report")

    final_report = f"""
==================================================
       CODE REVIEW PIPELINE — FINAL REPORT
==================================================

ORIGINAL CODE
-------------
{state.get('code_snippet', '')}

STATIC ANALYSIS
---------------
{analyzer_output}

CODE REVIEW
-----------
{reviewer_output}

GENERATED PATCH
---------------
{patch_output}

==================================================
"""
    print(final_report)

    return {
        "next_agent":   "END",
        "final_report": final_report,
        "messages": [AIMessage(
            content="[Orchestrator] Final report compiled.",
            name="orchestrator"
        )],
    }
