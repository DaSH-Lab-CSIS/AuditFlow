"""
Step 2 — Normal run with interceptor.
Same workflow, same code, same result.
Now every agent call is logged to traces/trace_normal.jsonl
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from log_events import (log_graph_start, log_graph_end,
                        log_llm_call, log_tool_call)
from pathlib import Path
import log_events

load_dotenv()

# reset logger
log_events.LOG_FILE = Path("traces/trace_normal.jsonl")
log_events.LOG_FILE.parent.mkdir(exist_ok=True)
log_events.LOG_FILE.write_text("")
log_events._run_id = __import__("uuid").uuid4().__str__()
log_events._seq = 0
log_events._last_event_id = None

from state import WorkflowState

BUGGY_CODE = """
def find_first_adjacent_duplicate(items):
    \"\"\"Return the first item that appears in two consecutive positions.\"\"\"
    for i in range(len(items)):
        if items[i] == items[i + 1]:
            return items[i]
    return None
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Instrumented agents ───────────────────────────────────────────────────────

def orchestrator_agent(state: dict) -> dict:
    analyzer_output = state.get("analyzer_output", "")
    reviewer_output = state.get("reviewer_output", "")
    patch_output    = state.get("patch_output", "")

    if not analyzer_output:
        print("\n[Orchestrator] → static_analyzer")
        return {
            "next_agent": "static_analyzer",
            "messages": [AIMessage(content="[Orchestrator] → static_analyzer",
                                   name="orchestrator")],
        }
    if not reviewer_output:
        print("\n[Orchestrator] → code_reviewer")
        return {
            "next_agent": "code_reviewer",
            "messages": [AIMessage(content="[Orchestrator] → code_reviewer",
                                   name="orchestrator")],
        }
    if not patch_output:
        print("\n[Orchestrator] → patch_generator")
        return {
            "next_agent": "patch_generator",
            "messages": [AIMessage(content="[Orchestrator] → patch_generator",
                                   name="orchestrator")],
        }

    print("\n[Orchestrator] → END")
    final_report = (
        f"ANALYSIS:\n{analyzer_output}\n\n"
        f"REVIEW:\n{reviewer_output}\n\n"
        f"PATCH:\n{patch_output}"
    )
    return {
        "next_agent": "END",
        "final_report": final_report,
        "messages": [AIMessage(content="[Orchestrator] Final report compiled.",
                               name="orchestrator")],
    }


def static_analyzer_agent(state: dict) -> dict:
    code = state.get("code_snippet", "")
    print("\n[Static Analyzer] Analysing...")

    prompt = f"""You are a static code analyser. Carefully analyse this Python function and identify any bugs.

CODE:
{code}

Provide your analysis in this exact format:

BUG TYPE: [the category of bug]
LOCATION: [line number or function name]
DESCRIPTION: [one sentence describing the bug precisely]
EVIDENCE: [quote the specific line(s) that contain the bug]
SEVERITY: [HIGH / MEDIUM / LOW]
"""
    messages_in = [{"role": "user", "content": prompt}]
    response    = llm.invoke(messages_in)

    # ── LOG ──
    log_llm_call(
        [type("M", (), {"type": "human", "content": prompt})()],
        response
    )

    print(f"[Static Analyzer] Output:\n{response.content}\n")
    return {
        "analyzer_output": response.content,
        "messages": [AIMessage(content="[Static Analyzer] Done.",
                               name="static_analyzer")],
    }


def code_reviewer_agent(state: dict) -> dict:
    code            = state.get("code_snippet", "")
    analyzer_output = state.get("analyzer_output", "")
    print("\n[Code Reviewer] Writing review...")

    prompt = f"""You are a senior software engineer conducting a code review.

The static analyser found:
{analyzer_output}

Code under review:
{code}

Write a structured review:

REVIEW SUMMARY:
[one sentence]

PROBLEM EXPLANATION:
[2-3 sentences]

RECOMMENDED FIX:
[specific fix instructions]

ADDITIONAL NOTES:
[brief]
"""
    messages_in = [{"role": "user", "content": prompt}]
    response    = llm.invoke(messages_in)

    # ── LOG ──
    log_llm_call(
        [type("M", (), {"type": "human", "content": prompt})()],
        response
    )

    print(f"[Code Reviewer] Review:\n{response.content}\n")
    return {
        "reviewer_output": response.content,
        "messages": [AIMessage(content="[Code Reviewer] Done.",
                               name="code_reviewer")],
    }


def patch_generator_agent(state: dict) -> dict:
    code            = state.get("code_snippet", "")
    reviewer_output = state.get("reviewer_output", "")
    print("\n[Patch Generator] Generating patch...")

    prompt = f"""You are an expert Python developer fixing code based on a code review.

ORIGINAL CODE:
{code}

CODE REVIEW:
{reviewer_output}

Return ONLY the corrected Python code. No explanation, no markdown fences.
"""
    messages_in = [{"role": "user", "content": prompt}]
    response    = llm.invoke(messages_in)

    output = response.content.strip()
    if output.startswith("```"):
        lines  = output.split("\n")
        output = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    # ── LOG ──
    log_llm_call(
        [type("M", (), {"type": "human", "content": prompt})()],
        response
    )

    print(f"[Patch Generator] Patch:\n{output}\n")
    return {
        "patch_output": output,
        "messages": [AIMessage(content="[Patch Generator] Done.",
                               name="patch_generator")],
    }


def _route(state: dict) -> str:
    return state.get("next_agent", "END")


# ── Build graph ───────────────────────────────────────────────────────────────

builder = StateGraph(WorkflowState)
builder.add_node("orchestrator",    orchestrator_agent)
builder.add_node("static_analyzer", static_analyzer_agent)
builder.add_node("code_reviewer",   code_reviewer_agent)
builder.add_node("patch_generator", patch_generator_agent)
builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", _route, {
    "static_analyzer": "static_analyzer",
    "code_reviewer":   "code_reviewer",
    "patch_generator": "patch_generator",
    "END":             END,
})
builder.add_edge("static_analyzer", "orchestrator")
builder.add_edge("code_reviewer",   "orchestrator")
builder.add_edge("patch_generator", "orchestrator")
graph = builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("  W2 Normal Run — with interceptor")
    print("=" * 55)

    log_graph_start("Code review: find_first_adjacent_duplicate")

    result = await graph.ainvoke({
        "messages":        [],
        "code_snippet":    BUGGY_CODE,
        "analyzer_output": "",
        "reviewer_output": "",
        "patch_output":    "",
        "final_report":    "",
        "next_agent":      "",
        "inject_failure":  False,
    })

    log_graph_end(result.get("patch_output", ""))

    print("\n=== Trace written to traces/trace_normal.jsonl ===")

asyncio.run(main())