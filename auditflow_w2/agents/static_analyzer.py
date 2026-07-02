"""
Static Analyzer Agent
=====================
Analyses the submitted code and identifies bugs.

Normal run : correctly identifies an off-by-one error
Failure run: misdiagnoses it as a type error (F6 injection point)

The inject_failure flag in state controls which prompt is used.
Everything else in the pipeline is identical between the two runs.
"""

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
import os


def static_analyzer_agent(state: dict) -> dict:
    code    = state.get("code_snippet", "")
    failure = state.get("inject_failure", False)

    print(f"\n[Static Analyzer] Analysing code... (failure_mode={failure})")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if not failure:
        # ── Normal prompt: find the real bug ──────────────────────────────
        prompt = f"""You are a static code analyser. Carefully analyse this Python function and identify any bugs.

CODE:
{code}

Provide your analysis in this exact format:

BUG TYPE: [the category of bug]
LOCATION: [line number or function name where bug occurs]
DESCRIPTION: [one sentence describing the bug precisely]
EVIDENCE: [quote the specific line(s) that contain the bug]
SEVERITY: [HIGH / MEDIUM / LOW]
"""
    else:
        # ── Failure prompt: steered toward wrong diagnosis ─────────────────
        # This simulates a misconfigured or degraded analyzer that
        # focuses on type checking and misses the actual off-by-one error.
        # In a real system this could happen due to a wrong model version,
        # a corrupted prompt template, or a misconfigured analysis rule.
        prompt = f"""You are a static code analyser specialising in type safety.
Analyse this Python function focusing on TYPE ERRORS and TYPE MISMATCHES.

CODE:
{code}

Provide your analysis in this exact format:

BUG TYPE: [the category of bug — focus on type issues]
LOCATION: [line number or function name where bug occurs]
DESCRIPTION: [one sentence describing the bug precisely]
EVIDENCE: [quote the specific line(s) that contain the bug]
SEVERITY: [HIGH / MEDIUM / LOW]
"""

    response = llm.invoke(prompt)
    output   = response.content

    print(f"[Static Analyzer] Output:\n{output}\n")

    return {
        "analyzer_output": output,
        "messages": [AIMessage(
            content=f"[Static Analyzer] Analysis complete.",
            name="static_analyzer"
        )],
    }
