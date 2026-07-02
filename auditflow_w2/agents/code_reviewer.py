"""
Code Reviewer Agent
===================
Receives the static analyzer output and writes a structured code review.
This agent is IDENTICAL in both normal and failure runs.
It trusts the analyzer output completely — which is the point.
If the analyzer was wrong, the review will be wrong too.
"""

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


def code_reviewer_agent(state: dict) -> dict:
    code            = state.get("code_snippet", "")
    analyzer_output = state.get("analyzer_output", "")

    print("\n[Code Reviewer] Writing review based on analyzer output...")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""You are a senior software engineer conducting a code review.

The static analyser has already identified the following issue:

{analyzer_output}

Here is the code under review:

{code}

Based on the static analyser's findings, write a structured code review:

REVIEW SUMMARY:
[One sentence summary of the main issue]

PROBLEM EXPLANATION:
[2-3 sentences explaining why this is a problem and what impact it has]

RECOMMENDED FIX:
[Specific instructions for how to fix the issue identified by the analyser]

ADDITIONAL NOTES:
[Any other observations — keep brief]
"""

    response = llm.invoke(prompt)
    output   = response.content

    print(f"[Code Reviewer] Review:\n{output}\n")

    return {
        "reviewer_output": output,
        "messages": [AIMessage(
            content="[Code Reviewer] Review complete.",
            name="code_reviewer"
        )],
    }
