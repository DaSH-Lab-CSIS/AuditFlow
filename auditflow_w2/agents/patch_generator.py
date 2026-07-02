"""
Patch Generator Agent
=====================
Receives the code review and generates a corrected version of the code.
This agent is IDENTICAL in both normal and failure runs.
It follows the reviewer's instructions precisely.
If the review described the wrong fix, the patch will implement the wrong fix.
"""

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


def patch_generator_agent(state: dict) -> dict:
    code            = state.get("code_snippet", "")
    reviewer_output = state.get("reviewer_output", "")

    print("\n[Patch Generator] Generating patch based on code review...")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""You are an expert Python developer. Your job is to fix the code based on the code review.

ORIGINAL CODE:
{code}

CODE REVIEW (follow these instructions exactly):
{reviewer_output}

Generate the fixed version of the code. Follow the reviewer's recommended fix precisely.
Return ONLY the corrected Python code — no explanation, no markdown fences, just the code.
"""

    response = llm.invoke(prompt)
    output   = response.content.strip()

    # strip markdown fences if model added them
    if output.startswith("```"):
        lines  = output.split("\n")
        output = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    print(f"[Patch Generator] Patch:\n{output}\n")

    return {
        "patch_output": output,
        "messages": [AIMessage(
            content="[Patch Generator] Patch generated.",
            name="patch_generator"
        )],
    }
