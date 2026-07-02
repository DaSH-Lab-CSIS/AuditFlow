"""
Coder Agent
===========
Iteration 1 : starts with a known buggy implementation
Iteration 2+: applies the critic's suggested fix
"""

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

# Known buggy code — off-by-one error in range()
# range(len(items)) should be range(len(items) - 1)
# This causes IndexError on the last iteration
BUGGY_CODE = """def find_first_adjacent_duplicate(items):
    \"\"\"Return the first item that appears in two consecutive positions.\"\"\"
    for i in range(len(items)):
        if items[i] == items[i + 1]:
            return items[i]
    return None"""


def coder_agent(state: dict) -> dict:
    iteration        = state.get("iteration", 1)
    task             = state.get("task", "")
    critic_diagnosis = state.get("critic_diagnosis", "")
    current_code     = state.get("current_code", "")

    print(f"\n[Coder] Iteration {iteration} — generating code...")

    if iteration == 1:
        # always start with the known buggy code
        print(f"[Coder] Starting with known buggy implementation")
        print(f"[Coder] Code:\n{BUGGY_CODE}\n")
        return {
            "current_code": BUGGY_CODE,
            "messages": [AIMessage(
                content=f"[Coder iter=1] Buggy code submitted.",
                name="coder"
            )],
        }

    # iteration 2+ — fix based on critic diagnosis
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""You are an expert Python developer fixing a bug.

ORIGINAL TASK: {task}

CURRENT CODE:
{current_code}

CRITIC DIAGNOSIS:
{critic_diagnosis}

Fix the code based ONLY on the critic's diagnosis.
Return ONLY the corrected Python code. No explanation, no markdown fences.
"""

    response = llm.invoke(prompt)
    code     = response.content.strip()

    if code.startswith("```"):
        lines = code.split("\n")
        code  = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    print(f"[Coder] Fixed code:\n{code}\n")

    return {
        "current_code": code,
        "messages": [AIMessage(
            content=f"[Coder iter={iteration}] Fix applied.",
            name="coder"
        )],
    }