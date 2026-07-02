"""
Critic Agent
============
Receives the test result and the current code.
Analyses the failure and writes a diagnosis for the Coder.
Identical in both normal and failure runs.

In the normal run   : receives a real IndexError → diagnoses correctly
In the failure run  : receives a misleading TypeError → misdiagnoses
The Critic behaves correctly given its input in both cases.
This is what makes F16 a cascading failure — the Critic is not broken,
it is working on wrong information.
"""

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


def critic_agent(state: dict) -> dict:
    code        = state.get("current_code", "")
    test_result = state.get("test_result", "")
    iteration   = state.get("iteration", 1)

    print(f"\n[Critic] Analysing failure at iteration {iteration}...")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""You are an expert code reviewer analysing a test failure.

CURRENT CODE:
{code}

TEST OUTPUT:
{test_result}

Based ONLY on the test output above, provide a precise diagnosis:

ROOT CAUSE: [one sentence — what is causing the test failure]
LOCATION:   [specific line or expression in the code]
FIX:        [exact change the developer should make]
"""

    response  = llm.invoke(prompt)
    diagnosis = response.content.strip()

    print(f"[Critic] Diagnosis:\n{diagnosis}\n")

    return {
        "critic_diagnosis": diagnosis,
        "messages": [AIMessage(
            content=f"[Critic iter={iteration}] Diagnosis written.",
            name="critic"
        )],
    }
