"""
Step 1 — W4 Normal run, no interceptor.
Confirm the feedback loop converges correctly.
"""

import asyncio
from dotenv import load_dotenv
from graph import graph

load_dotenv()

TASK = (
    "Write a Python function called find_first_adjacent_duplicate(items) "
    "that returns the first item that appears in two consecutive positions "
    "in a list. Return None if no such item exists."
)

async def main():
    print("=" * 60)
    print("  W4 — Autonomous Coding Agent")
    print("  Normal run — feedback loop should converge")
    print("=" * 60)
    print(f"\nTask: {TASK}\n")

    result = await graph.ainvoke({
        "messages":         [],
        "task":             TASK,
        "current_code":     "",
        "test_result":      "",
        "critic_diagnosis": "",
        "iteration":        1,
        "max_iterations":   4,
        "status":           "RUNNING",
        "inject_failure":   False,
    })

    print("\n" + "=" * 60)
    print(f"  Final status    : {result.get('status')}")
    print(f"  Iterations used : {result.get('iteration')}")
    print(f"  Final code:\n{result.get('current_code')}")
    print("=" * 60)

asyncio.run(main())
