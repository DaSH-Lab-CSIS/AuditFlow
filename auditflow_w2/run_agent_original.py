"""
Step 1 — Run the code review pipeline with no interceptor.
Just confirm the workflow runs end to end.
"""

import asyncio
import os
from dotenv import load_dotenv
from graph import graph

load_dotenv()

# ── The buggy code we will submit ─────────────────────────────────────────────
# Bug: off-by-one error in the loop — range(len(items)) should be
# range(len(items) - 1) for a sliding window comparison.
# The function is supposed to find the first duplicate in adjacent pairs.

BUGGY_CODE = """
def find_first_adjacent_duplicate(items):
    \"\"\"Return the first item that appears in two consecutive positions.\"\"\"
    for i in range(len(items)):          # BUG: should be range(len(items) - 1)
        if items[i] == items[i + 1]:     # IndexError on last iteration
            return items[i]
    return None
"""

async def main():
    print("=" * 55)
    print("  W2 — Code Review Pipeline")
    print("  Normal run (no failure injection)")
    print("=" * 55)
    print(f"\nSubmitting code:\n{BUGGY_CODE}")

    initial_state = {
        "messages":        [],
        "code_snippet":    BUGGY_CODE,
        "analyzer_output": "",
        "reviewer_output": "",
        "patch_output":    "",
        "final_report":    "",
        "next_agent":      "",
        "inject_failure":  False,      # ← normal run
    }

    result = await graph.ainvoke(initial_state)

    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"\nFinal patch:\n{result.get('patch_output', '')}")

asyncio.run(main())
