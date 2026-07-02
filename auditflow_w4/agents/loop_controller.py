"""
Loop Controller
===============
Decides after each test run whether to:
  - END (tests passed)
  - iterate (tests failed, under limit)
  - END with failure (max iterations reached)

This is a rule-based node — no LLM call needed.
"""

from langchain_core.messages import AIMessage


def loop_controller(state: dict) -> dict:
    test_result    = state.get("test_result", "")
    iteration      = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 4)

    passed = "PASSED" in test_result and "FAILED" not in test_result

    if passed:
        print(f"\n[Loop Controller] Tests PASSED at iteration {iteration} → END")
        return {
            "status": "PASSED",
            "iteration": iteration,
            "messages": [AIMessage(
                content=f"[Loop Controller] PASSED at iteration {iteration}.",
                name="loop_controller"
            )],
        }

    if iteration >= max_iterations:
        print(f"\n[Loop Controller] Max iterations ({max_iterations}) reached → END")
        return {
            "status": "FAILED_MAX_ITER",
            "iteration": iteration,
            "messages": [AIMessage(
                content=f"[Loop Controller] Max iterations reached. Loop terminated.",
                name="loop_controller"
            )],
        }

    next_iter = iteration + 1
    print(f"\n[Loop Controller] Tests failed → iterating (iteration {next_iter})")
    return {
        "status":    "RUNNING",
        "iteration": next_iter,
        "messages": [AIMessage(
            content=f"[Loop Controller] Iterating → iteration {next_iter}.",
            name="loop_controller"
        )],
    }
