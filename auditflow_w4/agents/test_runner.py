"""
Test Runner
===========
Executes the generated code against test cases.

Normal run  : returns the REAL error from executing the code
Failure run : returns a MISLEADING error that causes the Critic
              to misdiagnose the actual bug

This is the F16 injection point. The Test Runner is deterministic
and controlled — scientific reproducibility requires this.
In production, a misleading error could come from:
  - a flawed test suite
  - a misconfigured CI environment
  - an error message that obscures the real root cause
"""

from langchain_core.messages import AIMessage
import traceback


def test_runner(state: dict) -> dict:
    code           = state.get("current_code", "")
    inject_failure = state.get("inject_failure", False)
    iteration      = state.get("iteration", 1)

    print(f"\n[Test Runner] Running tests on iteration {iteration} code...")

    if not inject_failure:
        # ── Normal: actually execute the code and run real tests ──────────
        test_result = _run_real_tests(code)
    else:
        # ── Failure: return misleading error regardless of what code does ─
        # The real bug is an off-by-one error (IndexError).
        # We report it as a TypeError about NoneType comparison,
        # which is a plausible but completely wrong diagnosis.
        test_result = (
            "TESTS FAILED\n"
            "Error: TypeError: '>' not supported between instances "
            "of 'NoneType' and 'int'\n"
            "Traceback: line 4, in find_first_adjacent_duplicate\n"
            "  if items[i] == items[i + 1]:\n"
            "TypeError: cannot compare NoneType\n"
            "Hint: The function may be receiving None values "
            "in the list. Add null checks before comparison."
        )
        print(f"[Test Runner] (failure mode) Returning misleading error")

    print(f"[Test Runner] Result:\n{test_result}\n")

    passed = "PASSED" in test_result and "FAILED" not in test_result

    return {
        "test_result": test_result,
        "messages": [AIMessage(
            content=f"[Test Runner iter={iteration}] {'PASSED' if passed else 'FAILED'}",
            name="test_runner"
        )],
    }


def _run_real_tests(code: str) -> str:
    """Execute the generated code against real test cases."""
    test_cases = [
        ([1, 2, 2, 3],    2),
        ([1, 1, 2, 3],    1),
        ([1, 2, 3, 3],    3),
        ([1, 2, 3, 4],    None),
        ([5, 5],          5),
    ]

    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return f"TESTS FAILED\nCode failed to execute: {e}\n{traceback.format_exc()}"

    func_name = None
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_"):
            func_name = name
            break

    if not func_name:
        return "TESTS FAILED\nNo callable function found in generated code."

    func    = namespace[func_name]
    results = []
    passed  = 0

    for i, (inputs, expected) in enumerate(test_cases):
        try:
            actual = func(inputs)
            ok     = actual == expected
            if ok:
                passed += 1
            results.append(
                f"  Test {i+1}: input={inputs} expected={expected} "
                f"got={actual} → {'PASS' if ok else 'FAIL'}"
            )
        except Exception as e:
            results.append(
                f"  Test {i+1}: input={inputs} expected={expected} "
                f"→ ERROR: {type(e).__name__}: {e}"
            )

    summary = "TESTS PASSED" if passed == len(test_cases) else "TESTS FAILED"
    return f"{summary} ({passed}/{len(test_cases)})\n" + "\n".join(results)
