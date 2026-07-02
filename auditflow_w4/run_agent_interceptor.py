"""
Step 2 — W4 Normal run with interceptor.
"""

import asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from log_events import log_graph_start, log_graph_end, log_llm_call
from pathlib import Path
import log_events

load_dotenv()

log_events.LOG_FILE = Path("traces/trace_normal.jsonl")
log_events.LOG_FILE.parent.mkdir(exist_ok=True)
log_events.LOG_FILE.write_text("")
log_events._run_id = __import__("uuid").uuid4().__str__()
log_events._seq = 0
log_events._last_event_id = None

from state import CodingState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

TASK = (
    "Write a Python function called find_first_adjacent_duplicate(items) "
    "that returns the first item that appears in two consecutive positions "
    "in a list. Return None if no such item exists."
)

BUGGY_CODE = """def find_first_adjacent_duplicate(items):
    \"\"\"Return the first item that appears in two consecutive positions.\"\"\"
    for i in range(len(items)):
        if items[i] == items[i + 1]:
            return items[i]
    return None"""

# ── Instrumented agents ───────────────────────────────────────────────────────

def coder_agent(state: dict) -> dict:
    iteration        = state.get("iteration", 1)
    critic_diagnosis = state.get("critic_diagnosis", "")
    current_code     = state.get("current_code", "")

    print(f"\n[Coder] Iteration {iteration}")

    if iteration == 1:
        print(f"[Coder] Submitting known buggy code")
        # no LLM call on iter 1 — log as a special event
        log_llm_call(
            [type("M", (), {"type": "human",
             "content": f"[iter=1] Submit buggy code"})()],
            type("R", (), {"content": BUGGY_CODE, "tool_calls": []})()
        )
        return {
            "current_code": BUGGY_CODE,
            "messages": [AIMessage(content="[Coder iter=1] Buggy code submitted.",
                                   name="coder")],
        }

    prompt = f"""You are an expert Python developer fixing a bug.

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

    log_llm_call(
        [type("M", (), {"type": "human", "content": prompt})()],
        response
    )
    print(f"[Coder] Fixed code:\n{code}\n")
    return {
        "current_code": code,
        "messages": [AIMessage(content=f"[Coder iter={iteration}] Fix applied.",
                               name="coder")],
    }


def test_runner(state: dict) -> dict:
    import traceback
    code      = state.get("current_code", "")
    iteration = state.get("iteration", 1)
    print(f"\n[Test Runner] Iteration {iteration}")

    test_cases = [
        ([1, 2, 2, 3], 2), ([1, 1, 2, 3], 1),
        ([1, 2, 3, 3], 3), ([1, 2, 3, 4], None), ([5, 5], 5),
    ]
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        result = f"TESTS FAILED\nCode failed to execute: {e}"
        print(f"[Test Runner] {result}")
        return {"test_result": result,
                "messages": [AIMessage(content="[Test Runner] FAILED",
                                       name="test_runner")]}

    func = next((v for k, v in namespace.items()
                 if callable(v) and not k.startswith("_")), None)
    if not func:
        result = "TESTS FAILED\nNo callable function found."
        return {"test_result": result,
                "messages": [AIMessage(content="[Test Runner] FAILED",
                                       name="test_runner")]}

    results = []
    passed  = 0
    for i, (inp, exp) in enumerate(test_cases):
        try:
            got = func(inp)
            ok  = got == exp
            if ok: passed += 1
            results.append(f"  Test {i+1}: {inp} expected={exp} got={got} "
                           f"→ {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            results.append(f"  Test {i+1}: {inp} → ERROR: {type(e).__name__}: {e}")

    summary = "TESTS PASSED" if passed == len(test_cases) else "TESTS FAILED"
    result  = f"{summary} ({passed}/{len(test_cases)})\n" + "\n".join(results)
    print(f"[Test Runner] {summary}")

    # log as tool call
    import log_events as le
    le._new_event("TOOL_CALL", "test_runner",
                  {"tool": "test_runner", "input": {"iteration": iteration}},
                  {"output": result})

    return {"test_result": result,
            "messages": [AIMessage(
                content=f"[Test Runner iter={iteration}] {summary}",
                name="test_runner")]}


def critic_agent(state: dict) -> dict:
    code        = state.get("current_code", "")
    test_result = state.get("test_result", "")
    iteration   = state.get("iteration", 1)
    print(f"\n[Critic] Iteration {iteration}")

    prompt = f"""You are an expert code reviewer analysing a test failure.

CURRENT CODE:
{code}

TEST OUTPUT:
{test_result}

Based ONLY on the test output, provide a precise diagnosis:

ROOT CAUSE: [one sentence]
LOCATION:   [specific line or expression]
FIX:        [exact change to make]
"""
    response = llm.invoke(prompt)
    log_llm_call(
        [type("M", (), {"type": "human", "content": prompt})()],
        response
    )
    print(f"[Critic] Diagnosis:\n{response.content}\n")
    return {
        "critic_diagnosis": response.content,
        "messages": [AIMessage(content=f"[Critic iter={iteration}] Done.",
                               name="critic")],
    }


def loop_controller(state: dict) -> dict:
    test_result    = state.get("test_result", "")
    iteration      = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 4)
    passed = "PASSED" in test_result and "FAILED" not in test_result

    if passed:
        print(f"\n[Loop Controller] PASSED at iteration {iteration} → END")
        return {"status": "PASSED", "iteration": iteration,
                "messages": [AIMessage(content="PASSED", name="loop_controller")]}
    if iteration >= max_iterations:
        print(f"\n[Loop Controller] Max iterations reached → END")
        return {"status": "FAILED_MAX_ITER", "iteration": iteration,
                "messages": [AIMessage(content="MAX_ITER", name="loop_controller")]}

    next_iter = iteration + 1
    print(f"\n[Loop Controller] Iterating → {next_iter}")
    return {"status": "RUNNING", "iteration": next_iter,
            "messages": [AIMessage(content=f"ITERATE→{next_iter}",
                                   name="loop_controller")]}


def route_after_controller(state: dict) -> str:
    status = state.get("status", "RUNNING")
    return "end" if status in ("PASSED", "FAILED_MAX_ITER") else "critic"


# ── Build graph ───────────────────────────────────────────────────────────────

builder = StateGraph(CodingState)
builder.add_node("coder",           coder_agent)
builder.add_node("test_runner",     test_runner)
builder.add_node("loop_controller", loop_controller)
builder.add_node("critic",          critic_agent)
builder.set_entry_point("coder")
builder.add_edge("coder",           "test_runner")
builder.add_edge("test_runner",     "loop_controller")
builder.add_conditional_edges("loop_controller", route_after_controller,
                               {"end": END, "critic": "critic"})
builder.add_edge("critic",          "coder")
graph = builder.compile()


async def main():
    print("=" * 60)
    print("  W4 Normal Run — with interceptor")
    print("=" * 60)

    log_graph_start(TASK)

    result = await graph.ainvoke({
        "messages": [], "task": TASK,
        "current_code": "", "test_result": "",
        "critic_diagnosis": "", "iteration": 1,
        "max_iterations": 4, "status": "RUNNING",
        "inject_failure": False,
    })

    log_graph_end(f"status={result.get('status')} "
                  f"iterations={result.get('iteration')}")

    print(f"\n=== Trace written to traces/trace_normal.jsonl ===")

asyncio.run(main())