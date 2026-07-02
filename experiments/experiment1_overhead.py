"""
Experiment 1 — Runtime Overhead (RQ1)
======================================
Measures end-to-end latency of each workflow with AuditFlow
ON vs OFF. Uses Ollama llama3.2 (local) to eliminate network
variance so AuditFlow's true overhead is measurable.

Every individual run latency is saved — no summary-only storage.
Runs N=100 total, results saved at N=30, N=50, N=100 checkpoints
for convergence analysis.

Usage:
    python experiment1_overhead.py

Results saved to:
    results/raw/        — individual run latencies (JSON)
    results/summary/    — summary stats per checkpoint
"""

import asyncio
import hashlib
import json
import sys
import time
import uuid
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE        = Path(__file__).parent.parent
RAW_DIR     = Path("results/raw");     RAW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = Path("results/summary"); SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS      = 100
CHECKPOINTS = [30, 50, 100]


def get_llm():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0)
    except ImportError:
        raise RuntimeError(
            "Run: pip install langchain-ollama"
        )


class AuditLogger:
    def __init__(self, enabled: bool, log_file: Path):
        self.enabled      = enabled
        self.log_file     = log_file
        self.run_id       = str(uuid.uuid4())
        self._seq         = 0
        self._last_id     = None
        self.t_intercept  = 0.0
        self.t_serialise  = 0.0
        self.t_io         = 0.0
        self.n_events     = 0
        if enabled and str(log_file) != "/dev/null":
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("")

    def _emit(self, etype, node, inp, out):
        if not self.enabled:
            return
        t0 = time.perf_counter()
        self._seq += 1
        eid   = str(uuid.uuid4())
        event = {"event_id": eid, "run_id": self.run_id,
                 "seq": self._seq, "type": etype, "node": node,
                 "timestamp": round(time.time(), 4),
                 "parent_id": self._last_id,
                 "input": inp, "output": out}
        t1 = time.perf_counter()
        self.t_intercept += (t1 - t0) * 1000

        t2 = time.perf_counter()
        payload = {k: v for k, v in event.items()}
        event["hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True,
                       default=str).encode()).hexdigest()[:16]
        line = json.dumps(event, default=str)
        t3 = time.perf_counter()
        self.t_serialise += (t3 - t2) * 1000

        t4 = time.perf_counter()
        if str(self.log_file) != "/dev/null":
            with open(self.log_file, "a") as f:
                f.write(line + "\n")
        t5 = time.perf_counter()
        self.t_io    += (t5 - t4) * 1000
        self._last_id = eid
        self.n_events += 1

    def graph_start(self, q):
        self._emit("GRAPH_START", "graph", {"query": q[:100]}, {})

    def graph_end(self, a):
        self._emit("GRAPH_END", "graph", {}, {"answer": a[:100]})

    def llm_call(self, msgs, content, tool_calls):
        self._emit("LLM_CALL", "call_model",
                   {"n_messages": len(msgs),
                    "preview": str(msgs[0])[:80] if msgs else ""},
                   {"content": content[:150],
                    "tool_calls": tool_calls,
                    "has_tools": len(tool_calls) > 0})

    def tool_call(self, tool, inp, out):
        self._emit("TOOL_CALL", "tools",
                   {"tool": tool, "input": inp},
                   {"output": out[:150]})

    def component_breakdown(self):
        n = max(self.n_events, 1)
        return {
            "interception_ms":  round(self.t_intercept  / n, 4),
            "serialisation_ms": round(self.t_serialise  / n, 4),
            "io_ms":            round(self.t_io          / n, 4),
            "total_ms":         round((self.t_intercept +
                                       self.t_serialise +
                                       self.t_io) / n, 4),
            "n_events":         self.n_events,
        }


async def run_w1_email(llm, logger):
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    from typing import Dict, List, Literal, cast
    sys.path.insert(0, str(BASE / "auditflow_w1/react-agent-main/src"))
    from react_agent.state import InputState, State

    @tool
    def search_emails(query: str) -> dict:
        """Search inbox emails by keyword."""
        return {"results": [
            {"id": "email_001", "from": "newsletter@tech.com",
             "subject": "Weekly digest", "date": "2026-05-07"},
            {"id": "email_002", "from": "newsletter@medium.com",
             "subject": "Your digest", "date": "2026-05-06"},
        ], "count": 2}

    @tool
    def archive_emails(email_ids: list) -> dict:
        """Archive emails by ID."""
        return {"archived": len(email_ids), "status": "SUCCESS"}

    TOOLS = [search_emails, archive_emails]
    bound = llm.bind_tools(TOOLS)

    async def call_model(state: State) -> Dict[str, List[AIMessage]]:
        msgs = [{"role": "system",
                 "content": "You are a helpful email assistant. "
                            "Use search_emails then archive_emails."},
                *state.messages]
        response = cast(AIMessage, await bound.ainvoke(msgs))
        logger.llm_call(state.messages, str(response.content),
                        getattr(response, "tool_calls", []))
        if state.is_last_step and response.tool_calls:
            return {"messages": [AIMessage(id=response.id,
                    content="Max steps.")]}
        return {"messages": [response]}

    async def tools_node(state: State) -> Dict[str, list]:
        tn = ToolNode(TOOLS)
        result = await tn.ainvoke(state)
        last = next((m for m in reversed(state.messages)
                     if isinstance(m, AIMessage)), None)
        if last:
            for tc in last.tool_calls:
                tm = next((m for m in result["messages"]
                           if isinstance(m, ToolMessage)
                           and m.tool_call_id == tc["id"]), None)
                logger.tool_call(tc["name"], tc["args"],
                                 tm.content if tm else "")
        return result

    def route(state: State) -> Literal["__end__", "tools"]:
        last = state.messages[-1]
        if not isinstance(last, AIMessage):
            return "__end__"
        return "tools" if last.tool_calls else "__end__"

    builder = StateGraph(State, input_schema=InputState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tools_node)
    builder.add_edge("__start__", "call_model")
    builder.add_conditional_edges("call_model", route)
    builder.add_edge("tools", "call_model")
    graph = builder.compile()

    query = "Find all newsletter emails and archive them."
    logger.graph_start(query)
    t0 = time.perf_counter()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]})
    t1 = time.perf_counter()
    final = next((m.content for m in reversed(result["messages"])
                  if isinstance(m, AIMessage)), "")
    logger.graph_end(str(final))
    return (t1 - t0) * 1000


async def run_w2_codereview(llm, logger):
    from langchain_core.messages import AIMessage
    from langgraph.graph import StateGraph, END

    spec = importlib.util.spec_from_file_location(
        "state_w2", BASE / "auditflow_w2/state.py")
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    WorkflowState = sm.WorkflowState

    BUGGY_CODE = (
        "def find_first_adjacent_duplicate(items):\n"
        "    for i in range(len(items)):\n"
        "        if items[i] == items[i + 1]:\n"
        "            return items[i]\n"
        "    return None"
    )

    def orchestrator(state):
        if not state.get("analyzer_output"):
            return {"next_agent": "static_analyzer",
                    "messages": [AIMessage(content="→a",
                                           name="orchestrator")]}
        if not state.get("reviewer_output"):
            return {"next_agent": "code_reviewer",
                    "messages": [AIMessage(content="→r",
                                           name="orchestrator")]}
        if not state.get("patch_output"):
            return {"next_agent": "patch_generator",
                    "messages": [AIMessage(content="→p",
                                           name="orchestrator")]}
        return {"next_agent": "END", "final_report": "done",
                "messages": [AIMessage(content="END",
                                       name="orchestrator")]}

    def make_agent(prompt_fn, output_key):
        async def agent(state):
            prompt = prompt_fn(state)
            response = await llm.ainvoke(prompt)
            logger.llm_call([prompt], str(response.content), [])
            return {output_key: str(response.content)[:300],
                    "messages": [AIMessage(content=f"[{output_key}]",
                                           name=output_key)]}
        return agent

    analyzer = make_agent(
        lambda s: (f"Analyse this Python code for bugs:\n{BUGGY_CODE}\n"
                   "Reply: BUG TYPE / LOCATION / DESCRIPTION / SEVERITY"),
        "analyzer_output")
    reviewer = make_agent(
        lambda s: (f"Code review based on:\n"
                   f"{s.get('analyzer_output','')}\nCode:\n{BUGGY_CODE}"),
        "reviewer_output")
    patcher = make_agent(
        lambda s: (f"Fix this code:\n{BUGGY_CODE}\n"
                   f"Review:\n{s.get('reviewer_output','')}"),
        "patch_output")

    builder = StateGraph(WorkflowState)
    builder.add_node("orchestrator",    orchestrator)
    builder.add_node("static_analyzer", analyzer)
    builder.add_node("code_reviewer",   reviewer)
    builder.add_node("patch_generator", patcher)
    builder.set_entry_point("orchestrator")
    builder.add_conditional_edges("orchestrator",
        lambda s: s.get("next_agent", "END"), {
            "static_analyzer": "static_analyzer",
            "code_reviewer":   "code_reviewer",
            "patch_generator": "patch_generator",
            "END":             END})
    builder.add_edge("static_analyzer", "orchestrator")
    builder.add_edge("code_reviewer",   "orchestrator")
    builder.add_edge("patch_generator", "orchestrator")
    graph = builder.compile()

    logger.graph_start("Code review pipeline")
    t0 = time.perf_counter()
    result = await graph.ainvoke({
        "messages": [], "code_snippet": BUGGY_CODE,
        "analyzer_output": "", "reviewer_output": "",
        "patch_output": "", "final_report": "",
        "next_agent": "", "inject_failure": False})
    t1 = time.perf_counter()
    logger.graph_end(result.get("patch_output", "")[:80])
    return (t1 - t0) * 1000


async def run_w3_research(llm, logger):
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    from typing import Dict, List, Literal, cast
    sys.path.insert(0, str(BASE / "auditflow_w1/react-agent-main/src"))
    from react_agent.state import InputState, State

    @tool
    async def search(query: str) -> dict:
        """Search the web for information."""
        return {"query": query, "results": [
            {"title": "LLM Security",
             "url": "https://example.com/1",
             "content": "Prompt injection and data poisoning are key threats."},
            {"title": "Agent Security",
             "url": "https://example.com/2",
             "content": "Multi-agent cascading failures are a growing concern."},
            {"title": "OWASP Top 10",
             "url": "https://owasp.org/llm",
             "content": "Top LLM risks include prompt injection and supply chain."},
        ]}

    TOOLS = [search]
    bound = llm.bind_tools(TOOLS)

    async def call_model(state: State) -> Dict[str, List[AIMessage]]:
        msgs = [{"role": "system",
                 "content": "You are a research assistant. Search and summarise."},
                *state.messages]
        response = cast(AIMessage, await bound.ainvoke(msgs))
        logger.llm_call(state.messages, str(response.content),
                        getattr(response, "tool_calls", []))
        if state.is_last_step and response.tool_calls:
            return {"messages": [AIMessage(id=response.id,
                    content="Max steps.")]}
        return {"messages": [response]}

    async def tools_node(state: State) -> Dict[str, list]:
        tn = ToolNode(TOOLS)
        result = await tn.ainvoke(state)
        last = next((m for m in reversed(state.messages)
                     if isinstance(m, AIMessage)), None)
        if last:
            for tc in last.tool_calls:
                tm = next((m for m in result["messages"]
                           if isinstance(m, ToolMessage)
                           and m.tool_call_id == tc["id"]), None)
                logger.tool_call(tc["name"], tc["args"],
                                 tm.content if tm else "")
        return result

    def route(state: State) -> Literal["__end__", "tools"]:
        last = state.messages[-1]
        if not isinstance(last, AIMessage):
            return "__end__"
        return "tools" if last.tool_calls else "__end__"

    builder = StateGraph(State, input_schema=InputState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tools_node)
    builder.add_edge("__start__", "call_model")
    builder.add_conditional_edges("call_model", route)
    builder.add_edge("tools", "call_model")
    graph = builder.compile()

    query = "What are the latest LLM agent security vulnerabilities?"
    logger.graph_start(query)
    t0 = time.perf_counter()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]})
    t1 = time.perf_counter()
    final = next((m.content for m in reversed(result["messages"])
                  if isinstance(m, AIMessage)), "")
    logger.graph_end(str(final))
    return (t1 - t0) * 1000


async def run_w4_coding(llm, logger):
    from langchain_core.messages import AIMessage
    from langgraph.graph import StateGraph, END

    spec = importlib.util.spec_from_file_location(
        "state_w4", BASE / "auditflow_w4/state.py")
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    CodingState = sm.CodingState

    BUGGY_CODE = (
        "def find_first_adjacent_duplicate(items):\n"
        "    for i in range(len(items)):\n"
        "        if items[i] == items[i + 1]:\n"
        "            return items[i]\n"
        "    return None"
    )
    test_cases = [([1,2,2,3],2),([1,1,2,3],1),
                  ([1,2,3,3],3),([1,2,3,4],None),([5,5],5)]

    def run_tests(code):
        ns = {}
        try:
            exec(code, ns)
        except Exception as e:
            return f"TESTS FAILED\n{e}", False
        func = next((v for k,v in ns.items()
                     if callable(v) and not k.startswith("_")), None)
        if not func:
            return "TESTS FAILED", False
        passed = 0
        lines  = []
        for inp, exp in test_cases:
            try:
                got = func(list(inp))
                ok  = got == exp
                if ok: passed += 1
                lines.append(f"{'PASS' if ok else 'FAIL'}:{inp}→{got}")
            except Exception as e:
                lines.append(f"ERR:{inp}→{e}")
        ok = passed == len(test_cases)
        return (f"TESTS {'PASSED' if ok else 'FAILED'} "
                f"({passed}/{len(test_cases)})\n"+"\n".join(lines)), ok

    async def coder(state):
        it   = state.get("iteration", 1)
        diag = state.get("critic_diagnosis", "")
        code = state.get("current_code", "")
        if it == 1:
            logger.llm_call([], BUGGY_CODE, [])
            return {"current_code": BUGGY_CODE,
                    "messages": [AIMessage(content="buggy", name="coder")]}
        prompt   = (f"Fix this Python code:\n{code}\n"
                    f"Diagnosis:\n{diag}\n"
                    "Return ONLY corrected code, no markdown.")
        response = await llm.ainvoke(prompt)
        fixed    = response.content.strip()
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            fixed = "\n".join(lines[1:-1])
        logger.llm_call([prompt], fixed, [])
        return {"current_code": fixed,
                "messages": [AIMessage(content="fixed", name="coder")]}

    def test_runner(state):
        result, passed = run_tests(state.get("current_code", ""))
        logger.tool_call("test_runner",
                         {"iteration": state.get("iteration", 1)}, result)
        return {"test_result": result,
                "messages": [AIMessage(
                    content="PASSED" if passed else "FAILED",
                    name="test_runner")]}

    async def critic(state):
        prompt   = (f"Analyse test failure:\nCODE:\n"
                    f"{state.get('current_code','')}\n"
                    f"TESTS:\n{state.get('test_result','')}\n"
                    "ROOT CAUSE / LOCATION / FIX:")
        response = await llm.ainvoke(prompt)
        logger.llm_call([prompt], str(response.content), [])
        return {"critic_diagnosis": str(response.content),
                "messages": [AIMessage(content="diagnosed", name="critic")]}

    def loop_ctrl(state):
        tr = state.get("test_result", "")
        it = state.get("iteration", 1)
        mx = state.get("max_iterations", 4)
        ok = "PASSED" in tr and "FAILED" not in tr
        if ok:
            return {"status": "PASSED", "iteration": it,
                    "messages": [AIMessage(content="PASSED", name="lc")]}
        if it >= mx:
            return {"status": "FAILED_MAX_ITER", "iteration": it,
                    "messages": [AIMessage(content="MAX", name="lc")]}
        return {"status": "RUNNING", "iteration": it+1,
                "messages": [AIMessage(content=f"→{it+1}", name="lc")]}

    builder = StateGraph(CodingState)
    builder.add_node("coder",           coder)
    builder.add_node("test_runner",     test_runner)
    builder.add_node("loop_controller", loop_ctrl)
    builder.add_node("critic",          critic)
    builder.set_entry_point("coder")
    builder.add_edge("coder",           "test_runner")
    builder.add_edge("test_runner",     "loop_controller")
    builder.add_conditional_edges("loop_controller",
        lambda s: "end" if s.get("status") in
                  ("PASSED","FAILED_MAX_ITER") else "critic",
        {"end": END, "critic": "critic"})
    builder.add_edge("critic", "coder")
    graph = builder.compile()

    logger.graph_start("Autonomous coding agent")
    t0 = time.perf_counter()
    result = await graph.ainvoke({
        "messages": [], "task": "find_first_adjacent_duplicate",
        "current_code": "", "test_result": "",
        "critic_diagnosis": "", "iteration": 1,
        "max_iterations": 4, "status": "RUNNING",
        "inject_failure": False})
    t1 = time.perf_counter()
    logger.graph_end(f"status={result.get('status')}")
    return (t1 - t0) * 1000


WORKFLOWS = {
    "W1_email":      run_w1_email,
    "W2_codereview": run_w2_codereview,
    "W3_research":   run_w3_research,
    "W4_coding":     run_w4_coding,
}

WF_LABELS = {
    "W1_email":      "W1: Email Triage (Single-agent)",
    "W2_codereview": "W2: Code Review (Multi-agent)",
    "W3_research":   "W3: Web Research (Tool-augmented)",
    "W4_coding":     "W4: Coding Agent (Feedback-driven)",
}


def _save_checkpoint(wf, condition, latencies, components, n):
    import statistics as st
    s = sorted(latencies)
    summary = {
        "workflow":  wf, "condition": condition, "n": n,
        "mean":      round(st.mean(s), 3),
        "median":    round(st.median(s), 3),
        "stdev":     round(st.stdev(s), 3) if len(s) > 1 else 0,
        "p95":       round(s[int(len(s)*0.95)], 3),
        "min":       round(min(s), 3),
        "max":       round(max(s), 3),
        "latencies": latencies[:],
    }
    if components:
        summary["components"] = {
            "interception_ms":  round(sum(c["interception_ms"]
                                for c in components)/len(components), 4),
            "serialisation_ms": round(sum(c["serialisation_ms"]
                                for c in components)/len(components), 4),
            "io_ms":            round(sum(c["io_ms"]
                                for c in components)/len(components), 4),
        }
    path = SUMMARY_DIR / f"{wf}_{condition}_n{n}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  checkpoint saved → {path.name}")


async def run_experiment():
    print("\n" + "="*65)
    print("  Experiment 1 — Runtime Overhead (RQ1)")
    print(f"  N={N_RUNS} runs | Model: Ollama llama3.2 (local)")
    print("="*65)

    llm = get_llm()
    try:
        await llm.ainvoke("say ok")
        print("  Ollama llama3.2 ready ✓\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    all_raw = {}

    for wf_name, wf_fn in WORKFLOWS.items():
        print(f"\n{'─'*65}")
        print(f"  {WF_LABELS[wf_name]}")
        print(f"{'─'*65}")

        latencies_on  = []
        latencies_off = []
        components_on = []

        # AuditFlow ON
        print(f"\n  [AuditFlow ON]  {N_RUNS} trials...")
        for i in range(N_RUNS):
            log_file = RAW_DIR / f"tmp_{wf_name}_{i}.jsonl"
            logger   = AuditLogger(enabled=True, log_file=log_file)
            try:
                ms = await wf_fn(llm, logger)
                latencies_on.append(round(ms, 3))
                components_on.append(logger.component_breakdown())
                log_file.unlink(missing_ok=True)
            except Exception as e:
                print(f"\n  run {i} failed: {e}")
                continue
            done = len(latencies_on)
            bar  = "█"*(done*20//N_RUNS) + "░"*(20-done*20//N_RUNS)
            print(f"  [{bar}] {done}/{N_RUNS}  {ms:.0f}ms", end="\r")
            if done in CHECKPOINTS:
                _save_checkpoint(wf_name, "on", latencies_on,
                                 components_on, done)
        print(f"  [{'█'*20}] {len(latencies_on)}/{N_RUNS} done{' '*20}")

        # AuditFlow OFF
        print(f"\n  [AuditFlow OFF] {N_RUNS} trials...")
        for i in range(N_RUNS):
            logger = AuditLogger(enabled=False, log_file=Path("/dev/null"))
            try:
                ms = await wf_fn(llm, logger)
                latencies_off.append(round(ms, 3))
            except Exception as e:
                print(f"\n  run {i} failed: {e}")
                continue
            done = len(latencies_off)
            bar  = "█"*(done*20//N_RUNS) + "░"*(20-done*20//N_RUNS)
            print(f"  [{bar}] {done}/{N_RUNS}  {ms:.0f}ms", end="\r")
            if done in CHECKPOINTS:
                _save_checkpoint(wf_name, "off", latencies_off, [], done)
        print(f"  [{'█'*20}] {len(latencies_off)}/{N_RUNS} done{' '*20}")

        all_raw[wf_name] = {
            "on":            latencies_on,
            "off":           latencies_off,
            "components_on": components_on,
            "label":         WF_LABELS[wf_name],
        }

        if latencies_on and latencies_off:
            mo  = sum(latencies_on)  / len(latencies_on)
            mof = sum(latencies_off) / len(latencies_off)
            oh  = mo - mof
            print(f"\n  OFF: {mof:.1f}ms  ON: {mo:.1f}ms  "
                  f"Overhead: {oh:+.1f}ms ({oh/mof*100:+.1f}%)")

    # save all raw data
    raw_path = RAW_DIR / "all_latencies.json"
    with open(raw_path, "w") as f:
        json.dump(all_raw, f, indent=2)
    print(f"\n{'='*65}")
    print(f"  All raw data saved → {raw_path}")
    print(f"  Run experiment1_plots.py to generate figures.")
    print(f"{'='*65}\n")


asyncio.run(run_experiment())