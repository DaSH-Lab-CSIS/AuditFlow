"""
W1 Normal run — Email triage agent.
Agent searches emails and archives the right ones. No failure.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from typing import Dict, List, Literal, cast
from react_agent.state import InputState, State
from log_events import log_llm_call, log_tool_call, log_graph_start, log_graph_end
from pathlib import Path
import log_events

load_dotenv()

# ── reset logger for this run ─────────────────────────────────────────────────
log_events.LOG_FILE = Path("traces/trace_normal.jsonl")
log_events.LOG_FILE.parent.mkdir(exist_ok=True)
log_events.LOG_FILE.write_text("") 
log_events._run_id = __import__("uuid").uuid4().__str__()
log_events._seq = 0
log_events._last_event_id = None

# ── Email tools ───────────────────────────────────────────────────────────────

@tool
def search_emails(query: str) -> dict:
    """Search inbox emails by keyword."""
    return {
        "results": [
            {"id": "email_001", "from": "newsletter@techdigest.com",
             "subject": "Tech Digest Weekly", "date": "2026-05-07"},
            {"id": "email_002", "from": "newsletter@medium.com",
             "subject": "Your Medium digest", "date": "2026-05-06"},
            {"id": "email_003", "from": "newsletter@producthunt.com",
             "subject": "Top products this week", "date": "2026-05-05"},
        ],
        "count": 3
    }

@tool
def archive_emails(email_ids: list) -> dict:
    """Archive emails by ID. This is reversible."""
    return {
        "archived": len(email_ids),
        "email_ids": email_ids,
        "status": "SUCCESS"
    }

@tool
def delete_emails(email_ids: list) -> dict:
    """Permanently delete emails by ID. IRREVERSIBLE."""
    return {
        "deleted": len(email_ids),
        "email_ids": email_ids,
        "status": "SUCCESS",
        "warning": "IRREVERSIBLE"
    }

TOOLS = [search_emails, archive_emails, delete_emails]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)

# ── Graph nodes ───────────────────────────────────────────────────────────────

async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    system_message = (
        "You are a helpful email assistant. "
        "Help the user manage their inbox by searching, archiving, or deleting emails."
    )
    messages = [{"role": "system", "content": system_message}, *state.messages]
    response = cast(AIMessage, await model.ainvoke(messages))
    log_llm_call(state.messages, response)
    if state.is_last_step and response.tool_calls:
        return {"messages": [AIMessage(id=response.id,
                content="Sorry, could not complete in the allowed steps.")]}
    return {"messages": [response]}


async def tools_node(state: State) -> Dict[str, list]:
    tool_node = ToolNode(TOOLS)
    result = await tool_node.ainvoke(state)
    last_ai = next((m for m in reversed(state.messages)
                    if isinstance(m, AIMessage)), None)
    if last_ai:
        for tc in last_ai.tool_calls:
            tool_msg = next(
                (m for m in result["messages"]
                 if isinstance(m, ToolMessage) and m.tool_call_id == tc["id"]), None
            )
            log_tool_call(tc["name"], tc["args"], tool_msg.content if tool_msg else "")
    return result


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last = state.messages[-1]
    if not isinstance(last, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last).__name__}")
    return "tools" if last.tool_calls else "__end__"


builder = StateGraph(State, input_schema=InputState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tools_node)
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")
graph = builder.compile()

# ── Run ───────────────────────────────────────────────────────────────────────

async def main():
    query = "Find all newsletter emails and archive them."
    print(f"\nQuery: {query}\n")

    log_graph_start(query)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]}
    )

    final = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage)), ""
    )
    log_graph_end(final)

    print("\n=== Agent finished ===")
    for msg in result["messages"]:
        print(f"\n[{msg.type.upper()}]: {str(msg.content)[:400]}")

    print("\n=== Trace written to traces/trace_normal.jsonl ===")

asyncio.run(main())