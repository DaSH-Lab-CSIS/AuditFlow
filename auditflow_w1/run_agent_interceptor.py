"""Step 3: Run the agent and log all events."""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from typing import Dict, List, Literal, cast
from react_agent.state import InputState, State
from log_events import log_llm_call, log_tool_call, log_graph_start, log_graph_end

load_dotenv()

tavily = TavilySearch(max_results=3)

async def search(query: str) -> dict:
    """Search the web for current information."""
    return await tavily.ainvoke({"query": query})

TOOLS = [search]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)


async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    system_message = "You are a helpful AI assistant."
    messages = [{"role": "system", "content": system_message}, *state.messages]

    response = cast(AIMessage, await model.ainvoke(messages))

    # ── log the LLM call ──
    log_llm_call(state.messages, response)

    if state.is_last_step and response.tool_calls:
        return {"messages": [AIMessage(id=response.id,
                content="Sorry, I could not find an answer in the specified number of steps.")]}
    return {"messages": [response]}


async def tools_node(state: State) -> Dict[str, list]:
    """Wrap ToolNode so we can log each tool call."""
    tool_node = ToolNode(TOOLS)
    result = await tool_node.ainvoke(state)

    # log each tool message that came back
    last_ai = next(
        (m for m in reversed(state.messages) if isinstance(m, AIMessage)), None
    )
    if last_ai:
        for tc in last_ai.tool_calls:
            # find matching tool message in result
            tool_msg = next(
                (m for m in result["messages"] if isinstance(m, ToolMessage)
                 and m.tool_call_id == tc["id"]),
                None
            )
            output = tool_msg.content if tool_msg else "no output"
            log_tool_call(tc["name"], tc["args"], output)

    return result


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")
    if not last_message.tool_calls:
        return "__end__"
    return "tools"


# Build graph
builder = StateGraph(State, input_schema=InputState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tools_node)
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")
graph = builder.compile()


async def main():
    query = "What is the weather like in Paris today?"
    print(f"\nQuery: {query}\n")

    log_graph_start(query)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]}
    )

    # final answer is the last AI message
    final = next(
        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), ""
    )
    log_graph_end(final)

    print("\n=== Agent finished ===")
    for msg in result["messages"]:
        print(f"\n[{msg.type.upper()}]: {str(msg.content)[:400]}")

    print(f"\n=== Trace written to traces/trace.jsonl ===")

asyncio.run(main())