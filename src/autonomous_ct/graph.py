"""LangGraph wiring for the autonomous-ct agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from .llm import build_llm

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(
    llm: BaseChatModel | None = None,
    tools: Sequence[Callable[..., Any]] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile and return a tool-using agent graph.

    Domain agents (see :mod:`autonomous_ct.agents`) are expected to supply
    their own ``tools`` and ``system_prompt``; this function intentionally
    has no knowledge of any specific tool or domain.

    When a ``checkpointer`` is provided, the compiled app persists message
    state per ``thread_id`` (passed via ``config={"configurable":
    {"thread_id": "..."}}`` at ``invoke`` time). Callers can then resume
    conversations and rely on the ``add_messages`` reducer to accumulate
    history server-side, instead of re-sending the whole transcript.
    """
    resolved_tools = list(tools) if tools is not None else []
    chat = (llm if llm is not None else build_llm()).bind_tools(resolved_tools)
    system_message = SystemMessage(content=system_prompt)

    def agent_node(state: AgentState) -> dict:
        response = chat.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(resolved_tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)
