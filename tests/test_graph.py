"""Graph smoke tests using a stub chat model and a generic tool — no network calls."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from autonomous_ct.graph import build_graph

from ._fakes import ScriptedChatModel


@tool
def echo(text: str) -> str:
    """Echo the input back unchanged."""
    return text


def test_graph_routes_through_tool_and_terminates() -> None:
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "echo",
                "args": {"text": "hello"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    final_message = AIMessage(content="The tool said: hello")
    fake = ScriptedChatModel(responses=[tool_call_message, final_message])

    app = build_graph(llm=fake, tools=[echo])
    result = app.invoke({"messages": [HumanMessage(content="echo hello")]})

    contents = [str(m.content) for m in result["messages"]]
    assert any("The tool said: hello" in c for c in contents)
    assert any(c == "hello" for c in contents)


def test_graph_terminates_without_tool_call() -> None:
    direct_reply = AIMessage(content="Hello Bob!")
    fake = ScriptedChatModel(responses=[direct_reply])

    app = build_graph(llm=fake, tools=[echo])
    result = app.invoke({"messages": [HumanMessage(content="Hi, I'm Bob.")]})

    assert result["messages"][-1].content == "Hello Bob!"
