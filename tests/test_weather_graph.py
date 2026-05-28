"""Smoke test: the weather agent routes a tool call through get_weather."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from autonomous_ct.agents.weather import build_weather_graph

from ._fakes import ScriptedChatModel


def test_weather_agent_routes_through_tool_and_terminates() -> None:
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "Tokyo"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    final = AIMessage(content="It's sunny and 75 degrees in Tokyo!")
    fake = ScriptedChatModel(responses=[tool_call, final])

    app = build_weather_graph(llm=fake)
    result = app.invoke({"messages": [HumanMessage(content="weather in Tokyo?")]})

    contents = [str(m.content) for m in result["messages"]]
    assert any("75 degrees in Tokyo" in c for c in contents)


def test_weather_agent_terminates_without_tool_call() -> None:
    direct_reply = AIMessage(content="Hello Bob!")
    fake = ScriptedChatModel(responses=[direct_reply])

    app = build_weather_graph(llm=fake)
    result = app.invoke({"messages": [HumanMessage(content="Hi, I'm Bob.")]})

    assert result["messages"][-1].content == "Hello Bob!"
