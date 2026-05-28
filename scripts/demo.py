"""Interactive demo: the original Test 1/2/3 scenarios from main.py.

Run with:  uv run python scripts/demo.py
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from autonomous_ct.agents.weather import build_weather_graph


def main() -> None:
    app = build_weather_graph()

    # Test 1: Casual conversation (agent should NOT use tools).
    result = app.invoke({"messages": [HumanMessage(content="Hi! I'm Bob.")]})
    print("Agent:", result["messages"][-1].content)

    # Test 2: Weather query (agent SHOULD use the weather tool).
    result = app.invoke(
        {"messages": [HumanMessage(content="What is the weather like in Tokyo?")]}
    )
    print("Agent:", result["messages"][-1].content)

    # Test 3: Meta-question about the underlying model.
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content="What is the model being used to answer requests?")
            ]
        }
    )
    print("Agent:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
