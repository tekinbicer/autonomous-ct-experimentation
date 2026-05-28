"""Legacy weather demo agent.

Kept as a minimal smoke-test agent so the LangGraph wiring (agent <-> tools
cycle, system-prompt injection, Argo LLM client) can be exercised end-to-end
without requiring Docker, GPUs, or a built ``tomocupy`` image.

It is intentionally trivial: a single deterministic tool and a short system
prompt. New domain agents should follow the same shape as
:mod:`autonomous_ct.agents.imaging` rather than extending this one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel

from ..graph import build_graph
from ..tools import WEATHER_TOOLS

WEATHER_SYSTEM_PROMPT = (
    "You are a helpful weather assistant. "
    "Always use the weather tool if asked about the weather."
)


def build_weather_graph(
    llm: BaseChatModel | None = None,
    extra_tools: Sequence[Callable[..., Any]] | None = None,
) -> Any:
    """Compile the legacy weather demo agent graph.

    Parameters
    ----------
    llm:
        Optional pre-built chat model. Defaults to the project-configured Argo LLM.
    extra_tools:
        Additional tools to expose alongside the default weather tool set.
    """
    tools = list(WEATHER_TOOLS) + list(extra_tools or [])
    return build_graph(llm=llm, tools=tools, system_prompt=WEATHER_SYSTEM_PROMPT)
