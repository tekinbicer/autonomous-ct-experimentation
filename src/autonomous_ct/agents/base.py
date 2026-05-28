"""Domain-agnostic ``Agent`` definition shared by every specialized agent.

An :class:`Agent` is pure data: a name, the system prompt that scopes its
behavior, and the tools it is allowed to call. The actual LangGraph wiring
(LLM binding, ToolNode, routing) is performed elsewhere
(:mod:`autonomous_ct.computation_graph` for multi-agent assemblies,
:mod:`autonomous_ct.graph` for the single-agent base compiler).

Keeping this layer free of LangGraph imports lets agents be composed,
inspected, and tested without spinning up a graph.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str
    tools: Sequence[Callable[..., Any]] = field(default_factory=tuple)
