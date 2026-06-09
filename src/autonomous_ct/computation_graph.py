"""Multi-agent computation graph.

This module composes one or more :class:`~autonomous_ct.agents.base.Agent`
instances into a single LangGraph application. Today only the
single-agent case (N=1) is implemented; the function signature accepts a
sequence so multi-agent routing can be added without breaking callers.

For N=1 the resulting graph is identical to
``graph.build_graph(tools=agent.tools, system_prompt=agent.system_prompt)``.

When N>1 routing is implemented (supervisor, sequential pipeline, or
handoff-tool style), it will live here so the per-agent modules under
:mod:`autonomous_ct.agents` remain pure data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from .agents.base import Agent
from .graph import build_graph


def build_computation_graph(
    agents: Sequence[Agent],
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile a computation graph composed of one or more agents.

    Parameters
    ----------
    agents:
        Non-empty sequence of agents to include in the graph. With a single
        agent, this is the agent's tool-using loop. With multiple agents,
        an inter-agent routing layer will be added (not yet implemented).
    llm:
        Optional shared chat model. Defaults to the project-configured Argo LLM.
    checkpointer:
        Optional LangGraph checkpointer. When provided, the compiled app
        persists per-thread message state and supports resume across
        invocations. See :func:`autonomous_ct.graph.build_graph` for the
        ``thread_id`` config convention.
    """
    if not agents:
        raise ValueError("build_computation_graph requires at least one agent")

    if len(agents) == 1:
        agent = agents[0]
        return build_graph(
            llm=llm,
            tools=list(agent.tools),
            system_prompt=agent.system_prompt,
            checkpointer=checkpointer,
        )

    raise NotImplementedError(
        f"Multi-agent routing for {len(agents)} agents is not implemented yet. "
        "Decide on a routing pattern (supervisor / sequential pipeline / "
        "handoff tools) and wire it here."
    )
