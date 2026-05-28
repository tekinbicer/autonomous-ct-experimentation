"""Smoke tests for ``build_computation_graph``.

Today the computation graph contains a single agent (imaging). The tests
also pin the scaffolding contract for the multi-agent future: empty input
is rejected, and N>1 raises NotImplementedError until routing is wired.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from autonomous_ct.agents.base import Agent
from autonomous_ct.agents.imaging import IMAGING_AGENT
from autonomous_ct.computation_graph import build_computation_graph

from ._fakes import ScriptedChatModel


def test_imaging_agent_dry_runs_then_summarizes(tmp_path: Path) -> None:
    dataset = tmp_path / "scan.h5"
    dataset.write_bytes(b"\x89HDF\r\n\x1a\n")
    out_dir = tmp_path / "data_rec"
    out_dir.mkdir()

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "tomocupy_dry_run",
                "args": {
                    "input_file": str(dataset),
                    "output_dir": str(out_dir),
                    "output_prefix": "scan_rec",
                    "reconstruction_type": "full",
                    "rotation_axis": 782.5,
                    "nsino_per_chunk": 4,
                },
                "id": "call_dry_1",
                "type": "tool_call",
            }
        ],
    )
    final = AIMessage(content="Here's the planned command; confirm to run.")
    fake = ScriptedChatModel(responses=[tool_call, final])

    app = build_computation_graph(agents=[IMAGING_AGENT], llm=fake)
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"Plan a full recon of {dataset} into {out_dir}/scan_rec "
                        "with rotation axis 782.5 and nsino-per-chunk 4."
                    )
                )
            ]
        }
    )

    contents = [str(m.content) for m in result["messages"]]
    assert any("Here's the planned command" in c for c in contents)
    assert any("docker run --rm --gpus all" in c for c in contents)
    assert any("/data_rec/scan_rec" in c for c in contents)


def test_empty_agent_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one agent"):
        build_computation_graph(agents=[])


def test_multi_agent_not_yet_implemented() -> None:
    a = Agent(name="a", system_prompt="A", tools=())
    b = Agent(name="b", system_prompt="B", tools=())
    with pytest.raises(NotImplementedError, match="Multi-agent routing"):
        build_computation_graph(agents=[a, b])
