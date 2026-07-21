"""Tests for the Chainlit web UI's pure rendering helpers.

The Chainlit handler layer (``@cl.on_message`` etc.) requires a running
Chainlit server and the optional ``web`` extra, so it is not exercised here.
Instead we test the LangGraph-agnostic mapping helpers that decide *what* the
UI renders -- the same separation of concerns the CLI uses for
``_render_intermediate``.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from autonomous_ct.tools.tomo_recon import SLICE_PNG_MARKER
from autonomous_ct.webui import (
    ToolCallView,
    describe_tool_calls,
    describe_tool_result,
    extract_slice_png_paths,
    is_final_reply,
    new_thread_id,
    step_title_for_call,
)


def _tool_call_message(name: str = "echo", args: dict | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args if args is not None else {"text": "ping"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )


# ---------------------------------------------------------------------------
# describe_tool_calls
# ---------------------------------------------------------------------------


def test_describe_tool_calls_extracts_view() -> None:
    views = describe_tool_calls(_tool_call_message("inspect_hdf5_dataset", {"input_file": "/d.h5"}))
    assert len(views) == 1
    view = views[0]
    assert isinstance(view, ToolCallView)
    assert view.name == "inspect_hdf5_dataset"
    assert view.args == {"input_file": "/d.h5"}
    assert view.id == "call_1"


def test_describe_tool_calls_returns_empty_for_final_reply() -> None:
    assert describe_tool_calls(AIMessage(content="all done")) == []


def test_describe_tool_calls_returns_empty_for_human_message() -> None:
    assert describe_tool_calls(HumanMessage(content="hi")) == []


def test_describe_tool_calls_returns_empty_for_tool_message() -> None:
    assert describe_tool_calls(ToolMessage(content="ok", name="echo", tool_call_id="c1")) == []


def test_describe_tool_calls_handles_multiple_calls() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "a", "args": {"x": 1}, "id": "1", "type": "tool_call"},
            {"name": "b", "args": {"y": 2}, "id": "2", "type": "tool_call"},
        ],
    )
    views = describe_tool_calls(msg)
    assert [v.name for v in views] == ["a", "b"]
    assert [v.args for v in views] == [{"x": 1}, {"y": 2}]


# ---------------------------------------------------------------------------
# describe_tool_result
# ---------------------------------------------------------------------------


def test_describe_tool_result_extracts_view() -> None:
    result = describe_tool_result(
        ToolMessage(content="HDF5 summary ...", name="inspect_hdf5_dataset", tool_call_id="c1")
    )
    assert result is not None
    assert result.name == "inspect_hdf5_dataset"
    assert result.content == "HDF5 summary ..."
    assert result.tool_call_id == "c1"


def test_describe_tool_result_none_for_non_tool_message() -> None:
    assert describe_tool_result(AIMessage(content="reply")) is None
    assert describe_tool_result(_tool_call_message()) is None
    assert describe_tool_result(HumanMessage(content="hi")) is None


def test_describe_tool_result_handles_empty_content() -> None:
    result = describe_tool_result(ToolMessage(content="", name="echo", tool_call_id="c1"))
    assert result is not None
    assert result.content == ""


# ---------------------------------------------------------------------------
# is_final_reply
# ---------------------------------------------------------------------------


def test_is_final_reply_true_for_plain_assistant_text() -> None:
    assert is_final_reply(AIMessage(content="The reconstruction is queued."))


def test_is_final_reply_false_for_tool_call_message() -> None:
    assert not is_final_reply(_tool_call_message())


def test_is_final_reply_false_for_empty_assistant_message() -> None:
    assert not is_final_reply(AIMessage(content=""))


def test_is_final_reply_false_for_non_ai_messages() -> None:
    assert not is_final_reply(HumanMessage(content="hello"))
    assert not is_final_reply(ToolMessage(content="ok", name="echo", tool_call_id="c1"))


# ---------------------------------------------------------------------------
# step_title_for_call
# ---------------------------------------------------------------------------


def test_step_title_for_call_formats_args() -> None:
    view = ToolCallView(name="tomocupy_dry_run", args={"rotation_axis": 782.5}, id="c1")
    title = step_title_for_call(view)
    assert title == "tomocupy_dry_run(rotation_axis=782.5)"


def test_step_title_for_call_truncates_long_args() -> None:
    view = ToolCallView(name="t", args={"blob": "x" * 200}, id="c1")
    title = step_title_for_call(view)
    # Long values are clipped with an ellipsis before the closing paren.
    assert "..." in title
    assert title.endswith("...)")
    # The rendered title stays bounded regardless of input size.
    assert len(title) < 120


def test_step_title_for_call_multiple_args() -> None:
    view = ToolCallView(name="t", args={"a": 1, "b": "two"}, id="c1")
    assert step_title_for_call(view) == "t(a=1, b='two')"


# ---------------------------------------------------------------------------
# new_thread_id
# ---------------------------------------------------------------------------


def test_new_thread_id_is_short_alnum_and_unique() -> None:
    ids = {new_thread_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(isinstance(i, str) and i.isalnum() and len(i) == 12 for i in ids)


# ---------------------------------------------------------------------------
# extract_slice_png_paths
# ---------------------------------------------------------------------------


def test_extract_slice_png_paths_finds_marker() -> None:
    content = (
        "SLICE_RENDERED [kind=tiff]\n"
        "Volume: /x shape=(nz=10, ny=20, nx=20) dtype=float32\n"
        f"{SLICE_PNG_MARKER} /tmp/autonomous_ct_slices/slice_xy_00005.png"
    )
    msg = ToolMessage(content=content, name="view_reconstruction_slice", tool_call_id="c1")
    assert extract_slice_png_paths(msg) == ["/tmp/autonomous_ct_slices/slice_xy_00005.png"]


def test_extract_slice_png_paths_multiple() -> None:
    content = f"{SLICE_PNG_MARKER} /a.png\nfiller\n{SLICE_PNG_MARKER} /b.png"
    msg = ToolMessage(content=content, name="view_reconstruction_slice", tool_call_id="c1")
    assert extract_slice_png_paths(msg) == ["/a.png", "/b.png"]


def test_extract_slice_png_paths_none_without_marker() -> None:
    msg = ToolMessage(content="just text, no image", name="inspect", tool_call_id="c1")
    assert extract_slice_png_paths(msg) == []


def test_extract_slice_png_paths_ignores_non_tool_messages() -> None:
    assert extract_slice_png_paths(AIMessage(content=f"{SLICE_PNG_MARKER} /a.png")) == []
    assert extract_slice_png_paths(HumanMessage(content="hi")) == []
