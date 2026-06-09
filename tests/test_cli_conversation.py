"""Tests for the interactive ``--conversation`` mode of the CLI."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from autonomous_ct.cli import (
    CONVERSATION_PROMPT,
    FINALIZE_COMMAND,
    _is_finalize,
    _new_thread_id,
    run_conversation,
)
from autonomous_ct.graph import build_graph

from ._fakes import ScriptedChatModel


@tool
def echo(text: str) -> str:
    """Echo the input back unchanged."""
    return text


def _fixed_now() -> datetime:
    return datetime(2026, 5, 26, 14, 30, 45)


def _build_app(responses: list[AIMessage], checkpointer=None):
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    return build_graph(
        llm=ScriptedChatModel(responses=responses),
        tools=[echo],
        checkpointer=saver,
    )


def _config(thread_id: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def test_is_finalize_matches_case_and_whitespace_variants() -> None:
    assert _is_finalize("finalize")
    assert _is_finalize("FINALIZE")
    assert _is_finalize("  Finalize  \n")
    assert not _is_finalize("finalize now")
    assert not _is_finalize("please finalize")
    assert not _is_finalize("")


def test_new_thread_id_is_short_and_unique() -> None:
    ids = {_new_thread_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(isinstance(i, str) and i.isalnum() and len(i) == 12 for i in ids)


def test_conversation_persists_transcript_on_finalize(tmp_path: Path) -> None:
    app = _build_app(
        [
            AIMessage(content="Hello Alice."),
            AIMessage(content="It's sunny."),
        ]
    )
    stdin = io.StringIO("Hi, I'm Alice.\nWhat's the weather?\nfinalize\n")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    expected = tmp_path / "conversation_20260526-143045.log"
    assert log_path == expected
    assert expected.is_file()

    contents = expected.read_text(encoding="utf-8")
    assert "[HUMAN]" in contents
    assert "Hi, I'm Alice." in contents
    assert "What's the weather?" in contents
    assert "[AI]" in contents
    assert "Hello Alice." in contents
    assert "It's sunny." in contents

    out = stdout.getvalue()
    assert "Hello Alice." in out
    assert "It's sunny." in out
    assert str(expected) in out
    assert out.count(CONVERSATION_PROMPT) == 3


def test_conversation_full_history_visible_to_model_via_checkpointer() -> None:
    """The model sees the accumulated thread on every turn (server-side via the checkpointer)."""
    captured: list[int] = []

    class RecordingModel(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.append(len(messages))
            return super()._generate(messages, stop, run_manager, **kwargs)

    fake = RecordingModel(
        responses=[
            AIMessage(content="reply 1"),
            AIMessage(content="reply 2"),
            AIMessage(content="reply 3"),
        ]
    )
    app = build_graph(llm=fake, tools=[echo], checkpointer=InMemorySaver())

    stdin = io.StringIO("turn one\nturn two\nturn three\nfinalize\n")
    stdout = io.StringIO()
    run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=Path("/tmp"),
    )

    assert captured == [2, 4, 6]


def test_conversation_skips_empty_lines(tmp_path: Path) -> None:
    app = _build_app([AIMessage(content="ack")])
    stdin = io.StringIO("\n   \n\nhello\nfinalize\n")
    stdout = io.StringIO()

    run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    log = (tmp_path / "conversation_20260526-143045.log").read_text(encoding="utf-8")
    assert "hello" in log
    assert log.count("[HUMAN]") == 1


def test_conversation_eof_writes_log_when_state_exists(tmp_path: Path) -> None:
    """EOF after at least one successful turn still persists the transcript."""
    app = _build_app([AIMessage(content="ack")])
    stdin = io.StringIO("hello\n")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    assert log_path is not None
    assert log_path.is_file()
    assert "[end of input]" in stdout.getvalue()


def test_conversation_eof_before_any_input_writes_no_log(tmp_path: Path) -> None:
    app = _build_app([])
    stdin = io.StringIO("")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    assert log_path is None
    assert list(tmp_path.iterdir()) == []


def test_conversation_finalize_immediately_writes_no_log(tmp_path: Path) -> None:
    app = _build_app([])
    stdin = io.StringIO(f"{FINALIZE_COMMAND}\n")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    assert log_path is None
    assert list(tmp_path.iterdir()) == []
    assert "no conversation to save" in stdout.getvalue()


def test_conversation_agent_error_keeps_session_alive(tmp_path: Path) -> None:
    """An exception during one turn must not kill the loop or corrupt thread state."""

    class BoomThenWorkApp:
        def __init__(self) -> None:
            self.calls = 0
            self._state_messages: list = []

        def stream(self, payload, config=None, stream_mode=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated upstream failure")
            new = list(payload["messages"])
            self._state_messages.extend(new)
            reply = AIMessage(content="recovered")
            self._state_messages.append(reply)
            yield {"agent": {"messages": [reply]}}

        def get_state(self, config):
            from types import SimpleNamespace

            return SimpleNamespace(values={"messages": list(self._state_messages)})

    app = BoomThenWorkApp()
    stdin = io.StringIO("first\nsecond\nfinalize\n")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    assert log_path is not None
    out = stdout.getvalue()
    assert "[agent error]" in out
    assert "recovered" in out

    log = log_path.read_text(encoding="utf-8")
    assert "second" in log
    assert "recovered" in log
    assert "first" not in log


def test_conversation_log_records_tool_calls(tmp_path: Path) -> None:
    """Tool-call turns must appear in the saved transcript for auditability."""
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "echo", "args": {"text": "ping"}, "id": "call_1", "type": "tool_call"}
        ],
    )
    final = AIMessage(content="The tool said: ping")
    app = _build_app([tool_call, final])

    stdin = io.StringIO("please echo ping\nfinalize\n")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
        config=_config(),
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    assert log_path is not None
    log = log_path.read_text(encoding="utf-8")
    assert "tool_call: echo" in log
    assert "[TOOL:echo]" in log
    assert "ping" in log
    assert "The tool said: ping" in log


def test_conversation_different_thread_ids_are_isolated(tmp_path: Path) -> None:
    """Two thread_ids on the same checkpointer must not see each other's messages."""
    saver = InMemorySaver()
    app = build_graph(
        llm=ScriptedChatModel(
            responses=[
                AIMessage(content="hello A"),
                AIMessage(content="hello B"),
            ]
        ),
        tools=[echo],
        checkpointer=saver,
    )

    run_conversation(
        app,
        config=_config("thread-A"),
        stdin=io.StringIO("question for A\nfinalize\n"),
        stdout=io.StringIO(),
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )
    run_conversation(
        app,
        config=_config("thread-B"),
        stdin=io.StringIO("question for B\nfinalize\n"),
        stdout=io.StringIO(),
        now_factory=lambda: datetime(2026, 5, 26, 14, 30, 46),
        log_directory=tmp_path,
    )

    state_a = app.get_state(_config("thread-A"))
    state_b = app.get_state(_config("thread-B"))
    texts_a = [str(m.content) for m in state_a.values["messages"]]
    texts_b = [str(m.content) for m in state_b.values["messages"]]

    assert "question for A" in texts_a
    assert "hello A" in texts_a
    assert "question for B" not in texts_a
    assert "hello B" not in texts_a

    assert "question for B" in texts_b
    assert "hello B" in texts_b
    assert "question for A" not in texts_b
    assert "hello A" not in texts_b


def test_conversation_resume_same_thread_accumulates_state(tmp_path: Path) -> None:
    """Two run_conversation calls with the same thread_id share history."""
    saver = InMemorySaver()
    app = build_graph(
        llm=ScriptedChatModel(
            responses=[
                AIMessage(content="first answer"),
                AIMessage(content="second answer"),
            ]
        ),
        tools=[echo],
        checkpointer=saver,
    )
    config = _config("resumable")

    run_conversation(
        app,
        config=config,
        stdin=io.StringIO("opening turn\nfinalize\n"),
        stdout=io.StringIO(),
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )
    log_path = run_conversation(
        app,
        config=config,
        stdin=io.StringIO("follow-up turn\nfinalize\n"),
        stdout=io.StringIO(),
        now_factory=lambda: datetime(2026, 5, 26, 14, 30, 46),
        log_directory=tmp_path,
    )

    assert log_path is not None
    log = log_path.read_text(encoding="utf-8")
    assert "opening turn" in log
    assert "first answer" in log
    assert "follow-up turn" in log
    assert "second answer" in log


def test_conversation_sqlite_persists_across_app_instances(tmp_path: Path) -> None:
    """Real SqliteSaver round-trip: state survives across separate saver/app lifecycles."""
    db_path = tmp_path / "threads.sqlite"
    config = _config("sqlite-resumable")

    with SqliteSaver.from_conn_string(str(db_path)) as saver1:
        app1 = build_graph(
            llm=ScriptedChatModel(responses=[AIMessage(content="from session one")]),
            tools=[echo],
            checkpointer=saver1,
        )
        run_conversation(
            app1,
            config=config,
            stdin=io.StringIO("hello from session one\nfinalize\n"),
            stdout=io.StringIO(),
            now_factory=_fixed_now,
            log_directory=tmp_path,
        )

    with SqliteSaver.from_conn_string(str(db_path)) as saver2:
        app2 = build_graph(
            llm=ScriptedChatModel(responses=[AIMessage(content="from session two")]),
            tools=[echo],
            checkpointer=saver2,
        )
        log_path = run_conversation(
            app2,
            config=config,
            stdin=io.StringIO("hello from session two\nfinalize\n"),
            stdout=io.StringIO(),
            now_factory=lambda: datetime(2026, 5, 26, 14, 30, 46),
            log_directory=tmp_path,
        )

    assert log_path is not None
    log = log_path.read_text(encoding="utf-8")
    assert "hello from session one" in log
    assert "from session one" in log
    assert "hello from session two" in log
    assert "from session two" in log


def test_conversation_announces_thread_id(tmp_path: Path) -> None:
    app = _build_app([AIMessage(content="hi")])
    stdout = io.StringIO()
    run_conversation(
        app,
        config=_config("visible-thread-id"),
        stdin=io.StringIO("hello\nfinalize\n"),
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )
    out = stdout.getvalue()
    assert "visible-thread-id" in out


def test_conversation_streams_tool_calls_and_results_by_default(tmp_path: Path) -> None:
    """Default mode shows intermediate tool calls and tool results live."""
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "echo", "args": {"text": "ping"}, "id": "c1", "type": "tool_call"}
        ],
    )
    final = AIMessage(content="The tool said: ping")
    app = _build_app([tool_call, final])

    stdout = io.StringIO()
    run_conversation(
        app,
        config=_config(),
        stdin=io.StringIO("please echo ping\nfinalize\n"),
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    out = stdout.getvalue()
    assert "-> tool: echo(" in out
    assert "text='ping'" in out
    assert "<- echo: ping" in out
    assert "The tool said: ping" in out

    # Ordering: tool call must appear before tool result, which must appear
    # before the final reply.
    pos_call = out.index("-> tool: echo")
    pos_result = out.index("<- echo: ping")
    pos_final = out.index("The tool said: ping")
    assert pos_call < pos_result < pos_final


def test_conversation_quiet_mode_hides_intermediates(tmp_path: Path) -> None:
    """With quiet=True, only the final assistant reply is printed."""
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "echo", "args": {"text": "ping"}, "id": "c1", "type": "tool_call"}
        ],
    )
    final = AIMessage(content="The tool said: ping")
    app = _build_app([tool_call, final])

    stdout = io.StringIO()
    run_conversation(
        app,
        config=_config(),
        stdin=io.StringIO("please echo ping\nfinalize\n"),
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
        quiet=True,
    )

    out = stdout.getvalue()
    assert "-> tool:" not in out
    assert "<- echo:" not in out
    assert "The tool said: ping" in out


def test_conversation_quiet_mode_preserves_finalize_export(tmp_path: Path) -> None:
    """--quiet hides intermediates from stdout but the transcript still includes them."""
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "echo", "args": {"text": "ping"}, "id": "c1", "type": "tool_call"}
        ],
    )
    final = AIMessage(content="The tool said: ping")
    app = _build_app([tool_call, final])

    log_path = run_conversation(
        app,
        config=_config(),
        stdin=io.StringIO("please echo ping\nfinalize\n"),
        stdout=io.StringIO(),
        now_factory=_fixed_now,
        log_directory=tmp_path,
        quiet=True,
    )

    assert log_path is not None
    log = log_path.read_text(encoding="utf-8")
    assert "tool_call: echo" in log
    assert "[TOOL:echo]" in log


def test_render_intermediate_returns_none_for_final_reply() -> None:
    """Plain AIMessage with content and no tool_calls is the surrounding loop's job."""
    from autonomous_ct.cli import _render_intermediate

    assert _render_intermediate(AIMessage(content="just a reply")) is None
    assert _render_intermediate(AIMessage(content="")) is None


def test_render_intermediate_formats_tool_call_and_result() -> None:
    from langchain_core.messages import ToolMessage

    from autonomous_ct.cli import _render_intermediate

    call = AIMessage(
        content="",
        tool_calls=[{"name": "echo", "args": {"text": "ping"}, "id": "c1", "type": "tool_call"}],
    )
    rendered = _render_intermediate(call)
    assert rendered is not None
    assert "-> tool: echo(" in rendered
    assert "text='ping'" in rendered

    result = ToolMessage(content="ping", name="echo", tool_call_id="c1")
    rendered_result = _render_intermediate(result)
    assert rendered_result == "  <- echo: ping"
