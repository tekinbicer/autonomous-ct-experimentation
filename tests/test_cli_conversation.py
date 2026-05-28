"""Tests for the interactive ``--conversation`` mode of the CLI."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from autonomous_ct.cli import (
    CONVERSATION_PROMPT,
    FINALIZE_COMMAND,
    _is_finalize,
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


def _build_app(responses: list[AIMessage]):
    return build_graph(llm=ScriptedChatModel(responses=responses), tools=[echo])


def test_is_finalize_matches_case_and_whitespace_variants() -> None:
    assert _is_finalize("finalize")
    assert _is_finalize("FINALIZE")
    assert _is_finalize("  Finalize  \n")
    assert not _is_finalize("finalize now")
    assert not _is_finalize("please finalize")
    assert not _is_finalize("")


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


def test_conversation_preserves_history_across_turns(tmp_path: Path) -> None:
    """The agent should receive the full prior message list on each turn."""
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
    app = build_graph(llm=fake, tools=[echo])

    stdin = io.StringIO("turn one\nturn two\nturn three\nfinalize\n")
    stdout = io.StringIO()
    run_conversation(
        app,
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    assert captured == [2, 4, 6]


def test_conversation_skips_empty_lines(tmp_path: Path) -> None:
    app = _build_app([AIMessage(content="ack")])
    stdin = io.StringIO("\n   \n\nhello\nfinalize\n")
    stdout = io.StringIO()

    run_conversation(
        app,
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    log = (tmp_path / "conversation_20260526-143045.log").read_text(encoding="utf-8")
    assert "hello" in log
    assert log.count("[HUMAN]") == 1


def test_conversation_eof_exits_without_finalize_writes_log(tmp_path: Path) -> None:
    """EOF after at least one turn still persists the transcript."""
    app = _build_app([AIMessage(content="ack")])
    stdin = io.StringIO("hello\n")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
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
        stdin=stdin,
        stdout=stdout,
        now_factory=_fixed_now,
        log_directory=tmp_path,
    )

    assert log_path is None
    assert list(tmp_path.iterdir()) == []
    assert "no conversation to save" in stdout.getvalue()


def test_conversation_agent_error_keeps_session_alive(tmp_path: Path) -> None:
    """An exception during one turn should not kill the loop; the user can retry."""

    class BoomThenWorkApp:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, state):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated upstream failure")
            return {
                "messages": list(state["messages"]) + [AIMessage(content="recovered")],
            }

    app = BoomThenWorkApp()
    stdin = io.StringIO("first\nsecond\nfinalize\n")
    stdout = io.StringIO()

    log_path = run_conversation(
        app,
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
