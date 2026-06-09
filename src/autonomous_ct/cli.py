"""Command-line entry point for the autonomous-ct agent."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .agents.imaging import IMAGING_AGENT
from .agents.weather import build_weather_graph
from .computation_graph import build_computation_graph

logger = logging.getLogger(__name__)

FINALIZE_COMMAND = "finalize"
CONVERSATION_PROMPT = "you> "
DEFAULT_STATE_DB = ".autonomous_ct_threads.sqlite"


def _imaging_factory(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    return build_computation_graph(agents=[IMAGING_AGENT], checkpointer=checkpointer)


def _weather_factory(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    return build_weather_graph(checkpointer=checkpointer)


AgentFactory = Callable[[BaseCheckpointSaver | None], Any]

AGENT_BUILDERS: dict[str, AgentFactory] = {
    "imaging": _imaging_factory,
    "weather": _weather_factory,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autonomous-ct",
        description="Run a single prompt or an interactive session through an autonomous-ct agent.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help=(
            "Prompt to send. If omitted, reads from stdin (only when stdin is piped). "
            "Ignored when --conversation is set."
        ),
    )
    parser.add_argument(
        "-a",
        "--agent",
        choices=sorted(AGENT_BUILDERS.keys()),
        default="imaging",
        help="Which agent to run (default: imaging).",
    )
    parser.add_argument(
        "-c",
        "--conversation",
        action="store_true",
        help=(
            "Run in interactive multi-turn mode. The agent remembers the full "
            f"conversation via a LangGraph checkpointer. Type '{FINALIZE_COMMAND}' "
            "to export the transcript and exit."
        ),
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help=(
            "Conversation thread id. Reuse an existing id to resume a prior "
            "session; omit to start a new thread (a fresh id is generated and "
            "printed). Used by --conversation."
        ),
    )
    parser.add_argument(
        "--state-db",
        default=DEFAULT_STATE_DB,
        help=(
            f"SQLite file backing conversation state (default: {DEFAULT_STATE_DB}). "
            "Used by --conversation."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help=(
            "Suppress live display of intermediate tool calls and tool results; "
            "only the final assistant reply is printed. Useful for piping."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _resolve_prompt(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str | None:
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    parser.error("prompt is required when stdin is a TTY")
    return None  # pragma: no cover -- parser.error raises SystemExit


# ---------------------------------------------------------------------------
# Conversation mode
# ---------------------------------------------------------------------------


def _format_message_for_log(message: BaseMessage) -> str:
    """Render a single LangChain message for the persisted transcript.

    Keeps the format human-readable rather than JSON: this file is meant to be
    skimmed in a terminal or pasted into a bug report. Tool calls and tool
    results are included so the reader can audit what the agent actually did.
    """
    role = type(message).__name__.replace("Message", "").upper() or "MESSAGE"
    body = str(message.content) if message.content is not None else ""

    extras: list[str] = []
    # AIMessage may carry tool_calls even when content is empty (pure tool-call turn).
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            extras.append(f"  tool_call: {name}({args})")
    # ToolMessage carries the tool's name on .name.
    tool_name = getattr(message, "name", None)
    header = f"[{role}]" if not tool_name else f"[{role}:{tool_name}]"

    parts = [header]
    if body:
        parts.append(body)
    parts.extend(extras)
    return "\n".join(parts)


def _render_transcript(messages: list[BaseMessage]) -> str:
    """Render the full message history as a divider-separated transcript."""
    blocks = [_format_message_for_log(m) for m in messages]
    return ("\n" + ("-" * 72) + "\n").join(blocks) + "\n"


def _conversation_log_path(now: datetime, directory: Path | None = None) -> Path:
    """Compute the conversation log path for a given timestamp."""
    stamp = now.strftime("%Y%m%d-%H%M%S")
    base = directory if directory is not None else Path.cwd()
    return base / f"conversation_{stamp}.log"


def _is_finalize(line: str) -> bool:
    """The literal single word ``finalize`` (case-insensitive) on its own line ends the session."""
    return line.strip().lower() == FINALIZE_COMMAND


def _new_thread_id() -> str:
    """Short, URL-safe thread identifier."""
    return uuid.uuid4().hex[:12]


def _read_checkpointed_messages(app: Any, config: dict[str, Any]) -> list[BaseMessage]:
    """Return the current message list from the checkpointed thread, or [] if absent."""
    state = app.get_state(config)
    values = getattr(state, "values", None) or {}
    return list(values.get("messages", []))


def _format_tool_call_args(args: Any) -> str:
    """Compact one-line preview of tool-call arguments for live display."""
    if not isinstance(args, dict):
        return repr(args)
    parts = []
    for k, v in args.items():
        text = repr(v)
        if len(text) > 80:
            text = text[:77] + "..."
        parts.append(f"{k}={text}")
    return ", ".join(parts)


def _render_intermediate(message: BaseMessage) -> str | None:
    """Format an intermediate streamed message for live REPL display.

    Returns ``None`` for messages that the caller already prints as the
    final reply (a plain AIMessage with non-empty content and no tool
    calls); those are emitted by the surrounding loop, not here.
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    if isinstance(message, AIMessage) and tool_calls:
        lines = []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            lines.append(f"  -> tool: {name}({_format_tool_call_args(args)})")
        return "\n".join(lines)
    if isinstance(message, ToolMessage):
        name = getattr(message, "name", None) or "?"
        body = str(message.content) if message.content is not None else ""
        first_line, _, rest = body.partition("\n")
        if rest:
            indented_rest = rest.replace("\n", "\n      ")
            return f"  <- {name}: {first_line}\n      {indented_rest}"
        return f"  <- {name}: {first_line}"
    return None


def _stream_turn(
    app: Any,
    payload: dict[str, Any],
    config: dict[str, Any] | None,
    on_event: Callable[[BaseMessage], None],
) -> str:
    """Stream one agent turn, invoke ``on_event`` per new message, return final reply text.

    Uses ``stream_mode="updates"`` which yields one ``{node_name: {"messages": [...]}}``
    chunk per graph node execution. The final assistant reply is the last
    AIMessage with content and no tool calls.
    """
    final_text = ""
    for chunk in app.stream(payload, config=config, stream_mode="updates"):
        for _node, payload_out in chunk.items():
            for message in payload_out.get("messages", []) or []:
                on_event(message)
                if (
                    isinstance(message, AIMessage)
                    and not (getattr(message, "tool_calls", None) or [])
                    and message.content
                ):
                    final_text = str(message.content)
    return final_text


def run_conversation(
    app: Any,
    *,
    config: dict[str, Any],
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    now_factory: Callable[[], datetime] = datetime.now,
    log_directory: Path | None = None,
    quiet: bool = False,
) -> Path | None:
    """Drive the agent in interactive multi-turn mode against a checkpointed thread.

    The agent's conversational memory is owned by LangGraph: each turn invokes
    ``app`` with only the new ``HumanMessage`` plus the supplied ``config``
    (which carries the ``thread_id``). The checkpointer accumulates history
    server-side via the ``add_messages`` reducer, so failed invocations are
    not persisted and no manual rollback is needed.

    On clean exit (``finalize``, EOF, or Ctrl-C) the thread's full message
    list is read back via ``app.get_state(config)`` and exported as a
    ``conversation_{YYYYMMDD-HHMMSS}.log`` text file. The exported file is a
    convenience audit artifact; the canonical record is the checkpointer
    store. Returns the export path, or ``None`` if the thread had no
    messages to save.

    ``stdin``/``stdout``/``now_factory``/``log_directory`` are injectable so
    the loop is unit-testable without a terminal or a real LLM.
    """
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout

    def _emit(text: str = "") -> None:
        print(text, file=out_stream, flush=True)

    def _on_event(message: BaseMessage) -> None:
        if quiet:
            return
        rendered = _render_intermediate(message)
        if rendered is not None:
            _emit(rendered)

    thread_id = config.get("configurable", {}).get("thread_id", "<unset>")
    _emit(
        f"autonomous-ct conversation mode [thread_id={thread_id}]. "
        f"Type '{FINALIZE_COMMAND}' to export the transcript and exit "
        "(Ctrl-D / Ctrl-C also exit; state is persisted)."
    )

    saw_successful_turn = _read_checkpointed_messages(app, config) != []
    while True:
        try:
            print(CONVERSATION_PROMPT, end="", file=out_stream, flush=True)
            raw = in_stream.readline()
        except KeyboardInterrupt:
            _emit("\n[interrupted]")
            break

        if raw == "":
            _emit("\n[end of input]")
            break

        line = raw.rstrip("\n")
        if not line.strip():
            continue

        if _is_finalize(line):
            break

        try:
            reply = _stream_turn(
                app,
                {"messages": [HumanMessage(content=line)]},
                config,
                _on_event,
            )
        except KeyboardInterrupt:
            _emit("\n[interrupted during agent turn]")
            break
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the user, keep session alive
            logger.exception("Agent invocation failed")
            _emit(f"[agent error] {exc}")
            # Failed invocations are not checkpointed; no manual rollback needed.
            continue

        saw_successful_turn = True
        if reply:
            _emit(reply)

    if not saw_successful_turn:
        _emit("[finalize] no conversation to save.")
        return None

    persisted = _read_checkpointed_messages(app, config)
    if not persisted:
        _emit("[finalize] no conversation to save.")
        return None

    log_path = _conversation_log_path(now_factory(), log_directory)
    log_path.write_text(_render_transcript(persisted), encoding="utf-8")
    _emit(f"[finalize] transcript exported to {log_path} (thread_id={thread_id})")
    return log_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.conversation and args.prompt:
        parser.error("--conversation does not accept a positional prompt; omit it.")

    factory = AGENT_BUILDERS[args.agent]

    if args.conversation:
        thread_id = args.thread_id or _new_thread_id()
        config = {"configurable": {"thread_id": thread_id}}
        # SqliteSaver.from_conn_string is a context manager; scoping it here
        # keeps the connection's lifetime bounded by main() and avoids cross-
        # module connection ownership.
        with SqliteSaver.from_conn_string(args.state_db) as saver:
            try:
                app = factory(saver)
            except RuntimeError as exc:
                logger.error("%s", exc)
                return 2
            run_conversation(app, config=config, quiet=args.quiet)
        return 0

    try:
        app = factory(None)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    prompt = _resolve_prompt(parser, args)
    if not prompt:
        logger.error("No prompt provided.")
        return 2

    def _on_event(message: BaseMessage) -> None:
        if args.quiet:
            return
        rendered = _render_intermediate(message)
        if rendered is not None:
            print(rendered, flush=True)

    reply = _stream_turn(
        app,
        {"messages": [HumanMessage(content=prompt)]},
        None,
        _on_event,
    )
    if reply:
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGENT_BUILDERS",
    "CONVERSATION_PROMPT",
    "DEFAULT_STATE_DB",
    "FINALIZE_COMMAND",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "main",
    "run_conversation",
]
