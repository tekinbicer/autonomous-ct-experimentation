"""Command-line entry point for the autonomous-ct agent."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .agents.imaging import IMAGING_AGENT
from .agents.weather import build_weather_graph
from .computation_graph import build_computation_graph

logger = logging.getLogger(__name__)

# Single word (case-insensitive, whitespace-stripped) that terminates a
# conversation session and triggers transcript persistence.
FINALIZE_COMMAND = "finalize"

# User-facing prompt prefix in --conversation mode. Kept short so it's easy
# to spot the input boundary in long sessions.
CONVERSATION_PROMPT = "you> "


def _imaging_factory() -> Any:
    return build_computation_graph(agents=[IMAGING_AGENT])


def _weather_factory() -> Any:
    return build_weather_graph()


AGENT_BUILDERS: dict[str, Callable[[], Any]] = {
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
            f"conversation. Type '{FINALIZE_COMMAND}' on its own line to save "
            "the transcript and exit."
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


def run_conversation(
    app: Any,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    now_factory: Callable[[], datetime] = datetime.now,
    log_directory: Path | None = None,
) -> Path | None:
    """Drive the agent in interactive multi-turn mode.

    Reads user prompts line by line from ``stdin``, invokes the compiled
    LangGraph ``app`` with the accumulated message history each turn, and
    prints the agent's reply to ``stdout``. Returns the path of the written
    transcript on a clean ``finalize``, or ``None`` if the session ended
    without persisting (EOF before any user input, etc.).

    The function is intentionally injectable (``stdin``/``stdout``/
    ``now_factory``/``log_directory``) so it can be tested without a real
    terminal or a real LLM.
    """
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout

    history: list[BaseMessage] = []

    def _emit(text: str = "") -> None:
        print(text, file=out_stream, flush=True)

    _emit(
        f"autonomous-ct conversation mode. Type '{FINALIZE_COMMAND}' to save "
        "the transcript and exit (Ctrl-D / Ctrl-C also exits)."
    )

    finalized_cleanly = False
    while True:
        try:
            print(CONVERSATION_PROMPT, end="", file=out_stream, flush=True)
            raw = in_stream.readline()
        except KeyboardInterrupt:
            _emit("\n[interrupted]")
            break

        if raw == "":
            # EOF (Ctrl-D on a TTY, or end of a piped stream).
            _emit("\n[end of input]")
            break

        line = raw.rstrip("\n")
        if not line.strip():
            # Empty turn: don't burn an LLM call, just re-prompt.
            continue

        if _is_finalize(line):
            finalized_cleanly = True
            break

        history.append(HumanMessage(content=line))
        try:
            result = app.invoke({"messages": history})
        except KeyboardInterrupt:
            _emit("\n[interrupted during agent turn]")
            break
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the user, keep session alive
            logger.exception("Agent invocation failed")
            _emit(f"[agent error] {exc}")
            # Drop the user turn that caused the failure so the next retry is clean.
            history.pop()
            continue

        new_messages: list[BaseMessage] = list(result["messages"])
        # ``add_messages`` returns the full updated history; replace ours wholesale
        # so tool calls / tool messages produced inside the graph are preserved.
        history = new_messages

        _emit(str(history[-1].content))

    if not history:
        if finalized_cleanly:
            _emit("[finalize] no conversation to save.")
        return None

    log_path = _conversation_log_path(now_factory(), log_directory)
    log_path.write_text(_render_transcript(history), encoding="utf-8")
    _emit(f"[finalize] transcript saved to {log_path}")
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

    builder = AGENT_BUILDERS[args.agent]
    try:
        app = builder()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    if args.conversation:
        run_conversation(app)
        return 0

    prompt = _resolve_prompt(parser, args)
    if not prompt:
        logger.error("No prompt provided.")
        return 2

    result = app.invoke({"messages": [HumanMessage(content=prompt)]})
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Public re-exports used by tests.
# ---------------------------------------------------------------------------

__all__ = [
    "AGENT_BUILDERS",
    "CONVERSATION_PROMPT",
    "FINALIZE_COMMAND",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "main",
    "run_conversation",
]
