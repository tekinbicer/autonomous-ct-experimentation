"""Chainlit web UI for the autonomous-ct agent.

This module exposes the same computational-imaging agent that powers the CLI
(:mod:`autonomous_ct.cli`) through a browser chat interface built on
`Chainlit <https://chainlit.io>`_.

Design mirrors the CLI: the LangGraph-agnostic *rendering* logic lives in
pure, unit-testable helpers (``describe_tool_call`` /
``describe_tool_result`` / ``classify_stream_message``) that take only
LangChain message objects and return plain data. The thin Chainlit handler
layer (decorated with ``@cl.on_chat_start`` / ``@cl.on_message``) turns that
data into ``cl.Step`` / ``cl.Message`` UI elements. This keeps the import of
``chainlit`` itself out of the test path while still exercising the message
mapping that the UI depends on.

Run it via the top-level launcher (``web_app.py``)::

    uv sync --extra web
    uv run chainlit run web_app.py -w

The agent's intermediate reasoning -- every ``tomocupy_dry_run``,
``inspect_hdf5_dataset``, ``read_hdf5_values`` and ``tomocupy_reconstruct``
call plus its result -- is surfaced as a collapsible step in the chat, the
browser equivalent of the CLI's live ``-> tool:`` / ``<- tool:`` lines.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .agents.imaging import IMAGING_AGENT
from .computation_graph import build_computation_graph
from .tools.tomo_recon import SLICE_PNG_MARKER

# ---------------------------------------------------------------------------
# Session-state keys (stored in cl.user_session)
# ---------------------------------------------------------------------------

SESSION_APP_KEY = "ct_app"
SESSION_CONFIG_KEY = "ct_config"

# Default on-disk SQLite store backing per-thread conversation memory, matching
# the CLI default (autonomous_ct.cli.DEFAULT_STATE_DB).
DEFAULT_STATE_DB = ".autonomous_ct_threads.sqlite"


# ---------------------------------------------------------------------------
# Pure rendering helpers (no Chainlit import -- unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallView:
    """A flattened view of a single tool call for UI rendering."""

    name: str
    args: dict[str, Any]
    id: str | None


@dataclass(frozen=True)
class ToolResultView:
    """A flattened view of a single tool result for UI rendering."""

    name: str
    content: str
    tool_call_id: str | None


def _format_tool_call_args(args: Any) -> str:
    """Compact one-line preview of tool-call arguments.

    Mirrors :func:`autonomous_ct.cli._format_tool_call_args` so the web UI's
    step titles read the same as the CLI's live ``-> tool:`` lines.
    """
    if not isinstance(args, dict):
        return repr(args)
    parts = []
    for key, value in args.items():
        text = repr(value)
        if len(text) > 80:
            text = text[:77] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def describe_tool_calls(message: BaseMessage) -> list[ToolCallView]:
    """Extract tool calls from an ``AIMessage`` as render-ready views.

    Returns an empty list for any message that is not an ``AIMessage`` with
    tool calls (e.g. the final assistant reply, or a plain human turn).
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    if not (isinstance(message, AIMessage) and tool_calls):
        return []
    views: list[ToolCallView] = []
    for call in tool_calls:
        if isinstance(call, dict):
            name = call.get("name") or "?"
            args = call.get("args") or {}
            call_id = call.get("id")
        else:  # pragma: no cover -- defensive; LangChain uses dicts today
            name = getattr(call, "name", "?")
            args = getattr(call, "args", {})
            call_id = getattr(call, "id", None)
        views.append(
            ToolCallView(
                name=str(name),
                args=args if isinstance(args, dict) else {"value": args},
                id=call_id,
            )
        )
    return views


def describe_tool_result(message: BaseMessage) -> ToolResultView | None:
    """Extract a ``ToolMessage`` as a render-ready view, else ``None``."""
    if not isinstance(message, ToolMessage):
        return None
    return ToolResultView(
        name=str(getattr(message, "name", None) or "?"),
        content=str(message.content) if message.content is not None else "",
        tool_call_id=getattr(message, "tool_call_id", None),
    )


def step_title_for_call(view: ToolCallView) -> str:
    """Human-readable title for a tool-call step (matches CLI phrasing)."""
    return f"{view.name}({_format_tool_call_args(view.args)})"


def extract_slice_png_paths(message: BaseMessage) -> list[str]:
    """Return PNG paths emitted by ``view_reconstruction_slice``, if any.

    The slice-viewer tool writes a rendered PNG to a temp file and reports it
    in its (text) result via a ``SLICE_PNG: <path>`` marker line. The web UI
    parses those lines to attach the PNGs as inline images. Non-tool messages
    and tool results without the marker yield an empty list.
    """
    result = describe_tool_result(message)
    if result is None:
        return []
    paths: list[str] = []
    for line in result.content.splitlines():
        stripped = line.strip()
        if stripped.startswith(SLICE_PNG_MARKER):
            candidate = stripped[len(SLICE_PNG_MARKER) :].strip()
            if candidate:
                paths.append(candidate)
    return paths


def is_final_reply(message: BaseMessage) -> bool:
    """True only for a plain assistant reply (content, no tool calls).

    This is the message the UI renders as the visible chat answer; tool-call
    AIMessages and ToolMessages are rendered as steps instead.
    """
    if not isinstance(message, AIMessage):
        return False
    has_tool_calls = bool(getattr(message, "tool_calls", None) or [])
    return (not has_tool_calls) and bool(message.content)


def new_thread_id() -> str:
    """Short, URL-safe thread id (matches CLI convention)."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Chainlit handler layer (thin; imported lazily so tests don't need chainlit)
# ---------------------------------------------------------------------------

# Chainlit is imported lazily inside register_handlers(), NOT at module level.
# Importing chainlit eagerly would call load_dotenv() as a side effect, leaking
# .env values (e.g. TOMOCUPY_BACKEND) into the process and polluting other
# tests that import this module only for the pure helpers above. The launcher
# (web_app.py) calls register_handlers() so `chainlit run` still discovers the
# decorated handlers.


def _collect_stream(
    app: Any, payload: dict[str, Any], config: dict[str, Any]
) -> list[list[BaseMessage]]:
    """Drain the sync LangGraph stream into a list of per-chunk messages.

    Each element is the list of messages produced by one graph node execution.
    Kept synchronous and chainlit-free so it can run off the event loop via
    ``cl.make_async``.
    """
    chunks: list[list[BaseMessage]] = []
    for chunk in app.stream(payload, config=config, stream_mode="updates"):
        for _node, node_out in chunk.items():
            messages = node_out.get("messages", []) or []
            if messages:
                chunks.append(list(messages))
    return chunks


def register_handlers() -> None:  # pragma: no cover -- requires Chainlit runtime
    """Import Chainlit and register the chat handlers.

    Called by the top-level ``web_app.py`` launcher. Deferring the import here
    keeps ``import autonomous_ct.webui`` free of Chainlit's import-time side
    effects (notably ``load_dotenv()``).
    """
    import chainlit as cl

    @cl.on_chat_start
    async def _on_chat_start() -> None:
        from langgraph.checkpoint.sqlite import SqliteSaver

        # Long-lived saver tied to the chat session: opened manually (not as a
        # context manager) because its lifetime spans many awaited turns.
        saver_cm = SqliteSaver.from_conn_string(DEFAULT_STATE_DB)
        saver = saver_cm.__enter__()
        cl.user_session.set("_ct_saver_cm", saver_cm)

        try:
            app = build_computation_graph(agents=[IMAGING_AGENT], checkpointer=saver)
        except RuntimeError as exc:
            await cl.Message(
                content=(
                    "**Configuration error.** The agent could not start:\n\n"
                    f"> {exc}\n\n"
                    "Set the required `ARGO_*` environment variables "
                    "(see `.env.example`) and reload."
                )
            ).send()
            return

        thread_id = new_thread_id()
        cl.user_session.set(SESSION_APP_KEY, app)
        cl.user_session.set(SESSION_CONFIG_KEY, {"configurable": {"thread_id": thread_id}})

        await cl.Message(
            content=(
                "**autonomous-ct** imaging assistant ready "
                f"(thread `{thread_id}`).\n\n"
                "Ask me to inspect an HDF5 scan or plan/run a `tomocupy` "
                "reconstruction. I'll show every tool call and its result as a "
                "collapsible step."
            )
        ).send()

    @cl.on_chat_end
    async def _on_chat_end() -> None:
        saver_cm = cl.user_session.get("_ct_saver_cm")
        if saver_cm is not None:
            with contextlib.suppress(Exception):
                saver_cm.__exit__(None, None, None)

    async def _emit_message(message: BaseMessage) -> str | None:
        for call in describe_tool_calls(message):
            async with cl.Step(name=call.name, type="tool") as step:
                step.input = call.args
                # Matching ToolMessage arrives in a later chunk; show the
                # planned call now so the user sees activity immediately.
                step.output = "(running...)"

        result = describe_tool_result(message)
        if result is not None:
            async with cl.Step(name=result.name, type="tool") as step:
                step.output = result.content

        png_paths = extract_slice_png_paths(message)
        if png_paths:
            images = [
                cl.Image(path=path, name=Path(path).name, display="inline")
                for path in png_paths
            ]
            await cl.Message(content="Reconstruction slice:", elements=images).send()

        if is_final_reply(message):
            return str(message.content)
        return None

    @cl.on_message
    async def _on_message(message: cl.Message) -> None:
        app = cl.user_session.get(SESSION_APP_KEY)
        config = cl.user_session.get(SESSION_CONFIG_KEY)
        if app is None or config is None:
            await cl.Message(content="Session not initialized. Please reload the page.").send()
            return

        final_text = ""
        payload = {"messages": [HumanMessage(content=message.content)]}

        try:
            # Run the sync LangGraph stream off the event loop so long GPU runs
            # don't block Chainlit.
            for chunk in await cl.make_async(_collect_stream)(app, payload, config):
                for emitted in chunk:
                    rendered_final = await _emit_message(emitted)
                    if rendered_final is not None:
                        final_text = rendered_final
        except Exception as exc:  # noqa: BLE001 -- surface failure, keep session alive
            await cl.Message(content=f"**Agent error:** {exc}").send()
            return

        if final_text:
            await cl.Message(content=final_text).send()


__all__ = [
    "DEFAULT_STATE_DB",
    "SESSION_APP_KEY",
    "SESSION_CONFIG_KEY",
    "ToolCallView",
    "ToolResultView",
    "describe_tool_calls",
    "describe_tool_result",
    "extract_slice_png_paths",
    "is_final_reply",
    "new_thread_id",
    "register_handlers",
    "step_title_for_call",
]
