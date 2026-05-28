"""Shared test doubles."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Returns a queued sequence of ``AIMessage`` objects, ignoring inputs.

    Use this in graph tests instead of a live LLM so the test deterministically
    drives the agent down the desired tool path.
    """

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise AssertionError("No scripted response left for the fake model.")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        # Tools are irrelevant to scripted responses; ignore.
        return self
