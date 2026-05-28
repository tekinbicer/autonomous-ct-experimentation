"""LLM client factory for the Argo-hosted OpenAI-compatible endpoint."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import Settings, get_settings


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Construct a ``ChatOpenAI`` client pointed at the Argo endpoint."""
    settings = settings or get_settings()

    default_headers: dict[str, str] = {}
    if settings.argo_host_header:
        default_headers["Host"] = settings.argo_host_header

    return ChatOpenAI(
        base_url=settings.argo_base_url,
        api_key=settings.argo_api_key,
        model=settings.argo_model,
        default_headers=default_headers or None,
        timeout=settings.argo_timeout_seconds,
        max_retries=settings.argo_max_retries,
    )
