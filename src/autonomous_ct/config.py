"""Environment-driven configuration for the autonomous-ct agent.

Reads from process environment (and optionally a local ``.env`` file if
``python-dotenv`` is installed). Keeping configuration here avoids scattering
``os.environ`` lookups across the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """Best-effort, idempotent load of a local ``.env`` file."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        _dotenv_loaded = True
        return
    load_dotenv()
    _dotenv_loaded = True


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    argo_base_url: str
    argo_api_key: str
    argo_model: str
    argo_host_header: str | None
    argo_timeout_seconds: int = 60
    argo_max_retries: int = 2

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv_once()
        try:
            return cls(
                argo_base_url=os.environ["ARGO_BASE_URL"],
                argo_api_key=os.environ["ARGO_API_KEY"],
                argo_model=os.environ["ARGO_MODEL"],
                argo_host_header=os.environ.get("ARGO_HOST_HEADER"),
                argo_timeout_seconds=_get_int("ARGO_TIMEOUT_SECONDS", 60),
                argo_max_retries=_get_int("ARGO_MAX_RETRIES", 2),
            )
        except KeyError as missing:
            raise RuntimeError(
                f"Missing required environment variable: {missing.args[0]}. "
                "See .env.example for the full list."
            ) from missing


def get_settings() -> Settings:
    """Convenience accessor; constructs fresh ``Settings`` each call."""
    return Settings.from_env()
