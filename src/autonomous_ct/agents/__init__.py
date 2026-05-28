"""Domain-specific agent definitions (data) and the agent base type."""

from .base import Agent
from .imaging import IMAGING_AGENT, IMAGING_SYSTEM_PROMPT
from .weather import WEATHER_SYSTEM_PROMPT, build_weather_graph

__all__ = [
    "Agent",
    "IMAGING_AGENT",
    "IMAGING_SYSTEM_PROMPT",
    "WEATHER_SYSTEM_PROMPT",
    "build_weather_graph",
]
