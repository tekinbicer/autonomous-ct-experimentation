"""Placeholder weather tool. Replace with real CT-domain tools as the project grows."""

from __future__ import annotations


def get_weather(location: str) -> str:
    """Use this tool to get the current weather for a specific location."""
    if "tokyo" in location.lower():
        return "It's sunny and 75 degrees in Tokyo!"
    return f"It's raining in {location}."
