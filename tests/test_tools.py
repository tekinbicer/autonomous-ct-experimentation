from autonomous_ct.tools.weather import get_weather


def test_get_weather_tokyo_special_case() -> None:
    assert "Tokyo" in get_weather("Tokyo")
    assert "75" in get_weather("tokyo")


def test_get_weather_default() -> None:
    assert get_weather("Paris") == "It's raining in Paris."
