"""Tests for NwsWeather service."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from mibl.data.weather import (
    NwsWeather,
    VENUE_COORDS,
    INDOOR_VENUES,
    ROOF_TYPES,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


class TestNwsWeather:
    def test_venue_coords_has_30_entries(self):
        assert len(VENUE_COORDS) >= 30  # includes spring training venues

    def test_known_venue_has_coords(self):
        lat, lon = VENUE_COORDS[1]  # Angel Stadium
        assert 33 < lat < 34
        assert -118 < lon < -117

    def test_indoor_venues_known(self):
        assert 14 in INDOOR_VENUES  # Rogers Centre
        assert 15 in INDOOR_VENUES  # Chase Field

    def test_roof_types_known(self):
        assert ROOF_TYPES[14] == "retractable"
        assert ROOF_TYPES[2397] == "dome"

    def test_indoor_weather(self):
        result = NwsWeather._indoor_weather()
        assert result["temp"] == 72
        assert result["precip_pct"] == 0
        assert result["conditions"] == "Indoor"
        assert result["source"] == "indoor"

    def test_forecast_indoor_venue_returns_indoor(self):
        nws = NwsWeather()
        try:
            result = nws.forecast(14)  # Rogers Centre
            assert result["source"] == "indoor"
            assert result["conditions"] == "Indoor"
        finally:
            nws.close()

    def test_parse_period(self):
        period = {
            "temperature": 75,
            "windSpeed": "10 mph",
            "windDirection": "SW",
            "probabilityOfPrecipitation": {"value": 20},
            "shortForecast": "Partly Cloudy",
        }
        result = NwsWeather._parse_period(period, source="forecast")
        assert result["temp"] == 75
        assert result["wind_speed"] == "10 mph"
        assert result["wind_direction"] == "SW"
        assert result["precip_pct"] == 20
        assert result["conditions"] == "Partly Cloudy"
        assert result["source"] == "forecast"

    def test_parse_period_missing_precip(self):
        period = {
            "temperature": 68,
            "windSpeed": "5 mph",
            "windDirection": "N",
            "shortForecast": "Clear",
        }
        result = NwsWeather._parse_period(period, source="forecast")
        assert result["precip_pct"] is None

    def test_closest_period_returns_nearest(self):
        periods = [
            {"startTime": "2025-04-01T12:00:00+00:00", "temperature": 60},
            {"startTime": "2025-04-01T14:00:00+00:00", "temperature": 65},
            {"startTime": "2025-04-01T16:00:00+00:00", "temperature": 70},
        ]
        target = datetime(2025, 4, 1, 13, 30)
        best = NwsWeather._closest_period(periods, target)
        assert best["temperature"] == 65

    def test_closest_period_exact_match(self):
        periods = [
            {"startTime": "2025-04-01T12:00:00+00:00", "temperature": 60},
            {"startTime": "2025-04-01T14:00:00+00:00", "temperature": 65},
        ]
        target = datetime(2025, 4, 1, 14, 0)
        best = NwsWeather._closest_period(periods, target)
        assert best["temperature"] == 65

    def test_closest_period_empty_returns_first(self):
        periods = [{"startTime": "2025-04-01T12:00:00+00:00"}]
        target = datetime(2025, 4, 1, 0, 0)
        best = NwsWeather._closest_period(periods, target)
        assert best is not None

    def test_unknown_venue_returns_empty(self):
        nws = NwsWeather()
        try:
            result = nws.forecast(99999)
            assert result == {}
        finally:
            nws.close()

    def test_grid_is_cached(self):
        nws = NwsWeather()
        try:
            # Force a grid lookup that won't hit the API
            nws._grid_cache[1] = {"grid_id": "TEST"}
            grid = nws._get_grid(1)
            assert grid["grid_id"] == "TEST"
        finally:
            nws.close()


@pytest.mark.slow
class TestNwsWeatherLive:
    """Tests that hit the live NWS API.  Run with: pytest --runslow"""

    def test_live_forecast_open_air(self):
        nws = NwsWeather()
        try:
            result = nws.forecast(1)  # Angel Stadium
            assert result["temp"] is not None
            assert result["wind_speed"] is not None
            assert result["conditions"] is not None
            assert result["source"] == "forecast"
        finally:
            nws.close()

    def test_live_forecast_at_future_time(self):
        nws = NwsWeather()
        try:
            from datetime import timedelta
            future = datetime.now() + timedelta(hours=6)
            result = nws.forecast(1, target_time=future)
            assert result["temp"] is not None
        finally:
            nws.close()

    def test_live_observation_falls_back_to_forecast(self):
        nws = NwsWeather()
        try:
            from datetime import timedelta
            past = datetime.now() - timedelta(days=3)
            result = nws.observation(1, past)
            assert result["temp"] is not None
        finally:
            nws.close()
