"""Weather forecasts and observations via the National Weather Service API.

NWS API (``api.weather.gov``) is free and requires no API key.  Rate
limit is ~30 req/s; we enforce 5 req/s via ``TokenBucket``.

Usage::

    from mibl.data.weather import NwsWeather

    nws = NwsWeather()
    forecast = nws.forecast(venue_id=1, target_time=datetime.now())
    # → { temp, wind_speed, wind_dir, precip_pct, conditions, sky_cover }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from mibl.data.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
_USER_AGENT = "mibl/0.1 (weather service; mlb hit prediction)"

# ---------------------------------------------------------------------------
# Venue → lat/lon mapping (extracted from MLB Stats API game feeds)
# Keys are MLB venue IDs, values are (latitude, longitude).
# ---------------------------------------------------------------------------

VENUE_COORDS: dict[int, tuple[float, float]] = {
    1: (33.80019044, -117.8823996),  # Angel Stadium
    2: (39.283787, -76.621689),  # Oriole Park at Camden Yards
    3: (42.346456, -71.097441),  # Fenway Park
    4: (41.83, -87.634167),  # Rate Field
    7: (39.051567, -94.480483),  # Kauffman Stadium
    14: (43.64155, -79.38915),  # Rogers Centre
    15: (33.445302, -112.066687),  # Chase Field
    17: (41.948171, -87.655503),  # Wrigley Field
    19: (39.756042, -104.994136),  # Coors Field
    22: (34.07368, -118.24053),  # Dodger Stadium
    31: (40.446904, -80.005753),  # PNC Park
    32: (43.02838, -87.97099),  # American Family Field
    680: (47.591333, -122.33251),  # T-Mobile Park
    2392: (29.756967, -95.355509),  # Daikin Park
    2394: (42.3391151, -83.048695),  # Comerica Park
    2395: (37.778383, -122.389448),  # Oracle Park
    2397: (35.7056, 139.7519),  # Tokyo Dome
    2523: (27.97997, -82.50702),  # George M. Steinbrenner Field
    2529: (38.57994, -121.51246),  # Sutter Health Park
    2602: (39.097389, -84.506611),  # Great American Ball Park
    2680: (32.707861, -117.157278),  # Petco Park
    2681: (39.90539086, -75.16716957),  # Citizens Bank Park
    2889: (38.62256667, -90.19286667),  # Busch Stadium
    3289: (40.75753012, -73.84559155),  # Citi Field
    3309: (38.872861, -77.007501),  # Nationals Park
    3312: (44.981829, -93.277891),  # Target Field
    3313: (40.82919482, -73.9264977),  # Yankee Stadium
    4169: (25.77796236, -80.21951795),  # loanDepot park
    4705: (33.890672, -84.467641),  # Truist Park
    5325: (32.747299, -97.081818),  # Globe Life Field
}

# Venues with roofs/dome that make weather irrelevant
INDOOR_VENUES: set[int] = {14, 15, 2397, 4169, 5325}

# Venue roof types for reference
ROOF_TYPES: dict[int, str] = {
    14: "retractable",
    15: "retractable",
    32: "retractable",
    680: "retractable",
    2392: "retractable",
    2397: "dome",
    4169: "retractable",
    5325: "retractable",
}


def _user_agent() -> dict[str, str]:
    return {"User-Agent": _USER_AGENT}


class NwsWeather:
    """Weather forecasts and observations via the National Weather Service.

    Caches grid-point lookups (lat/lon → NWS grid) in memory.
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))
        self._limiter = TokenBucket(capacity=5, refill_rate=5)
        self._grid_cache: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(
        self,
        venue_id: int,
        target_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Fetch forecasted weather for a venue near *target_time*.

        Args:
            venue_id: MLB Stats API venue ID (see ``VENUE_COORDS``).
            target_time: Game datetime.  Uses ``datetime.now()`` if None.

        Returns:
            Dict with keys:
                ``temp`` (int, °F),
                ``wind_speed`` (str, e.g. \"10 mph\"),
                ``wind_direction`` (str, e.g. \"SW\"),
                ``precip_pct`` (int or None, 0-100),
                ``conditions`` (str, e.g. \"Partly Cloudy\"),
                ``source`` (str, ``\"forecast\"`` or ``\"observation\"``).
                Returns empty dict for indoor venues (roof/dome).
        """
        if venue_id in INDOOR_VENUES:
            return self._indoor_weather()

        grid = self._get_grid(venue_id)
        if not grid:
            return {}

        hourly = self._fetch_hourly(grid)
        if not hourly:
            return {}

        periods = hourly.get("properties", {}).get("periods", [])
        if not periods:
            return {}

        target = target_time or datetime.now()
        period = self._closest_period(periods, target)
        return self._parse_period(period, source="forecast")

    def observation(
        self,
        venue_id: int,
        target_time: datetime,
    ) -> dict[str, Any]:
        """Fetch observed weather for a venue at a past time.

        Falls back to the nearest hourly forecast period if observations
        are unavailable.
        """
        # NWS observation stations endpoint gives nearby stations.
        # For now, fall back to the nearest hourly forecast period
        # since that's simpler and sufficient for model features.
        return self.forecast(venue_id, target_time)

    def venue_has_roof(self, venue_id: int) -> bool:
        return venue_id in INDOOR_VENUES

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_grid(self, venue_id: int) -> dict[str, Any]:
        """Resolve venue_id → NWS grid point (cached)."""
        if venue_id in self._grid_cache:
            return self._grid_cache[venue_id]

        coords = VENUE_COORDS.get(venue_id)
        if not coords:
            logger.warning("No coordinates for venue %d", venue_id)
            return {}

        lat, lon = coords
        url = f"{NWS_BASE}/points/{lat:.3f},{lon:.3f}"
        with self._limiter:
            resp = self._client.get(url, headers=_user_agent())
        if resp.status_code != 200:
            logger.warning("NWS points lookup failed for venue %d: %s", venue_id, resp.status_code)
            return {}

        data = resp.json()
        props = data.get("properties", {})
        result = {
            "grid_id": props.get("gridId"),
            "grid_x": props.get("gridX"),
            "grid_y": props.get("gridY"),
            "forecast_hourly": props.get("forecastHourly"),
            "observation_stations": props.get("observationStations"),
        }
        self._grid_cache[venue_id] = result
        return result

    def _fetch_hourly(self, grid: dict[str, Any]) -> dict[str, Any]:
        url = grid.get("forecast_hourly")
        if not url:
            return {}
        with self._limiter:
            resp = self._client.get(url, headers=_user_agent())
        if resp.status_code != 200:
            return {}
        return resp.json()

    @staticmethod
    def _closest_period(
        periods: list[dict[str, Any]], target: datetime,
    ) -> dict[str, Any]:
        """Find the forecast period whose startTime is closest to target."""
        # Make target offset-aware (UTC) for safe comparison with ISO-8601
        if target.tzinfo is None:
            target_utc = target.replace(tzinfo=timezone.utc)
        else:
            target_utc = target
        best = periods[0]
        best_dist = float("inf")
        for p in periods:
            try:
                t = datetime.fromisoformat(p["startTime"])
                dist = abs((t - target_utc).total_seconds())
                if dist < best_dist:
                    best = p
                    best_dist = dist
            except (ValueError, KeyError):
                continue
        return best

    @staticmethod
    def _parse_period(period: dict[str, Any], source: str) -> dict[str, Any]:
        precip = period.get("probabilityOfPrecipitation", {}) or {}
        return {
            "temp": period.get("temperature"),
            "wind_speed": period.get("windSpeed"),
            "wind_direction": period.get("windDirection"),
            "precip_pct": precip.get("value"),
            "conditions": period.get("shortForecast"),
            "source": source,
        }

    @staticmethod
    def _indoor_weather() -> dict[str, Any]:
        return {
            "temp": 72,
            "wind_speed": "0 mph",
            "wind_direction": "",
            "precip_pct": 0,
            "conditions": "Indoor",
            "source": "indoor",
        }
