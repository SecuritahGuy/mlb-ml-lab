"""Historical weather data via Meteostat (NOAA weather stations).

Fetches hourly observations from the nearest station per venue and caches
results to disk so the harvest only pays the download cost once.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from meteostat import Stations, Hourly

from mlb_ml_lab.data.weather import INDOOR_VENUES, VENUE_COORDS

logger = logging.getLogger(__name__)

# Wind direction degrees → compass point  (16-point compass)
_DEG_TO_COMPASS = [
    (11.25, "N"),
    (33.75, "NNE"),
    (56.25, "NE"),
    (78.75, "ENE"),
    (101.25, "E"),
    (123.75, "ESE"),
    (146.25, "SE"),
    (168.75, "SSE"),
    (191.25, "S"),
    (213.75, "SSW"),
    (236.25, "SW"),
    (258.75, "WSW"),
    (281.25, "W"),
    (303.75, "WNW"),
    (326.25, "NW"),
    (348.75, "NNW"),
    (360.0, "N"),
]

# Meteostat condition code → human-readable label
_COCO_LABELS: dict[int, str] = {
    1: "Clear",
    2: "Fair",
    3: "Cloudy",
    4: "Overcast",
    5: "Fog",
    6: "Freezing Fog",
    7: "Light Rain",
    8: "Rain",
    9: "Heavy Rain",
    10: "Freezing Rain",
    11: "Heavy Freezing Rain",
    12: "Sleet",
    13: "Heavy Sleet",
    14: "Light Snow",
    15: "Snow",
    16: "Heavy Snow",
    17: "Rain and Snow",
    18: "Light Showers",
    19: "Showers",
    20: "Heavy Showers",
    21: "Light Thunderstorms",
    22: "Thunderstorms",
    23: "Heavy Thunderstorms",
    24: "Tornado",
}


def _deg_to_compass(deg: float) -> str:
    for threshold, label in _DEG_TO_COMPASS:
        if deg < threshold:
            return label
    return "N"


def _coco_label(code: object) -> str | None:
    if code is None:
        return None
    try:
        return _COCO_LABELS.get(int(code), "Unknown")
    except (ValueError, TypeError):
        return None


_CACHE_DIR = Path("data/cache/meteostat")


class _RateLimiter:
    """Simple per-class rate limiter for meteostat HTTP requests."""

    def __init__(self, min_interval: float = 0.25) -> None:
        self._min_interval = min_interval
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = _time.time() - self._last_call
        if elapsed < self._min_interval:
            _time.sleep(self._min_interval - elapsed)
        self._last_call = _time.time()


class MeteostatWeather:
    """Historical weather for MLB venues via Meteostat.

    Usage::

        mw = MeteostatWeather()
        wx = mw.weather(venue_id=1, target_time=datetime(2024, 6, 1, 19, 0))
        # → {"temp": 72.0, "wind_speed": "3.4 mph", "wind_direction": "S",
        #     "precip_inches": 0.0, "conditions": "Clear"}
    """

    _rate_limiter = _RateLimiter(min_interval=0.25)

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._cache_dir = Path(cache_dir or _CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._station_cache: dict[int, str] = {}
        self._hourly_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def weather(
        self,
        venue_id: int,
        target_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Return observed weather at a venue near *target_time*.

        Returns fields consistent with the existing ``NwsWeather.forecast()``
        output so extractors remain compatible.
        """
        if venue_id in INDOOR_VENUES:
            return self._indoor_weather()

        if target_time is None:
            target_time = datetime.now()

        station = self._station_for_venue(venue_id)
        if station is None:
            return {}

        df = self._hourly_for_station(station, target_time)
        if df is None or df.empty:
            return {}

        row = self._nearest_row(df, target_time)
        if row is None:
            return {}

        return {
            "temp": None if pd.isna(row.get("temp")) else row["temp"],
            "wind_speed": f"{row.get('wspd', 0):.0f} mph"
            if pd.notna(row.get("wspd"))
            else "0 mph",
            "wind_direction": _deg_to_compass(row["wdir"])
            if pd.notna(row.get("wdir"))
            else "",
            "precip_inches": float(row["prcp"]) if pd.notna(row.get("prcp")) else 0.0,
            "conditions": _coco_label(row.get("coco")) or "Unknown",
            "source": "meteostat",
        }

    def prefetch(
        self,
        venue_ids: list[int],
        start: datetime,
        end: datetime,
    ) -> None:
        """Pre-cache hourly data for a set of venues over a date range."""
        for vid in venue_ids:
            if vid in INDOOR_VENUES:
                continue
            station = self._station_for_venue(vid)
            if station is None:
                continue
            self._hourly_for_station(station, start, end)

    # ------------------------------------------------------------------
    # Station resolution
    # ------------------------------------------------------------------

    def _station_for_venue(self, venue_id: int) -> str | None:
        if venue_id in self._station_cache:
            return self._station_cache[venue_id]

        coords = VENUE_COORDS.get(venue_id)
        if not coords:
            return None

        lat, lon = coords
        nearby = Stations().nearby(lat, lon)
        if nearby is None or nearby.empty:
            return None

        station_id = nearby.index[0]
        self._station_cache[venue_id] = station_id
        return station_id

    # ------------------------------------------------------------------
    # Hourly data fetch with disk + memory cache
    # ------------------------------------------------------------------

    def _hourly_for_station(
        self,
        station_id: str,
        target_time: datetime,
        _end: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Fetch hourly data for *station_id* covering *target_time*.

        Caches on disk keyed by ``{station_id}_{YYYY}_{MM}`` so we only
        download each month once.
        """
        month_key = target_time.strftime("%Y_%m")
        cache_key = f"{station_id}_{month_key}"

        # Memory cache
        if cache_key in self._hourly_cache:
            return self._hourly_cache[cache_key]

        # Disk cache
        cache_path = self._cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._hourly_cache[cache_key] = df
            return df

        # Download from meteostat
        month_start = target_time.replace(day=1, hour=0, minute=0, second=0)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)

        for attempt in range(2):
            self._rate_limiter.wait()
            try:
                ts = Hourly(station_id, month_start, month_end)
                df = ts.fetch()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Meteostat fetch failed for %s %s (attempt %d)",
                    station_id,
                    month_key,
                    attempt + 1,
                )
                df = None

            if df is not None and not df.empty:
                df.to_parquet(cache_path)
                self._hourly_cache[cache_key] = df
                return df

            # Brief delay before retry for transient errors
            _time.sleep(0.5)

        # Both attempts failed — cache failure so we don't retry this
        # station+month for every subsequent game at this venue
        pd.DataFrame().to_parquet(cache_path)
        self._hourly_cache[cache_key] = pd.DataFrame()
        return None

    @staticmethod
    def _nearest_row(
        df: pd.DataFrame,
        target: datetime,
    ) -> dict[str, Any] | None:
        if df.empty:
            return None
        # Strip timezone from target to match naive DatetimeIndex
        if target.tzinfo is not None:
            target_naive = target.replace(tzinfo=None)
        else:
            target_naive = target
        idx = df.index.get_indexer([target_naive], method="nearest")
        if idx[0] == -1:
            return None
        return dict(df.iloc[idx[0]])

    @staticmethod
    def _indoor_weather() -> dict[str, Any]:
        return {
            "temp": 72,
            "wind_speed": "0 mph",
            "wind_direction": "",
            "precip_inches": 0.0,
            "conditions": "Indoor",
            "source": "indoor",
        }


def venue_date_ranges(
    game_logs: list[Any],
    venue_map: dict[int, int],
) -> dict[int, tuple[datetime, datetime]]:
    """Compute (start, end) datetime ranges per venue from game logs."""
    by_venue: dict[int, list[datetime]] = {}
    for log in game_logs:
        home_id = log.team_id if log.is_home else log.opponent_id
        vid = venue_map.get(home_id)
        if vid is None or vid in INDOOR_VENUES:
            continue
        dt = datetime.fromisoformat(log.date)
        by_venue.setdefault(vid, []).append(dt)

    ranges: dict[int, tuple[datetime, datetime]] = {}
    for vid, dts in by_venue.items():
        ranges[vid] = (min(dts), max(dts))
    return ranges
