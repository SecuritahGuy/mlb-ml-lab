"""Dynamic park factors fetched from Baseball Savant.

Usage::

    from mibl.data.parks import ParkFactors

    pf = ParkFactors()
    coors_woba = pf.factor(19, "wOBA", season=2025)  # 1.12

Factors are 3-year rolling indexes (100 = neutral, >100 = hitter-friendly).
Fetched live from baseballsavant.mlb.com; falls back to static data for
years or venues the scraper cannot reach.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from mibl.data.cache import DiskCache

logger = logging.getLogger(__name__)

SAVANT_PARK_FACTORS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=year&year={season}&batSide=&stat=index_wOBA"
    "&condition=All&rolling=3&parks=mlb"
)

CACHE_DIR = "data/cache"

# Static fallback: venue_id -> {metric: index}
# Used when the Savant scrape fails or returns no data for a venue.
_FALLBACK: dict[int, dict[str, float]] = {
    15: {"wOBA": 99, "HR": 104, "1B": 99, "2B": 99, "3B": 89},
    2529: {"wOBA": 101, "HR": 108, "1B": 99, "2B": 100, "3B": 97},
    4705: {"wOBA": 100, "HR": 105, "1B": 99, "2B": 98, "3B": 93},
    2: {"wOBA": 101, "HR": 107, "1B": 99, "2B": 100, "3B": 89},
    3: {"wOBA": 100, "HR": 99, "1B": 103, "2B": 104, "3B": 102},
    17: {"wOBA": 102, "HR": 105, "1B": 101, "2B": 101, "3B": 99},
    4: {"wOBA": 99, "HR": 101, "1B": 99, "2B": 99, "3B": 90},
    2602: {"wOBA": 102, "HR": 112, "1B": 99, "2B": 99, "3B": 91},
    5: {"wOBA": 97, "HR": 96, "1B": 99, "2B": 97, "3B": 93},
    19: {"wOBA": 110, "HR": 116, "1B": 105, "2B": 107, "3B": 122},
    2394: {"wOBA": 98, "HR": 94, "1B": 100, "2B": 98, "3B": 98},
    2392: {"wOBA": 99, "HR": 102, "1B": 99, "2B": 97, "3B": 93},
    7: {"wOBA": 97, "HR": 92, "1B": 100, "2B": 98, "3B": 96},
    1: {"wOBA": 99, "HR": 99, "1B": 100, "2B": 99, "3B": 94},
    22: {"wOBA": 99, "HR": 99, "1B": 99, "2B": 100, "3B": 97},
    4169: {"wOBA": 97, "HR": 94, "1B": 99, "2B": 98, "3B": 91},
    32: {"wOBA": 100, "HR": 103, "1B": 99, "2B": 99, "3B": 91},
    3312: {"wOBA": 98, "HR": 96, "1B": 100, "2B": 99, "3B": 94},
    3289: {"wOBA": 98, "HR": 97, "1B": 99, "2B": 99, "3B": 89},
    3313: {"wOBA": 100, "HR": 108, "1B": 100, "2B": 99, "3B": 87},
    2681: {"wOBA": 102, "HR": 106, "1B": 100, "2B": 102, "3B": 97},
    31: {"wOBA": 98, "HR": 97, "1B": 99, "2B": 99, "3B": 93},
    2680: {"wOBA": 96, "HR": 93, "1B": 99, "2B": 97, "3B": 92},
    2395: {"wOBA": 96, "HR": 89, "1B": 100, "2B": 98, "3B": 99},
    680: {"wOBA": 95, "HR": 90, "1B": 98, "2B": 96, "3B": 92},
    2889: {"wOBA": 97, "HR": 93, "1B": 99, "2B": 99, "3B": 89},
    12: {"wOBA": 97, "HR": 96, "1B": 99, "2B": 97, "3B": 94},
    5325: {"wOBA": 100, "HR": 103, "1B": 99, "2B": 101, "3B": 93},
    14: {"wOBA": 101, "HR": 105, "1B": 100, "2B": 101, "3B": 93},
    3309: {"wOBA": 100, "HR": 105, "1B": 99, "2B": 99, "3B": 92},
}

# Savant JSON field name -> our metric
_METRIC_MAP: dict[str, str] = {
    "index_woba": "wOBA",
    "index_hr": "HR",
    "index_1b": "1B",
    "index_2b": "2B",
    "index_3b": "3B",
    "index_hits": "hits",
    "index_runs": "runs",
}


class ParkFactors:
    """Park factors for MLB venues, fetched per-season from Baseball Savant.

    Caches results for the current process lifetime (not to disk — the raw
    data is compact and refreshes daily on Savant).
    """

    def __init__(self, cache_dir: str = CACHE_DIR, timeout: float = 15.0) -> None:
        self._cache: dict[int, dict[int, dict[str, float]]] = {}
        self._disk_cache = DiskCache(cache_dir, default_ttl=86400)
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))

    def factor(
        self, venue_id: int, metric: str = "wOBA", season: int | None = None
    ) -> float:
        """Return park factor index for a venue and season (100 = neutral).

        Args:
            venue_id: MLB Stats API venue ID.
            metric: One of 'wOBA', 'HR', '1B', '2B', '3B', 'hits', 'runs'.
            season: 4-digit season year. Defaults to the most recent
                    season available.

        Returns:
            Factor as a ratio (e.g. 1.05 = 5% above average). Returns 1.0
            when no data is available.
        """
        season_data = self._for_season(season)
        venue_data = season_data.get(venue_id, _FALLBACK.get(venue_id, {}))
        raw = venue_data.get(metric, 100.0)
        return raw / 100.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _for_season(
        self, season: int | None
    ) -> dict[int, dict[str, float]]:
        season = season or 2025
        if season in self._cache:
            return self._cache[season]

        data = self._load_from_savant(season)

        # Merge with fallback for any venues Savant didn't return
        for vid, fallback in _FALLBACK.items():
            if vid not in data:
                data[vid] = dict(fallback)

        self._cache[season] = data
        return data

    def _load_from_savant(self, season: int) -> dict[int, dict[str, float]]:
        """Fetch park factors from Baseball Savant (scrapes embedded JS var)."""
        cache_key = f"park_factors:{season}"
        cached = self._disk_cache.get(cache_key)
        if cached is not None:
            return {int(k): v for k, v in cached.items()}

        url = SAVANT_PARK_FACTORS_URL.format(season=season)
        logger.debug("Fetching park factors from %s", url)
        resp = self._client.get(url)
        resp.raise_for_status()

        raw = self._extract_json(resp.text)
        if raw is None:
            logger.warning("Could not extract park factors for %d", season)
            return {}

        result: dict[int, dict[str, float]] = {}
        for entry in raw:
            vid = int(entry["venue_id"])
            result[vid] = {}
            for savant_key, our_key in _METRIC_MAP.items():
                val = entry.get(savant_key)
                if val is not None:
                    result[vid][our_key] = float(val)

        self._disk_cache.set(cache_key, {str(k): v for k, v in result.items()})
        return result

    @staticmethod
    def _extract_json(html: str) -> list[dict[str, Any]] | None:
        """Extract the ``var data = [...]`` JSON array from Savant's HTML."""
        match = re.search(r"var data\s*=\s*(\[.*?\]);", html, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(1))

    def close(self) -> None:
        self._client.close()
