from __future__ import annotations

import csv
import io
import logging
from typing import Any

import httpx

from mlb_ml_lab.data.cache import DiskCache
from mlb_ml_lab.data.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"
CACHE_DIR = "data/cache"


def _float_or_none(val: str | float | None) -> float | None:
    if val is None or val == "" or val == ".---":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _str_avg_or_none(val: str | None) -> float | None:
    """Parse MLB batting-average string ('.260') to float (0.260)."""
    if val is None or val in ("", ".---", "----"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class MlbClient:
    """Client for the official MLB Stats API (statsapi.mlb.com).

    Rate-limited to 10 requests/second with a 5-second burst.
    Responses are cached to disk with a default 24h TTL.
    """

    def __init__(
        self,
        base_url: str = MLB_API_BASE,
        timeout: float = 30.0,
        cache_dir: str = CACHE_DIR,
        cache_ttl: int = 86_400,
        rate_limit: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))
        self._cache = DiskCache(cache_dir, default_ttl=cache_ttl)
        self._limiter = TokenBucket(capacity=rate_limit, refill_rate=rate_limit)

    # ------------------------------------------------------------------
    # Teams & Rosters
    # ------------------------------------------------------------------

    def get_teams(self, sport_id: int = 1) -> list[dict[str, Any]]:
        data = self._get("/teams", params={"sportId": sport_id})
        return data.get("teams", [])

    def get_roster(
        self, team_id: int, season: int, roster_type: str = "40Man"
    ) -> list[dict[str, Any]]:
        data = self._get(
            f"/teams/{team_id}/roster",
            params={"season": season, "rosterType": roster_type},
        )
        return data.get("roster", [])

    # ------------------------------------------------------------------
    # Player Game Logs
    # ------------------------------------------------------------------

    def get_player_game_log(
        self, player_id: int, season: int, group: str = "hitting"
    ) -> list[dict[str, Any]]:
        data = self._get(
            f"/people/{player_id}/stats",
            params={"stats": "gameLog", "group": group, "season": season},
        )
        stats = data.get("stats", [])
        if not stats:
            return []
        return stats[0].get("splits", [])

    # ------------------------------------------------------------------
    # Schedule / Games
    # ------------------------------------------------------------------

    def get_season_schedule(
        self, season: int, sport_id: int = 1, game_type: str = "R"
    ) -> list[dict[str, Any]]:
        data = self._get(
            "/schedule",
            params={"sportId": sport_id, "season": season, "gameType": game_type},
        )
        games: list[dict[str, Any]] = []
        for date_entry in data.get("dates", []):
            games.extend(date_entry.get("games", []))
        return games

    # ------------------------------------------------------------------
    # Game Context (weather, venue, score)
    # ------------------------------------------------------------------

    def get_game_context(self, game_pk: int) -> dict[str, Any]:
        """Return weather, venue, and boxscore summary for a game."""
        data = self._get(f"/game/{game_pk}/feed/live", version="v1.1")
        gi = data.get("gameData", {})
        ld = data.get("liveData", {})

        weather = gi.get("weather", {}) or {}
        venue = gi.get("venue", {}) or {}
        game_info = gi.get("game", {}) or {}
        datetime_info = gi.get("datetime", {}) or {}

        bs = ld.get("boxscore", {}) or {}
        teams_bs = bs.get("teams", {}) or {}

        home = teams_bs.get("home", {}) or {}
        away = teams_bs.get("away", {}) or {}

        home_team = gi.get("teams", {}).get("home", {}) or {}
        away_team = gi.get("teams", {}).get("away", {}) or {}

        return {
            "game_pk": game_pk,
            "game_date": game_info.get("date"),
            "game_datetime": datetime_info.get("dateTime"),
            "day_night": game_info.get("dayNight"),
            "venue_id": venue.get("id"),
            "venue_name": venue.get("name"),
            "weather_condition": weather.get("condition"),
            "weather_temp": weather.get("temp"),
            "weather_wind": weather.get("wind"),
            "home_team_id": home_team.get("id"),
            "home_team_name": home_team.get("name"),
            "away_team_id": away_team.get("id"),
            "away_team_name": away_team.get("name"),
            "home_score": home.get("runs"),
            "away_score": away.get("runs"),
            "home_hits": home.get("hits"),
            "away_hits": away.get("hits"),
            "status": game_info.get("status", {}).get("detailedState"),
        }

    # ------------------------------------------------------------------
    # Player Details
    # ------------------------------------------------------------------

    def get_player(self, player_id: int, season: int | None = None) -> dict[str, Any]:
        """Fetch full player details (name, age, bats/throws, position, etc.).

        Args:
            player_id: MLB Stats API person ID.
            season: If provided, includes ``currentTeam`` hydration.

        Returns:
            A single player dict with keys like ``fullName``, ``birthDate``,
            ``height``, ``weight``, ``batSide``, ``pitchHand``,
            ``primaryPosition``, ``mlbDebutDate``, ``currentTeam``, etc.
        """
        params: dict[str, Any] = {"hydrate": "currentTeam"}
        if season is not None:
            params["season"] = season
        data = self._get(f"/people/{player_id}", params=params)
        people = data.get("people", [])
        return people[0] if people else {}

    def get_player_season_stats(
        self, player_id: int, season: int, group: str = "hitting"
    ) -> dict[str, Any]:
        """Fetch a player's season-level aggregated stats.

        Args:
            player_id: MLB Stats API person ID.
            season: Season year.
            group: ``"hitting"`` or ``"pitching"``.

        Returns:
            Dict of stat keys (e.g. ``avg``, ``homeRuns``, ``rbi``, ``era``,
            ``strikeOuts``, ``whip``, etc.) or empty dict.
        """
        data = self._get(
            f"/people/{player_id}/stats",
            params={"stats": "season", "group": group, "season": season},
        )
        stats = data.get("stats", [])
        if not stats:
            return {}
        splits = stats[0].get("splits", [])
        return splits[0].get("stat", {}) if splits else {}

    # ------------------------------------------------------------------
    # Team Stats
    # ------------------------------------------------------------------

    def get_team_hitting_stats(
        self, team_ids: list[int], season: int
    ) -> dict[int, dict[str, Any]]:
        """Fetch season-level hitting stats for one or more teams.

        Returns a dict mapping team_id → stat dict (avg, homeRuns, runs,
        obp, slg, ops, etc.).
        """
        result: dict[int, dict[str, Any]] = {}
        for tid in team_ids:
            data = self._get(
                f"/teams/{tid}/stats",
                params={"season": season, "group": "hitting", "stats": "season"},
            )
            stats = data.get("stats", [])
            if not stats:
                continue
            splits = stats[0].get("splits", [])
            if not splits:
                continue
            result[tid] = splits[0].get("stat", {})
        return result

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------

    def get_standings(
        self, season: int, league_id: int = 103
    ) -> list[dict[str, Any]]:
        """Fetch division standings for a league.

        Args:
            season: Season year.
            league_id: 103 = AL, 104 = NL.

        Returns:
            List of record dicts sorted by division. Each dict includes
            ``team``, ``leagueRecord``, ``divisionRank``, ``leagueRank``,
            ``gamesBack``, ``wildCardGamesBack``, ``runsScored``,
            ``runsAllowed``, ``streak``, ``records`` (home/away/last10).
        """
        data = self._get(
            "/standings",
            params={"leagueId": league_id, "season": season},
        )
        records: list[dict[str, Any]] = []
        for division in data.get("records", []):
            for tr in division.get("teamRecords", []):
                tr["league"] = {"id": league_id}
                tr["division"] = {
                    "id": division.get("division", {}).get("id"),
                    "name": division.get("division", {}).get("name"),
                }
                records.append(tr)
        return records

    # ------------------------------------------------------------------
    # Game Boxscore
    # ------------------------------------------------------------------

    def get_boxscore(self, game_pk: int) -> dict[str, Any]:
        """Fetch full game boxscore with player-level stats.

        Returns a dict with keys::

            teams: {
                home: { players: { ID: { person, position, battingOrder, stats, ... } }, ... },
                away: { ... }
            },
            officials, info, topPerformers

        Each player entry contains ``battingOrder`` (int, e.g. 600 = 6th),
        ``stats`` (batting/pitching game stats), ``seasonStats``,
        ``gameStatus`` (isCurrentBatter, isSubstitute, etc.).
        """
        return self._get(f"/game/{game_pk}/boxscore")

    # ------------------------------------------------------------------
    # Venues
    # ------------------------------------------------------------------

    def get_venue(self, venue_id: int) -> dict[str, Any]:
        """Fetch venue details (name, location, active status).

        Returns a dict with keys: ``name``, ``location`` (city/state/
        country), ``active``, ``season``.
        """
        data = self._get(f"/venues/{venue_id}", params={"hydrate": "location"})
        venues = data.get("venues", [])
        return venues[0] if venues else {}

    # ------------------------------------------------------------------
    # Team Pitching Stats (for opponent-pitching features)
    # ------------------------------------------------------------------

    def get_team_pitching_stats(
        self, team_ids: list[int], season: int
    ) -> dict[int, dict[str, float]]:
        """Fetch season-level pitching stats for one or more teams.

        Returns a dict mapping team_id → dict with keys: ``era``,
        ``k_per_9``, ``whip``, ``ba_against``, ``hr_per_9``.

        This is the format expected by ``TeamPitchingFeatures``.
        """
        result: dict[int, dict[str, float]] = {}
        for tid in team_ids:
            data = self._get(
                f"/teams/{tid}/stats",
                params={"season": season, "group": "pitching", "stats": "season"},
            )
            stats = data.get("stats", [])
            if not stats:
                continue
            splits = stats[0].get("splits", [])
            if not splits:
                continue
            s = splits[0].get("stat", {})
            # Map MLB API field names → our feature keys
            era = _float_or_none(s.get("era"))
            k_per_9 = _float_or_none(s.get("strikeoutsPer9Inn"))
            whip = _float_or_none(s.get("whip"))
            ba_against = _str_avg_or_none(s.get("avg"))
            hr_per_9 = _float_or_none(s.get("homeRunsPer9"))
            if era is not None:
                result[tid] = {
                    "era": era,
                    "k_per_9": k_per_9 or 0.0,
                    "whip": whip or 0.0,
                    "ba_against": ba_against or 0.0,
                    "hr_per_9": hr_per_9 or 0.0,
                }
        return result

    # ------------------------------------------------------------------
    # Baseball Savant CSV endpoints (statcast)
    # ------------------------------------------------------------------

    def get_statcast_batters(self, season: int, min_qual: str = "q") -> list[dict[str, str]]:
        """Fetch statcast batters leaderboard (barrel %, exit velo, launch angle)."""
        return self._fetch_savant_csv(
            "/leaderboard/statcast",
            params={"type": "batter", "year": season, "min": min_qual, "csv": "true"},
        )

    def get_expected_stats(self, season: int, min_qual: str = "q") -> list[dict[str, str]]:
        """Fetch expected stats leaderboard (xBA, xSLG, xwOBA)."""
        return self._fetch_savant_csv(
            "/leaderboard/expected_statistics",
            params={"type": "batter", "year": season, "min": min_qual, "csv": "true"},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(
        self, path: str, params: dict[str, Any] | None = None, version: str = "v1"
    ) -> dict[str, Any]:
        base = self._base_url.replace("/v1", f"/{version}")
        cache_key = self._cache_key(path, params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{base}{path}"
        logger.debug("GET %s %s", url, params)
        with self._limiter:
            resp = self._client.get(url, params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        self._cache.set(cache_key, data)
        return data

    def _fetch_savant_csv(
        self, path: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, str]]:
        cache_key = f"savant:{self._cache_key(path, params)}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{SAVANT_BASE}{path}"
        logger.debug("GET %s %s", url, params)
        with self._limiter:
            resp = self._client.get(url, params=params)
        resp.raise_for_status()

        text = resp.text.lstrip("\ufeff")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        self._cache.set(cache_key, rows)
        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(path: str, params: dict[str, Any] | None) -> str:
        if not params:
            return path
        sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{path}?{sorted_params}"

    def close(self) -> None:
        self._client.close()
