"""Game-context features: park factors, home/away, weather, rest days."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mlb_ml_lab.data.parks import ParkFactors
from mlb_ml_lab.data.schemas import PlayerGameLog

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class ParkFactorFeatures(FeatureExtractor):
    """Park factor adjustments for each game's venue.

    Requires ``teams`` (list of team dicts from ``MlbClient.get_teams()``)
    in kwargs to resolve team_id → venue_id.  If absent, all factors
    default to 1.0 (neutral).
    """

    def __init__(self) -> None:
        self._pf = ParkFactors()

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="park_wOBA",
                description="Park factor for wOBA (ratio, 1.0 = neutral)",
                source="context",
            ),
            FeatureMeta(
                name="park_HR",
                description="Park factor for home runs",
                source="context",
            ),
            FeatureMeta(
                name="park_1B",
                description="Park factor for singles",
                source="context",
            ),
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        teams: list[dict[str, Any]] | None = kwargs.get("teams")
        venue_map = self._resolve_venue_map(teams)
        season: int | None = kwargs.get("season")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            factors = self._factors_for_game(log, venue_map, season)
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "park_wOBA": factors.get("wOBA", 1.0),
                "park_HR": factors.get("HR", 1.0),
                "park_1B": factors.get("1B", 1.0),
            })
        return rows

    def _factors_for_game(
        self,
        log: PlayerGameLog,
        venue_map: dict[int, int],
        season: int | None,
    ) -> dict[str, float]:
        home_team_id = log.team_id if log.is_home else log.opponent_id
        venue_id = venue_map.get(home_team_id)
        if venue_id is None:
            return {}
        return {
            "wOBA": self._pf.factor(venue_id, "wOBA", season=season),
            "HR": self._pf.factor(venue_id, "HR", season=season),
            "1B": self._pf.factor(venue_id, "1B", season=season),
        }

    @staticmethod
    def _resolve_venue_map(
        teams: list[dict[str, Any]] | None,
    ) -> dict[int, int]:
        if not teams:
            return {}
        mapping: dict[int, int] = {}
        for t in teams:
            venue = t.get("venue") or {}
            venue_id = venue.get("id")
            if venue_id is not None:
                mapping[t["id"]] = venue_id
        return mapping


@register
class HomeAwayFeature(FeatureExtractor):
    """Whether the player is playing at home."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="is_home",
                description="1 if game is at player's home stadium",
                source="context",
            )
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "is_home": 1 if log.is_home else 0,
            }
            for log in game_logs
        ]


@register
class RestDaysFeature(FeatureExtractor):
    """Days of rest since the player's last game."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="rest_days",
                description="Days since player's last game",
                source="context",
            )
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: dict[int, str | None] = {}
        for log in game_logs:
            last_date = seen.get(log.player_id)
            rest: int | None = None
            if last_date is not None:
                try:
                    d1 = datetime.strptime(last_date, "%Y-%m-%d")
                    d2 = datetime.strptime(log.date, "%Y-%m-%d")
                    rest = (d2 - d1).days - 1
                except ValueError:
                    rest = None
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "rest_days": rest,
            })
            seen[log.player_id] = log.date

        return rows


@register
class WeatherFeatures(FeatureExtractor):
    """Weather conditions at game time (condition, temperature, wind).

    Requires ``game_contexts`` in kwargs — a dict mapping game_pk → dict
    with keys ``weather_condition``, ``weather_temp``, ``weather_wind``
    (as returned by ``MlbClient.get_game_context()``).
    If absent, all weather values default to None.
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="weather_condition",
                description="Weather condition string (e.g. 'Cloudy', 'Clear')",
                source="context",
            ),
            FeatureMeta(
                name="weather_temp",
                description="Temperature at game time (F)",
                source="context",
            ),
            FeatureMeta(
                name="weather_wind",
                description="Wind at game time (mph/direction)",
                source="context",
            ),
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        contexts: dict[int, dict[str, Any]] | None = kwargs.get("game_contexts")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            ctx = (contexts or {}).get(log.game_pk, {}) if contexts else {}
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "weather_condition": ctx.get("weather_condition"),
                "weather_temp": ctx.get("weather_temp"),
                "weather_wind": ctx.get("weather_wind"),
            })
        return rows
