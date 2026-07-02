"""League-wide context features: season-level averages for BA, OBP, SLG, OPS.

Provides context for whether a player's performance is above or below
league average.  Requires ``league_stats`` in kwargs — a single dict
with keys like ``avg``, ``obp``, ``slg``, ``ops``, ``runs_per_game``.

When absent, all features default to None.
"""

from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class LeagueContextFeatures(FeatureExtractor):
    """Season-level league batting context."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(name="league_avg",
                        description="League-wide batting average for the season",
                        source="league"),
            FeatureMeta(name="league_obp",
                        description="League-wide on-base percentage",
                        source="league"),
            FeatureMeta(name="league_slg",
                        description="League-wide slugging percentage",
                        source="league"),
            FeatureMeta(name="league_ops",
                        description="League-wide OPS", source="league"),
            FeatureMeta(name="league_runs_per_game",
                        description="League-wide runs per game",
                        source="league"),
        ]

    def extract(self, game_logs: list[Any], **kwargs: Any) -> list[dict[str, Any]]:
        league: dict[str, float] | None = kwargs.get("league_stats")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "league_avg": (league or {}).get("avg"),
                "league_obp": (league or {}).get("obp"),
                "league_slg": (league or {}).get("slg"),
                "league_ops": (league or {}).get("ops"),
                "league_runs_per_game": (league or {}).get("runs_per_game"),
            })
        return rows
