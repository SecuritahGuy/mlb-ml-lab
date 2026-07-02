"""Opponent matchup features: team-level pitching stats faced by each batter.

These features describe the quality of the opposing pitching staff for each
game.  Since individual pitcher data is expensive to collect, we use the
opponent team's season-level or rolling pitching aggregates.

Requires ``opponent_pitching`` in kwargs — a dict mapping team_id → dict
of pitching stat keys:
    ``era``, ``k_per_9``, ``whip``, ``ba_against``, ``hr_per_9``
"""

from __future__ import annotations

from typing import Any

from mlb_ml_lab.data.schemas import PlayerGameLog

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class TeamPitchingFeatures(FeatureExtractor):
    """Opponent team pitching stats for each game."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="opp_era",
                description="Opponent team ERA",
                source="matchup",
            ),
            FeatureMeta(
                name="opp_k_per_9",
                description="Opponent team strikeouts per 9 innings",
                source="matchup",
            ),
            FeatureMeta(
                name="opp_whip",
                description="Opponent team WHIP",
                source="matchup",
            ),
            FeatureMeta(
                name="opp_ba_against",
                description="Opponent team batting average against",
                source="matchup",
            ),
            FeatureMeta(
                name="opp_hr_per_9",
                description="Opponent team home runs allowed per 9 innings",
                source="matchup",
            ),
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        pitching: dict[int, dict[str, float]] | None = kwargs.get("opponent_pitching")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            opp_stats = (pitching or {}).get(log.opponent_id, {}) if pitching else {}
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "opp_era": opp_stats.get("era"),
                "opp_k_per_9": opp_stats.get("k_per_9"),
                "opp_whip": opp_stats.get("whip"),
                "opp_ba_against": opp_stats.get("ba_against"),
                "opp_hr_per_9": opp_stats.get("hr_per_9"),
            })
        return rows
