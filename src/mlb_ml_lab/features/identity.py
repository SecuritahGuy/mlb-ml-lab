from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class IdentityFeatures(FeatureExtractor):
    """Identity features extracted directly from game logs.

    These features are primarily useful as categorical inputs for CatBoost:
    batter's team, opponent, month, and position.  They pass through the
    standard numeric pipeline as integers (except ``position_code`` which
    is a string and is dropped by ``_feature_columns()``).
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="team_id",
                description="Batter's team MLBAM ID",
                source="identity",
            ),
            FeatureMeta(
                name="opponent_id",
                description="Opposing team MLBAM ID",
                source="identity",
            ),
            FeatureMeta(
                name="month",
                description="Game month (1–12)",
                source="identity",
            ),
            FeatureMeta(
                name="position_code",
                description="Position abbreviation (e.g. CF, SS, DH)",
                source="identity",
            ),
        ]

    def extract(
        self,
        game_logs: list[Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [
            {
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "team_id": log.team_id,
                "opponent_id": log.opponent_id,
                "month": int(log.date.split("-")[1]),
                "position_code": log.position_abbr,
            }
            for log in game_logs
        ]
