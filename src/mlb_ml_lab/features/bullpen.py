from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class BullpenQualityFeatures(FeatureExtractor):
    """Opponent bullpen (reliever-only) quality features.

    Bullpen stats are computed from individual pitcher season stats,
    filtered to relievers (``gamesStarted / gamesPlayed < 0.5``).

    Expects ``bullpen_stats`` kwarg — a dict mapping team_id → dict::

        {
            "era": 3.87,
            "k_per_9": 9.2,
            "whip": 1.28,
            "ba_against": 0.242,
            "hr_per_9": 1.1,
        }

    from ``MlbClient.get_team_bullpen_stats()``.
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="bullpen_era",
                description="Opponent bullpen ERA",
                source="bullpen",
            ),
            FeatureMeta(
                name="bullpen_k_per_9",
                description="Opponent bullpen strikeouts per 9 innings",
                source="bullpen",
            ),
            FeatureMeta(
                name="bullpen_whip",
                description="Opponent bullpen WHIP",
                source="bullpen",
            ),
            FeatureMeta(
                name="bullpen_ba_against",
                description="Opponent bullpen batting average against",
                source="bullpen",
            ),
            FeatureMeta(
                name="bullpen_hr_per_9",
                description="Opponent bullpen home runs per 9 innings",
                source="bullpen",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        bullpen: dict[int, dict[str, float]] | None = kwargs.get("bullpen_stats")
        if not bullpen or not game_logs:
            return _empty_rows(game_logs)

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            bp = bullpen.get(log.opponent_id, {})
            rows.append(
                {
                    "player_id": log.player_id,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    "bullpen_era": bp.get("era"),
                    "bullpen_k_per_9": bp.get("k_per_9"),
                    "bullpen_whip": bp.get("whip"),
                    "bullpen_ba_against": bp.get("ba_against"),
                    "bullpen_hr_per_9": bp.get("hr_per_9"),
                }
            )

        return rows


def _empty_rows(game_logs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": log.player_id,
            "game_pk": log.game_pk,
            "date": log.date,
            "bullpen_era": None,
            "bullpen_k_per_9": None,
            "bullpen_whip": None,
            "bullpen_ba_against": None,
            "bullpen_hr_per_9": None,
        }
        for log in game_logs
    ]
