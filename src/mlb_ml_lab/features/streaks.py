from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class StreaksFeature(FeatureExtractor):
    """Player hot/cold streaks (hitting, on-base).

    Uses the MLB Stats API ``/stats/streaks`` endpoint to provide the
    length of each player's current hitting and on-base streaks.

    Expects ``streaks_stats`` kwarg — a dict mapping player_id → dict::

        {
            "hitting": 8,      # games with a hit
            "onbase": 12,      # games reaching base
        }
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="hitting_streak",
                description="Current hitting streak length (consecutive games with a hit)",
                source="streaks",
            ),
            FeatureMeta(
                name="onbase_streak",
                description="Current on-base streak length (consecutive games reaching base)",
                source="streaks",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        streaks: dict[int, dict[str, int]] | None = kwargs.get("streaks_stats")
        if not streaks or not game_logs:
            return _empty_rows(game_logs)

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            s = streaks.get(log.player_id, {})
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "hitting_streak": s.get("hitting"),
                "onbase_streak": s.get("onbase"),
            })

        return rows


def _empty_rows(game_logs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": log.player_id,
            "game_pk": log.game_pk,
            "date": log.date,
            "hitting_streak": None,
            "onbase_streak": None,
        }
        for log in game_logs
    ]
