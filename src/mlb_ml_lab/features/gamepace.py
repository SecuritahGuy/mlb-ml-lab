from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class GamePaceFeature(FeatureExtractor):
    """Opponent team pace-of-game features.

    Faster-paced games mean more total plate appearances for batters,
    which increases hit opportunities.  Uses the MLB Stats API
    ``/gamePace`` endpoint.

    Expects ``game_pace_stats`` kwarg — a dict mapping team_id → dict::

        {
            "time_per_game": 185.0,        # minutes
            "pitches_per_game": 290.0,     # average pitches per game
        }
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="opp_pace_time_per_game",
                description="Opponent team average game duration (minutes)",
                source="game_pace",
            ),
            FeatureMeta(
                name="opp_pace_pitches_per_game",
                description="Opponent team average pitches per game",
                source="game_pace",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        pace: dict[int, dict[str, float]] | None = kwargs.get("game_pace_stats")
        if not pace or not game_logs:
            return _empty_rows(game_logs)

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            p = pace.get(log.opponent_id, {})
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "opp_pace_time_per_game": p.get("time_per_game"),
                "opp_pace_pitches_per_game": p.get("pitches_per_game"),
            })

        return rows


def _empty_rows(game_logs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": log.player_id,
            "game_pk": log.game_pk,
            "date": log.date,
            "opp_pace_time_per_game": None,
            "opp_pace_pitches_per_game": None,
        }
        for log in game_logs
    ]
