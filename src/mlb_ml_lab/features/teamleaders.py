from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class TeamLeadersFeature(FeatureExtractor):
    """Opponent team statistical leaders.

    Provides the opponent team's top hitter batting average, home runs,
    and RBI as a proxy for opponent offensive quality.  Uses the MLB
    Stats API ``/teams/{id}/leaders`` endpoint.

    Expects ``team_leaders`` kwarg — a dict mapping team_id → dict::

        {
            "top_avg": 0.320,
            "top_hr": 25,
            "top_rbi": 80,
        }
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="opp_top_avg",
                description="Opponent team top hitter batting average",
                source="team_leaders",
            ),
            FeatureMeta(
                name="opp_top_hr",
                description="Opponent team top hitter home runs",
                source="team_leaders",
            ),
            FeatureMeta(
                name="opp_top_rbi",
                description="Opponent team top hitter runs batted in",
                source="team_leaders",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        leaders: dict[int, dict[str, float]] | None = kwargs.get("team_leaders")
        if not leaders or not game_logs:
            return _empty_rows(game_logs)

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            ld = leaders.get(log.opponent_id, {})
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "opp_top_avg": ld.get("top_avg"),
                "opp_top_hr": ld.get("top_hr"),
                "opp_top_rbi": ld.get("top_rbi"),
            })

        return rows


def _empty_rows(game_logs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": log.player_id,
            "game_pk": log.game_pk,
            "date": log.date,
            "opp_top_avg": None,
            "opp_top_hr": None,
            "opp_top_rbi": None,
        }
        for log in game_logs
    ]
