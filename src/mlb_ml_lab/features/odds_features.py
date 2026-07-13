"""Features derived from SBR moneyline odds.

These are game-level market signals: the team's moneyline and implied win
probability, plus the opponent's.  Same value for all players in a game.
"""

from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


def ml_to_implied_prob(ml: int | None) -> float | None:
    """Convert American moneyline odds to implied probability."""
    if ml is None or abs(ml) >= 10000:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 100.0 / (ml + 100.0)


@register
class OddsFeatures(FeatureExtractor):
    """Moneyline odds from SBR for each game."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(name="team_ml", description="Team's moneyline odds", source="odds"),
            FeatureMeta(name="opp_ml", description="Opponent's moneyline odds", source="odds"),
            FeatureMeta(name="team_implied_prob", description="Team's implied win pct from moneyline", source="odds"),
            FeatureMeta(name="opp_implied_prob", description="Opponent's implied win probability", source="odds"),
        ]

    def extract(
        self,
        game_logs: list[Any],
        odds_by_game: dict[tuple[int, int], dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if odds_by_game is None:
            odds_by_game = {}

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            odds = odds_by_game.get((log.team_id, log.game_pk), {})
            team_ml = odds.get("team_ml")
            opp_ml = odds.get("opp_ml")

            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "team_ml": team_ml,
                "opp_ml": opp_ml,
                "team_implied_prob": ml_to_implied_prob(team_ml),
                "opp_implied_prob": ml_to_implied_prob(opp_ml),
            })
        return rows
