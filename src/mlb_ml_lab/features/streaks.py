from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class StreaksFeature(FeatureExtractor):
    """Player hot/cold streaks (hitting, on-base).

    When ``streaks_stats`` kwarg is provided (dict mapping player_id →
    ``{"hitting": int, "onbase": int}``), uses those values directly
    (backward-compatible with the old MLB Stats API endpoint).

    When ``streaks_stats`` is absent/empty, computes streaks from the
    game_logs themselves: for each player, sorts games by date and
    counts consecutive games with a hit (hitting streak) or with
    hits+walks > 0 (on-base streak) going into each game.
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

        if streaks:
            return _from_api_streaks(game_logs, streaks)

        if game_logs:
            return _compute_from_logs(game_logs)

        return _empty_rows(game_logs)


def _from_api_streaks(
    game_logs: list[Any],
    streaks: dict[int, dict[str, int]],
) -> list[dict[str, Any]]:
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


def _compute_from_logs(game_logs: list[Any]) -> list[dict[str, Any]]:
    by_player: dict[int, list[Any]] = defaultdict(list)
    for log in game_logs:
        by_player[log.player_id].append(log)

    per_game: dict[tuple[int, int], dict[str, int]] = {}

    for pid, logs in by_player.items():
        sorted_logs = sorted(logs, key=lambda log: log.date)
        hitting_streak = 0
        onbase_streak = 0

        for log in sorted_logs:
            per_game[(pid, log.game_pk)] = {
                "hitting": hitting_streak,
                "onbase": onbase_streak,
            }
            if log.hits > 0:
                hitting_streak += 1
            else:
                hitting_streak = 0

            if log.hits + log.walks > 0:
                onbase_streak += 1
            else:
                onbase_streak = 0

    rows: list[dict[str, Any]] = []
    for log in game_logs:
        s = per_game.get((log.player_id, log.game_pk), {})
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
