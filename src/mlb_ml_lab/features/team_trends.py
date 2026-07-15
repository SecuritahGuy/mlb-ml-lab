"""Team-level recent performance features.

Aggregates per-player game logs to team-game level, then computes
rolling windows (last 5/10 games) of team hitting output.  This
captures team-level momentum that individual player features miss.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register
from mlb_ml_lab.data.schemas import PlayerGameLog


@register
class TeamTrendFeatures(FeatureExtractor):
    """Rolling team-level hit totals over recent games."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="team_hits_last_5",
                description="Average team hits per game over last 5 team games",
                source="game_log",
            ),
            FeatureMeta(
                name="team_hits_last_10",
                description="Average team hits per game over last 10 team games",
                source="game_log",
            ),
            FeatureMeta(
                name="team_opp_hits_last_5",
                description="Average hits allowed by opponent over last 5 games",
                source="game_log",
            ),
            FeatureMeta(
                name="team_opp_hits_last_10",
                description="Average hits allowed by opponent over last 10 games",
                source="game_log",
            ),
        ]

    def extract(
        self,
        game_logs: list[PlayerGameLog],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Pre-compute team-game aggregates
        # (team_id, game_pk) → {team_hits, opp_hits, date}
        team_games: dict[tuple[int, int], dict[str, Any]] = {}
        for log in game_logs:
            key = (log.team_id, log.game_pk)
            if key not in team_games:
                team_games[key] = {
                    "team_id": log.team_id,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    "team_hits": 0,
                    "opponent_id": log.opponent_id,
                    "opp_hits": 0,
                }
            team_games[key]["team_hits"] += log.hits

        # Build opponent-team hits index
        # (opponent_id, game_pk) → opp_hits (same as team_hits of opposing team)
        opp_game_hits: dict[tuple[int, int], int] = {}
        for tg in team_games.values():
            opp_key = (tg["opponent_id"], tg["game_pk"])
            opp_game_hits[opp_key] = opp_game_hits.get(opp_key, 0) + tg["team_hits"]

        # Group by team_id, sorted by date
        team_histories: dict[int, list[dict]] = defaultdict(list)
        for tg in team_games.values():
            team_histories[tg["team_id"]].append(tg)
        for tid in team_histories:
            team_histories[tid].sort(key=lambda x: x["date"])

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            key = (log.team_id, log.game_pk)

            # Rolling team hits
            history = team_histories.get(log.team_id, [])
            before = [g for g in history if g["date"] < log.date]
            hits_l5 = _avg_hits(before, 5)
            hits_l10 = _avg_hits(before, 10)

            # Rolling opponent hits allowed (same team, from opponent's perspective)
            opp_before = [g for g in history if g["date"] < log.date]
            opp_hits_l5 = _avg_opp_hits(opp_before, opp_game_hits, 5)
            opp_hits_l10 = _avg_opp_hits(opp_before, opp_game_hits, 10)

            rows.append(
                {
                    "player_id": log.player_id,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    "team_hits_last_5": hits_l5,
                    "team_hits_last_10": hits_l10,
                    "team_opp_hits_last_5": opp_hits_l5,
                    "team_opp_hits_last_10": opp_hits_l10,
                }
            )

        return rows


def _avg_hits(games: list[dict], n: int) -> float | None:
    recent = games[-n:] if len(games) >= n else games
    if not recent:
        return None
    return round(sum(g["team_hits"] for g in recent) / len(recent), 2)


def _avg_opp_hits(
    team_games: list[dict],
    opp_game_hits: dict[tuple[int, int], int],
    n: int,
) -> float | None:
    recent = team_games[-n:] if len(team_games) >= n else team_games
    if not recent:
        return None
    total = 0
    count = 0
    for g in recent:
        opp_key = (g["opponent_id"], g["game_pk"])
        oh = opp_game_hits.get(opp_key)
        if oh is not None:
            total += oh
            count += 1
    if count == 0:
        return None
    return round(total / count, 2)
