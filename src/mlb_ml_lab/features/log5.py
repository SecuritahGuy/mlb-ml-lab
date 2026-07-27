from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


def _log5(p1: float, p2: float) -> float:
    """Log5 probability of event with rate *p1* vs opponent with rate *p2*."""
    num = p1 - p1 * p2
    den = p1 + p2 - 2 * p1 * p2
    return num / den if den > 0 else 0.5


def _rolling_rates(
    logs: list[Any],
    group_key: str,
    window: int,
) -> dict[tuple[int, str | int], float]:
    """Precompute rolling hit rates going into each (player_id, game_pk).

    For each game, the rate is computed from the *window* most recent
    preceding games for the same *group_key* (player_id or team_id).
    """
    by_group: dict[int, list[Any]] = defaultdict(list)
    for log in logs:
        gid = getattr(log, group_key)
        by_group[gid].append(log)

    rates: dict[tuple[int, int], float] = {}
    for gid, entries in by_group.items():
        entries.sort(key=lambda e: e.date)
        for i, log in enumerate(entries):
            start = max(0, i - window)
            chunk = entries[start:i]
            if not chunk:
                rate = 0.5
            else:
                total_pa = sum(e.plate_appearances for e in chunk)
                total_h = sum(e.hits for e in chunk)
                rate = total_h / total_pa if total_pa > 0 else 0.5
            rates[(gid, log.game_pk)] = rate
    return rates


def _opponent_team_rates(
    logs: list[Any],
    window: int,
) -> dict[tuple[int, int], float]:
    """Precompute opponent team hit rate allowed going into each game.

    For each (team_id, game_pk), the rate is the opponents' hit rate
    (hits / PA) against this team in the *window* most recent preceding
    games.
    """
    by_team: dict[int, list[Any]] = defaultdict(list)
    for log in logs:
        by_team[log.opponent_id].append(log)

    rates: dict[tuple[int, int], float] = {}
    for tid, entries in by_team.items():
        entries.sort(key=lambda e: e.date)
        for i, log in enumerate(entries):
            start = max(0, i - window)
            chunk = entries[start:i]
            if not chunk:
                rate = 0.5
            else:
                total_pa = sum(e.plate_appearances for e in chunk)
                total_h = sum(e.hits for e in chunk)
                rate = total_h / total_pa if total_pa > 0 else 0.5
            rates[(tid, log.game_pk)] = rate
    return rates


def _league_hit_rate(logs: list[Any]) -> float:
    """Compute overall league hit rate (hits / PA) from all game logs."""
    total_pa = sum(e.plate_appearances for e in logs)
    total_h = sum(e.hits for e in logs)
    return total_h / total_pa if total_pa > 0 else 0.5


@register
class Log5Features(FeatureExtractor):
    """Log5 matchup probabilities using rolling hit rates.

    Uses the Bill James Log5 formula to estimate the probability a batter
    gets a hit against a given opponent, based on their respective rolling
    hit rates and the league average.
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="log5_hit_prob_5",
                description="Log5 hit prob (5-game rolling batter vs opponent)",
                source="log5",
            ),
            FeatureMeta(
                name="log5_hit_prob_10",
                description="Log5 hit prob (10-game rolling batter vs opponent)",
                source="log5",
            ),
            FeatureMeta(
                name="log5_hit_prob_20",
                description="Log5 hit prob (20-game rolling batter vs opponent)",
                source="log5",
            ),
            FeatureMeta(
                name="log5_vs_league",
                description="Log5 hit prob (batter vs league-average pitcher)",
                source="log5",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        if not game_logs:
            return []

        league_rate = _league_hit_rate(game_logs)

        batter_rates_5 = _rolling_rates(game_logs, "player_id", 5)
        batter_rates_10 = _rolling_rates(game_logs, "player_id", 10)
        batter_rates_20 = _rolling_rates(game_logs, "player_id", 20)
        opp_rates_5 = _opponent_team_rates(game_logs, 5)
        opp_rates_10 = _opponent_team_rates(game_logs, 10)
        opp_rates_20 = _opponent_team_rates(game_logs, 20)

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            key = (log.player_id, log.game_pk)
            opp_key = (log.opponent_id, log.game_pk)

            b_rate_5 = batter_rates_5.get(key, 0.5)
            b_rate_10 = batter_rates_10.get(key, 0.5)
            b_rate_20 = batter_rates_20.get(key, 0.5)
            o_rate_5 = opp_rates_5.get(opp_key, 0.5)
            o_rate_10 = opp_rates_10.get(opp_key, 0.5)
            o_rate_20 = opp_rates_20.get(opp_key, 0.5)

            rows.append(
                {
                    "player_id": log.player_id,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    "log5_hit_prob_5": round(_log5(b_rate_5, o_rate_5), 4),
                    "log5_hit_prob_10": round(_log5(b_rate_10, o_rate_10), 4),
                    "log5_hit_prob_20": round(_log5(b_rate_20, o_rate_20), 4),
                    "log5_vs_league": round(_log5(b_rate_20, league_rate), 4),
                }
            )
        return rows
