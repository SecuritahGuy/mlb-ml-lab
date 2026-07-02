"""Opponent matchup features: team-level pitching stats faced by each batter.

Two feature extractors:

- ``TeamPitchingFeatures`` — uses season-level stats from
  ``opponent_pitching`` kwarg (simple but has lookahead bias).
- ``RollingOpponentPitching`` — time-respecting, computes per-game
  opponent rolling stats from the game logs passed to ``extract()``.
  No lookahead because it aggregates only games before the current
  game date.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mlb_ml_lab.data.schemas import PlayerGameLog

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class TeamPitchingFeatures(FeatureExtractor):
    """Opponent team season-level pitching stats for each game.

    Uses pre-computed ``opponent_pitching`` dict from kwargs.  Falls
    back to None when no data provided.
    """

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


@register
class RollingOpponentPitching(FeatureExtractor):
    """Time-respecting opponent pitching stats computed from game logs.

    For each game, aggregates stats from all preceding games (across all
    tracked players) against the same opponent.  No lookahead — only
    games before the current game's date are included in the rolling
    window.
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="rolling_opp_k_rate",
                description="Opponent K-per-PA rate from games before this date",
                source="matchup",
            ),
            FeatureMeta(
                name="rolling_opp_ba_against",
                description="Opponent BAA from games before this date",
                source="matchup",
            ),
            FeatureMeta(
                name="rolling_opp_walk_rate",
                description="Opponent BB-per-PA rate from games before this date",
                source="matchup",
            ),
            FeatureMeta(
                name="rolling_opp_sample_games",
                description="Number of tracked games against this opponent before this date",
                source="matchup",
            ),
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        # Build game records grouped by opponent
        games_by_opp: dict[int, list[_GameRecord]] = defaultdict(list)
        for log in game_logs:
            games_by_opp[log.opponent_id].append(
                _GameRecord(
                    date=log.date,
                    hits=log.hits,
                    at_bats=log.at_bats,
                    walks=log.walks,
                    strikeouts=log.strikeouts,
                    plate_appearances=log.plate_appearances,
                )
            )

        # Sort each opponent's games by date
        for opp_id in games_by_opp:
            games_by_opp[opp_id].sort(key=lambda g: g.date)

        # Build rolling aggregates
        opp_rolling: dict[int, list[_RollingPoint]] = {}
        for opp_id, games in games_by_opp.items():
            cum_k = 0
            cum_bb = 0
            cum_h = 0
            cum_pa = 0
            points: list[_RollingPoint] = []
            for i, g in enumerate(games):
                if i > 0 and cum_pa > 0:
                    points.append(_RollingPoint(
                        date=g.date,
                        k_rate=cum_k / cum_pa,
                        ba_against=cum_h / cum_pa,
                        walk_rate=cum_bb / cum_pa,
                        sample_games=i,
                    ))
                else:
                    points.append(_RollingPoint(
                        date=g.date, k_rate=None,
                        ba_against=None, walk_rate=None,
                        sample_games=0,
                    ))
                cum_k += g.strikeouts
                cum_bb += g.walks
                cum_h += g.hits
                cum_pa += g.plate_appearances
            opp_rolling[opp_id] = points

        rows: list[dict[str, Any]] = []
        # Track next index to consume for each opponent
        next_idx: dict[int, int] = defaultdict(int)

        for log in game_logs:
            points = opp_rolling.get(log.opponent_id, [])
            idx = next_idx.get(log.opponent_id, 0)
            point = points[idx] if idx < len(points) else _RollingPoint()
            next_idx[log.opponent_id] = idx + 1

            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "rolling_opp_k_rate": point.k_rate,
                "rolling_opp_ba_against": point.ba_against,
                "rolling_opp_walk_rate": point.walk_rate,
                "rolling_opp_sample_games": point.sample_games,
            })
        return rows


class _GameRecord:
    __slots__ = ("date", "hits", "at_bats", "walks", "strikeouts", "plate_appearances")

    def __init__(
        self, date: str, hits: int, at_bats: int,
        walks: int, strikeouts: int, plate_appearances: int,
    ) -> None:
        self.date = date
        self.hits = hits
        self.at_bats = at_bats
        self.walks = walks
        self.strikeouts = strikeouts
        self.plate_appearances = plate_appearances


class _RollingPoint:
    __slots__ = ("date", "k_rate", "ba_against", "walk_rate", "sample_games")

    def __init__(
        self, date: str = "", k_rate: float | None = None,
        ba_against: float | None = None,
        walk_rate: float | None = None, sample_games: int = 0,
    ) -> None:
        self.date = date
        self.k_rate = k_rate
        self.ba_against = ba_against
        self.walk_rate = walk_rate
        self.sample_games = sample_games
