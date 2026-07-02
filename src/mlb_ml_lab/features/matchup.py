"""Opponent matchup features: team-level pitching stats faced by each batter.

Feature extractors:

- ``TeamPitchingFeatures`` — uses season-level stats from
  ``opponent_pitching`` kwarg (simple but has lookahead bias).
- ``RollingOpponentPitching`` — time-respecting, computes per-game
  opponent rolling stats from the game logs.
  No lookahead because it aggregates only games before the current
  game date.
- ``MonthlyTeamPitchingFeatures`` — uses month-level splits from
  ``monthly_pitching`` kwarg.  For each game, aggregates only months
  before the game date, reducing lookahead to at most one month.
- ``TeamDefenseFeatures`` — opponent team defensive quality (errors,
  fielding percentage, double plays) from ``team_fielding`` kwarg.
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


# ---------------------------------------------------------------------------
# Monthly Pitching (reduced lookahead)
# ---------------------------------------------------------------------------


@register
class MonthlyTeamPitchingFeatures(FeatureExtractor):
    """Month-level opponent pitching stats filtered to months before game date.

    Requires ``monthly_pitching`` in kwargs — dict mapping team_id → list
    of dicts with keys ``month`` and ``stat`` (from
    ``MlbClient.get_team_pitching_monthly_stats()``).
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(name="mth_opp_era",
                        description="Opponent ERA through month before game",
                        source="matchup"),
            FeatureMeta(name="mth_opp_k_per_9",
                        description="Opponent K/9 through month before game",
                        source="matchup"),
            FeatureMeta(name="mth_opp_whip",
                        description="Opponent WHIP through month before game",
                        source="matchup"),
            FeatureMeta(name="mth_opp_ba_against",
                        description="Opponent BAA through month before game",
                        source="matchup"),
            FeatureMeta(name="mth_opp_games",
                        description="Monthly splits aggregated",
                        source="matchup"),
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        monthly: dict[int, list[dict[str, Any]]] | None = kwargs.get("monthly_pitching")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            game_month = _month_from_date(log.date)
            opp_months = (monthly or {}).get(log.opponent_id, []) if monthly else []

            total_ip = 0.0
            total_er = 0
            total_so = 0
            total_bb = 0
            total_h = 0
            count = 0
            for m in opp_months:
                if m["month"] < game_month:
                    s = m["stat"]
                    ip = _float_or_none(s.get("inningsPitched"))
                    if ip is not None:
                        total_ip += ip
                        total_er += int(s.get("earnedRuns", 0))
                        total_so += int(s.get("strikeOuts", 0))
                        total_bb += int(s.get("baseOnBalls", 0))
                        total_h += int(s.get("hits", 0))
                        count += 1

            era: float | None = None
            k9: float | None = None
            whip: float | None = None
            baa: float | None = None
            if total_ip > 0:
                era = round(total_er * 9 / total_ip, 2)
                k9 = round(total_so * 9 / total_ip, 2)
                whip = round((total_bb + total_h) / total_ip, 3)
                estimated_bf = total_h + total_so + total_bb + int(total_ip * 3)
                if estimated_bf > 0:
                    baa = _round3(total_h / estimated_bf)

            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "mth_opp_era": era,
                "mth_opp_k_per_9": k9,
                "mth_opp_whip": whip,
                "mth_opp_ba_against": baa,
                "mth_opp_games": count,
            })
        return rows


# ---------------------------------------------------------------------------
# Team Defense
# ---------------------------------------------------------------------------


@register
class TeamDefenseFeatures(FeatureExtractor):
    """Opponent team defensive quality features.

    Requires ``team_fielding`` in kwargs — dict mapping team_id → stat
    dict from ``MlbClient.get_team_fielding_stats()``.
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(name="opp_fielding_pct",
                        description="Opponent team fielding percentage",
                        source="matchup"),
            FeatureMeta(name="opp_errors",
                        description="Opponent team total errors",
                        source="matchup"),
            FeatureMeta(name="opp_double_plays",
                        description="Opponent team double plays turned",
                        source="matchup"),
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        fielding: dict[int, dict[str, Any]] | None = kwargs.get("team_fielding")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            fd = (fielding or {}).get(log.opponent_id, {}) if fielding else {}
            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "opp_fielding_pct": _float_or_none(fd.get("fieldingPct")),
                "opp_errors": _float_or_none(fd.get("errors")),
                "opp_double_plays": _float_or_none(fd.get("doublePlays")),
            })
        return rows


def _month_from_date(date_str: str) -> int:
    try:
        return int(date_str.split("-")[1])
    except (ValueError, IndexError):
        return 99


def _round3(val: float) -> float:
    return round(val, 3)


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
