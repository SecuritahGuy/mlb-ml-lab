from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class ScheduleDensityFeatures(FeatureExtractor):
    """Opponent team schedule density — a fatigue proxy.

    Counts how many games the opponent team played in recent days
    before each game (not including the current game).  More games
    → more tired bullpen / potentially worse performance.

    Expects ``season_schedule`` kwarg (list of game dicts from
    ``MlbClient.get_season_schedule()``).
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self._windows = windows or [5, 10, 14]

    @property
    def features(self) -> list[FeatureMeta]:
        cols: list[FeatureMeta] = [
            FeatureMeta(
                name="opp_rest_days",
                description="Days since opponent's last game before this game",
                source="schedule",
            ),
        ]
        for w in self._windows:
            cols.append(
                FeatureMeta(
                    name=f"opp_games_last_{w}",
                    description=f"Games opponent played in last {w} days",
                    source="schedule",
                )
            )
        return cols

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        schedule: list[dict[str, Any]] | None = kwargs.get("season_schedule")
        if not schedule or not game_logs:
            return _empty_rows(game_logs)

        team_dates = _build_team_dates(schedule)

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            log_date = datetime.strptime(log.date, "%Y-%m-%d").date()
            opp_dates = team_dates.get(log.opponent_id, [])
            before = [d for d in opp_dates if d < log_date]

            rest: int | None = None
            if before:
                rest = (log_date - before[-1]).days

            row: dict[str, Any] = {
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "opp_rest_days": rest,
            }
            for w in self._windows:
                cutoff = log_date - timedelta(days=w)
                count = sum(1 for d in before if d >= cutoff)
                row[f"opp_games_last_{w}"] = count

            rows.append(row)

        return rows


def _build_team_dates(schedule: list[dict[str, Any]]) -> dict[int, list]:
    result: dict[int, list] = defaultdict(list)
    for game in schedule:
        status = game.get("status", {}) or {}
        state = status.get("detailedState", "")
        if state in ("Postponed", "Cancelled", "Scheduled"):
            continue
        official_date = game.get("officialDate", "")
        if not official_date:
            continue
        try:
            parsed = datetime.strptime(official_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        teams = game.get("teams", {})
        for side in ("away", "home"):
            team = (teams.get(side, {}) or {}).get("team", {}) or {}
            tid = team.get("id")
            if tid is not None:
                result[tid].append(parsed)

    for tid in result:
        result[tid] = sorted(set(result[tid]))
    return result


def _empty_rows(game_logs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": log.player_id,
            "game_pk": log.game_pk,
            "date": log.date,
            "opp_rest_days": None,
            "opp_games_last_5": None,
            "opp_games_last_10": None,
            "opp_games_last_14": None,
        }
        for log in game_logs
    ]
