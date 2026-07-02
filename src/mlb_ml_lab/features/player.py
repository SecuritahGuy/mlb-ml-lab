"""Player-level baseline quality features.

Uses pre-season or career data only — no within-season stats that would
introduce lookahead bias.

Requires ``player_details`` (dict of player_id → ``get_player()`` result)
and/or ``career_stats`` (dict of player_id → dict with weighted career
averages for avg/obp/slg/ops/hr) in kwargs.  When absent, features
default to None/0.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from mlb_ml_lab.data.schemas import PlayerGameLog

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class PlayerQualityFeatures(FeatureExtractor):
    """Player-level baseline quality: age, bats/throws, experience, prev-season stats."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(name="player_age", description="Player age at game date (years)",
                        source="player"),
            FeatureMeta(name="years_experience",
                        description="Years since MLB debut at game date",
                        source="player"),
            FeatureMeta(name="bats_right", description="1 if bats right-handed",
                        source="player"),
            FeatureMeta(name="bats_left", description="1 if bats left-handed",
                        source="player"),
            FeatureMeta(name="throws_right", description="1 if throws right-handed",
                        source="player"),
            FeatureMeta(name="throws_left", description="1 if throws left-handed",
                        source="player"),
            FeatureMeta(name="position_cat", description="0=IF, 1=OF, 2=C, 3=DH",
                        source="player"),
            FeatureMeta(name="career_avg",
                        description="Weighted career batting average (last 3 seasons)",
                        source="player"),
            FeatureMeta(name="career_obp",
                        description="Weighted career on-base percentage",
                        source="player"),
            FeatureMeta(name="career_slg",
                        description="Weighted career slugging percentage",
                        source="player"),
            FeatureMeta(name="career_ops",
                        description="Weighted career OPS", source="player"),
            FeatureMeta(name="career_hr",
                        description="Weighted career home runs", source="player"),
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        player_details: dict[int, dict[str, Any]] | None = kwargs.get("player_details")
        career_stats: dict[int, dict[str, Any]] | None = kwargs.get("career_stats")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            pid = log.player_id
            detail = (player_details or {}).get(pid, {}) if player_details else {}
            stats = (career_stats or {}).get(pid, {}) if career_stats else {}

            game_dt = _parse_date(log.date)

            rows.append({
                "player_id": pid,
                "game_pk": log.game_pk,
                "date": log.date,
                "player_age": _compute_age(detail.get("birthDate", ""), game_dt),
                "years_experience": _compute_experience(
                    detail.get("mlbDebutDate", ""), game_dt,
                ),
                "bats_right": _bats_code(detail, "R"),
                "bats_left": _bats_code(detail, "L"),
                "throws_right": _throws_code(detail, "R"),
                "throws_left": _throws_code(detail, "L"),
                "position_cat": _position_category(detail),
                "career_avg": _str_avg_or_none(stats.get("avg")),
                "career_obp": _str_avg_or_none(stats.get("obp")),
                "career_slg": _str_avg_or_none(stats.get("slg")),
                "career_ops": _float_or_none(stats.get("ops")),
                "career_hr": _float_or_none(stats.get("homeRuns")),
            })
        return rows


def _parse_date(s: str) -> date | None:
    try:
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def _compute_age(birth_date_str: str, game_dt: date | None) -> float | None:
    if not birth_date_str or game_dt is None:
        return None
    try:
        bd = date.fromisoformat(birth_date_str)
        return round((game_dt - bd).days / 365.25, 1)
    except (ValueError, TypeError):
        return None


def _compute_experience(debut_str: str, game_dt: date | None) -> float | None:
    if not debut_str or game_dt is None:
        return None
    try:
        dd = date.fromisoformat(debut_str)
        years = (game_dt - dd).days / 365.25
        return round(years, 1)
    except (ValueError, TypeError):
        return None


def _bats_code(detail: dict[str, Any], side: str) -> int:
    bats = detail.get("batSide", {}) or {}
    return 1 if bats.get("code") == side else 0


def _throws_code(detail: dict[str, Any], side: str) -> int:
    throws = detail.get("pitchHand", {}) or {}
    return 1 if throws.get("code") == side else 0


def _position_category(detail: dict[str, Any]) -> int | None:
    pos = detail.get("primaryPosition", {}) or {}
    abbr = pos.get("abbreviation", "")
    mapping = {
        "1B": 0, "2B": 0, "3B": 0, "SS": 0,
        "LF": 1, "CF": 1, "RF": 1, "OF": 1,
        "C": 2,
        "DH": 3,
    }
    return mapping.get(abbr)


def _str_avg_or_none(val: str | None) -> float | None:
    if val is None or val in ("", ".---", "----"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
