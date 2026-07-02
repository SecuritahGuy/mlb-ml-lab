"""Opposing starting pitcher quality and platoon advantage features.

Requires ``game_contexts`` (with ``home_probable_pitcher_id`` and
``away_probable_pitcher_id``), ``pitcher_stats`` (dict of pitcher_id →
season stat dict), and ``player_details`` (dict of player_id →
``get_player()`` result).

Pitcher stats are expected to be previous-season or season-to-date to
avoid lookahead bias.
"""

from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class StartingPitcherFeatures(FeatureExtractor):
    """Opposing starting pitcher quality and platoon matchup features."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(name="opp_pitcher_era",
                        description="Opposing starter ERA (prev season or season-to-date)",
                        source="pitching"),
            FeatureMeta(name="opp_pitcher_k_per_9",
                        description="Opposing starter K/9", source="pitching"),
            FeatureMeta(name="opp_pitcher_whip",
                        description="Opposing starter WHIP", source="pitching"),
            FeatureMeta(name="opp_pitcher_ba_against",
                        description="Opposing starter BAA", source="pitching"),
            FeatureMeta(name="opp_pitcher_hr_per_9",
                        description="Opposing starter HR/9", source="pitching"),
            FeatureMeta(name="opp_pitcher_k_rate",
                        description="Opposing starter K per PA", source="pitching"),
            FeatureMeta(name="same_hand_advantage",
                        description="1 if batter and pitcher share handedness (pitcher advantage)",
                        source="pitching"),
        ]

    def extract(self, game_logs: list[Any], **kwargs: Any) -> list[dict[str, Any]]:
        contexts: dict[int, dict[str, Any]] | None = kwargs.get("game_contexts")
        pitcher_stats: dict[int, dict[str, Any]] | None = kwargs.get("pitcher_stats")
        player_details: dict[int, dict[str, Any]] | None = kwargs.get("player_details")

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            ctx = (contexts or {}).get(log.game_pk, {}) if contexts else {}

            # Determine opposing pitcher: pitcher for the OTHER team
            if log.is_home:
                opp_pitcher_id = ctx.get("away_probable_pitcher_id")
            else:
                opp_pitcher_id = ctx.get("home_probable_pitcher_id")

            pstats: dict[str, Any] = {}
            if pitcher_stats and opp_pitcher_id:
                pstats = pitcher_stats.get(opp_pitcher_id, {})
            pdetail = None
            if player_details and opp_pitcher_id:
                pdetail = player_details.get(opp_pitcher_id)

            # Platoon advantage: same handedness = pitcher advantage
            batter_detail = (player_details or {}).get(log.player_id, {}) if player_details else {}
            platoon = _compute_platoon(batter_detail, pdetail)

            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "opp_pitcher_era": _float_or_none(pstats.get("era")),
                "opp_pitcher_k_per_9": _float_or_none(pstats.get("strikeoutsPer9Inn")),
                "opp_pitcher_whip": _float_or_none(pstats.get("whip")),
                "opp_pitcher_ba_against": _str_avg_or_none(pstats.get("avg")),
                "opp_pitcher_hr_per_9": _float_or_none(pstats.get("homeRunsPer9")),
                "opp_pitcher_k_rate": _compute_k_rate(pstats),
                "same_hand_advantage": platoon,
            })
        return rows


def _compute_platoon(
    batter_detail: dict[str, Any],
    pitcher_detail: dict[str, Any] | None,
) -> int | None:
    if not pitcher_detail:
        return None
    bats = (batter_detail.get("batSide", {}) or {}).get("code", "")
    throws = (pitcher_detail.get("pitchHand", {}) or {}).get("code", "")
    if not bats or not throws:
        return None
    if bats == "S":
        return 0  # switch-hitter neutralizes platoon
    return 1 if bats == throws else 0


def _compute_k_rate(stats: dict[str, Any]) -> float | None:
    so = _float_or_none(stats.get("strikeOuts"))
    bf = _float_or_none(stats.get("battersFaced"))
    if so is not None and bf is not None and bf > 0:
        return round(so / bf, 3)
    return None


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _str_avg_or_none(val: str | None) -> float | None:
    if val is None or val in ("", ".---", "----"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
