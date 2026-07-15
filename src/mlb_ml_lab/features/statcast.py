"""Features derived from Statcast / Baseball Savant leaderboards.

These are season-level aggregates for each player — xBA, hard hit %,
barrel rate, etc.  Since they're per-season, we join them onto game
logs by (player_id, season).
"""

from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class StatcastAdvancedFeatures(FeatureExtractor):
    """Advanced hitting metrics from Statcast leaderboards."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            # From expected stats CSV
            FeatureMeta(
                name="ba", description="Actual batting average", source="statcast"
            ),
            FeatureMeta(
                name="slg", description="Actual slugging percentage", source="statcast"
            ),
            FeatureMeta(name="woba", description="Actual wOBA", source="statcast"),
            FeatureMeta(
                name="xba", description="Expected batting average", source="statcast"
            ),
            FeatureMeta(name="xwoba", description="Expected wOBA", source="statcast"),
            FeatureMeta(
                name="xslg", description="Expected slugging", source="statcast"
            ),
            # Luck differentials
            FeatureMeta(
                name="xba_diff", description="xBA minus actual BA", source="statcast"
            ),
            FeatureMeta(
                name="xslg_diff", description="xSLG minus actual SLG", source="statcast"
            ),
            FeatureMeta(
                name="xwoba_diff",
                description="xwOBA minus actual wOBA",
                source="statcast",
            ),
            # Batted ball quality from statcast leaderboard CSV
            FeatureMeta(
                name="hardhit_percent",
                description="Hard hit rate (95+ mph EV %)",
                source="statcast",
            ),
            FeatureMeta(
                name="barrels_per_bbe_percent",
                description="Barrels per BBE",
                source="statcast",
            ),
            FeatureMeta(
                name="brl_pa",
                description="Barrels per plate appearance",
                source="statcast",
            ),
            FeatureMeta(
                name="avg_hit_speed",
                description="Average exit velocity",
                source="statcast",
            ),
            FeatureMeta(
                name="max_hit_speed", description="Max exit velocity", source="statcast"
            ),
            FeatureMeta(
                name="ev50",
                description="50th percentile exit velocity",
                source="statcast",
            ),
            FeatureMeta(
                name="avg_launch_angle",
                description="Average launch angle",
                source="statcast",
            ),
            FeatureMeta(
                name="anglesweetspotpercent",
                description="Sweet-spot contact rate (8-32° launch angle)",
                source="statcast",
            ),
            FeatureMeta(
                name="fbld",
                description="Fly ball + line drive percentage",
                source="statcast",
            ),
            FeatureMeta(
                name="gb", description="Ground ball percentage", source="statcast"
            ),
            FeatureMeta(
                name="avg_distance",
                description="Average batted ball distance (ft)",
                source="statcast",
            ),
            FeatureMeta(
                name="max_distance",
                description="Max batted ball distance (ft)",
                source="statcast",
            ),
            FeatureMeta(
                name="avg_hr_distance",
                description="Average home run distance (ft)",
                source="statcast",
            ),
            FeatureMeta(
                name="ev95plus",
                description="Count of 95+ mph exit velocity",
                source="statcast",
            ),
            FeatureMeta(
                name="barrels", description="Count of barrels", source="statcast"
            ),
        ]

    def extract(
        self,
        game_logs: list[Any],
        statcast_batters: list[dict[str, str]] | None = None,
        expected_stats: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        sc_lookup: dict[str, dict[str, str]] = {}
        if statcast_batters:
            for row in statcast_batters:
                sc_lookup[row["player_id"]] = row

        es_lookup: dict[str, dict[str, str]] = {}
        if expected_stats:
            for row in expected_stats:
                es_lookup[row["player_id"]] = row

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            pid_str = str(log.player_id)
            sc = sc_lookup.get(pid_str, {})
            es = es_lookup.get(pid_str, {})

            ba = _float_or_none(es.get("ba"))
            slg = _float_or_none(es.get("slg"))
            woba = _float_or_none(es.get("woba"))
            xba = _float_or_none(es.get("est_ba"))
            xslg = _float_or_none(es.get("est_slg"))
            xwoba = _float_or_none(es.get("est_woba"))

            rows.append(
                {
                    "player_id": log.player_id,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    # From expected stats CSV
                    "ba": ba,
                    "slg": slg,
                    "woba": woba,
                    "xba": xba,
                    "xwoba": xwoba,
                    "xslg": xslg,
                    # Luck differentials
                    "xba_diff": (xba - ba)
                    if (xba is not None and ba is not None)
                    else None,
                    "xslg_diff": (xslg - slg)
                    if (xslg is not None and slg is not None)
                    else None,
                    "xwoba_diff": (xwoba - woba)
                    if (xwoba is not None and woba is not None)
                    else None,
                    # From statcast leaderboard CSV
                    "hardhit_percent": _float_or_none(sc.get("ev95percent")),
                    "barrels_per_bbe_percent": _float_or_none(sc.get("brl_percent")),
                    "brl_pa": _float_or_none(sc.get("brl_pa")),
                    "avg_hit_speed": _float_or_none(sc.get("avg_hit_speed")),
                    "max_hit_speed": _float_or_none(sc.get("max_hit_speed")),
                    "ev50": _float_or_none(sc.get("ev50")),
                    "avg_launch_angle": _float_or_none(sc.get("avg_hit_angle")),
                    "anglesweetspotpercent": _float_or_none(
                        sc.get("anglesweetspotpercent")
                    ),
                    "fbld": _float_or_none(sc.get("fbld")),
                    "gb": _float_or_none(sc.get("gb")),
                    "avg_distance": _float_or_none(sc.get("avg_distance")),
                    "max_distance": _float_or_none(sc.get("max_distance")),
                    "avg_hr_distance": _float_or_none(sc.get("avg_hr_distance")),
                    "ev95plus": _float_or_none(sc.get("ev95plus")),
                    "barrels": _float_or_none(sc.get("barrels")),
                }
            )
        return rows


def _float_or_none(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
