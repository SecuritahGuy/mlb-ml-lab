"""Features derived from Statcast / Baseball Savant leaderboards.

These are season-level aggregates for each player — xBA, hard hit %,
barrel rate, etc.  Since they're per-season, we join them onto game
logs by (player_id, season).
"""

from __future__ import annotations

from typing import Any

from pipeline.base import FeatureExtractor, FeatureMeta, register


@register
class StatcastAdvancedFeatures(FeatureExtractor):
    """Advanced hitting metrics from Statcast leaderboards."""

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(name="xba", description="Expected batting average", source="statcast"),
            FeatureMeta(name="xwoba", description="Expected wOBA", source="statcast"),
            FeatureMeta(name="xslg", description="Expected slugging", source="statcast"),
            FeatureMeta(name="hardhit_percent", description="Hard hit rate", source="statcast"),
            FeatureMeta(
                name="barrels_per_bbe_percent",
                description="Barrels per BBE",
                source="statcast",
            ),
            FeatureMeta(name="avg_hit_speed", description="Average exit velocity", source="statcast"),
            FeatureMeta(name="avg_launch_angle", description="Average launch angle", source="statcast"),
            FeatureMeta(name="k_percent", description="Strikeout rate", source="statcast"),
            FeatureMeta(name="bb_percent", description="Walk rate", source="statcast"),
            FeatureMeta(name="babip", description="BABIP", source="statcast"),
        ]

    def extract(
        self,
        game_logs: list[Any],
        statcast_batters: list[dict[str, str]] | None = None,
        expected_stats: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Merge Savant CSV rows into game-log rows by player_id.

        Args:
            game_logs: Player game logs (used to determine which players/seasons
                       to include).
            statcast_batters: Rows from get_statcast_batters() CSV.
            expected_stats: Rows from get_expected_stats() CSV.

        Returns:
            List of feature dicts keyed by (player_id, game_pk, date).
        """
        # Build lookup: player_id -> statcast row
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

            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                # From expected stats CSV
                "xba": _float_or_none(es.get("est_ba")),
                "xwoba": _float_or_none(es.get("est_woba")),
                "xslg": _float_or_none(es.get("est_slg")),
                # From statcast leaderboard CSV
                "hardhit_percent": _float_or_none(sc.get("ev95percent")),
                "barrels_per_bbe_percent": _float_or_none(sc.get("brl_percent")),
                "avg_hit_speed": _float_or_none(sc.get("avg_hit_speed")),
                "avg_launch_angle": _float_or_none(sc.get("avg_hit_angle")),
                "k_percent": _float_or_none(sc.get("k_percent")),
                "bb_percent": _float_or_none(sc.get("bb_percent")),
                "babip": _float_or_none(sc.get("babip")),
            })
        return rows


def _float_or_none(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
