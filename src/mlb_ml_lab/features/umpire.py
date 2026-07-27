from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class UmpireFeatures(FeatureExtractor):
    """Home-plate umpire identity and experience features.

    Reads ``hp_umpire_id`` and ``hp_umpire_name`` from ``game_contexts``
    (enriched schedule data).  Optionally accepts pre-computed
    ``umpire_game_counts`` (dict[int, int]) in kwargs for an experience
    proxy.

    All features default to ``None`` when data is unavailable.
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="hp_umpire_id",
                description="Home-plate umpire MLBAM ID",
                source="umpire",
            ),
            FeatureMeta(
                name="umpire_game_count",
                description="Number of games this HP umpire worked this season",
                source="umpire",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        contexts: dict[int, dict[str, Any]] | None = kwargs.get("game_contexts")
        game_counts: dict[int, int] | None = kwargs.get("umpire_game_counts")

        if not game_logs:
            return []

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            ctx = (contexts or {}).get(log.game_pk, {}) if contexts else {}
            ump_id: int | None = ctx.get("hp_umpire_id")
            rows.append(
                {
                    "player_id": log.player_id,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    "hp_umpire_id": ump_id,
                    "umpire_game_count": (game_counts or {}).get(ump_id)
                    if ump_id
                    else None,
                }
            )
        return rows
