"""Feature matrix assembler.

Orchestrates all registered ``FeatureExtractor`` instances to produce
a unified feature matrix and target vector from raw game logs.
"""

from __future__ import annotations

from typing import Any

from mibl.data.schemas import PlayerGameLog

from mibl.features.base import get_registry, FeatureMeta


def build_feature_matrix(
    game_logs: list[PlayerGameLog],
    season: int | None = None,
    teams: list[dict[str, Any]] | None = None,
    statcast_batters: list[dict[str, str]] | None = None,
    expected_stats: list[dict[str, str]] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a unified feature matrix from game logs.

    Runs every registered ``FeatureExtractor``, then merges all outputs
    into a single list of dicts keyed by (player_id, game_pk, date).

    Args:
        game_logs: Player game logs to compute features from.
        season: Season year (used by context features).
        teams: Team list from ``MlbClient.get_teams()`` (used by park
               factor features).
        statcast_batters: Rows from ``MlbClient.get_statcast_batters()``.
        expected_stats: Rows from ``MlbClient.get_expected_stats()``.
        extra_kwargs: Additional keyword arguments forwarded to every
                      extractor, e.g.::

                          game_contexts: dict[int, dict]  # game_pk → context
                          opponent_pitching: dict[int, dict]  # team_id → stats

    Returns:
        List of feature dicts.  Each dict has ``player_id``, ``game_pk``,
        ``date`` plus all extracted feature columns.
    """
    kwargs: dict[str, Any] = {
        "game_logs": game_logs,
        "season": season,
        "teams": teams,
        "statcast_batters": statcast_batters,
        "expected_stats": expected_stats,
        **(extra_kwargs or {}),
    }

    registry = get_registry()
    all_feature_rows: list[dict[str, Any]] = []

    for name, extractor_cls in registry.items():
        extractor = extractor_cls()
        rows = extractor.extract(**kwargs)
        all_feature_rows.extend(rows)

    if not all_feature_rows:
        return []

    key_cols = ("player_id", "game_pk", "date")
    merged: dict[tuple, dict[str, Any]] = {}
    for row in all_feature_rows:
        key = tuple(row[k] for k in key_cols)
        if key not in merged:
            merged[key] = {k: row[k] for k in key_cols}
        for k, v in row.items():
            if k not in key_cols:
                merged[key][k] = v

    return list(merged.values())


def describe_features() -> list[FeatureMeta]:
    """Return metadata for all registered features."""
    registry = get_registry()
    metas: list[FeatureMeta] = []
    for name, cls in registry.items():
        metas.extend(cls().features)
    return metas
