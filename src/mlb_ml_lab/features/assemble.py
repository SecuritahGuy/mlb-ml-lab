"""Feature matrix assembler.

Orchestrates all registered ``FeatureExtractor`` instances to produce
a unified feature matrix and target vector from raw game logs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from mlb_ml_lab.data.schemas import PlayerGameLog

from mlb_ml_lab.features.base import get_registry, FeatureMeta


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

    for _, extractor_cls in registry.items():
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
    for _, cls in registry.items():
        metas.extend(cls().features)
    return metas


# ---------------------------------------------------------------------------
# Cache feature matrix + targets to disk  (JSONL format)
# ---------------------------------------------------------------------------


def save_feature_data(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    directory: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write feature matrix and targets to a directory as JSONL files.

    Creates *directory* (if it doesn't exist) and writes::

        {directory}/
            features.jsonl   — one JSON object per feature row
            targets.jsonl    — one JSON object per target row
            metadata.json    — run metadata dict

    Args:
        feature_matrix: Output from ``build_feature_matrix()``.
        targets: Output from ``make_targets()``.
        directory: Directory path to write into.
        metadata: Optional extra metadata to merge (e.g. ``season``,
                  ``team_id``).

    Returns:
        The *directory* path.
    """
    os.makedirs(directory, exist_ok=True)

    feature_cols = _col_names(feature_matrix)
    target_cols = _col_names(targets)
    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "feature_rows": len(feature_matrix),
        "target_rows": len(targets),
        **(metadata or {}),
    }

    with open(os.path.join(directory, "features.jsonl"), "w", encoding="utf-8") as f:
        for row in feature_matrix:
            f.write(json.dumps(row, default=str) + "\n")

    with open(os.path.join(directory, "targets.jsonl"), "w", encoding="utf-8") as f:
        for row in targets:
            f.write(json.dumps(row, default=str) + "\n")

    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return directory


def load_feature_data(
    directory: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load feature matrix and targets previously written by
    ``save_feature_data()``.

    Args:
        directory: Directory written by ``save_feature_data()``.

    Returns:
        ``(feature_matrix, targets, metadata)`` tuple.
    """
    features: list[dict[str, Any]] = []
    with open(os.path.join(directory, "features.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                features.append(json.loads(line))

    targets_list: list[dict[str, Any]] = []
    with open(os.path.join(directory, "targets.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                targets_list.append(json.loads(line))

    with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    return features, targets_list, metadata


def load_game_logs(directory: str) -> list[dict[str, Any]]:
    """Load raw game logs saved by ``harvest_dataset.py``."""
    logs: list[dict[str, Any]] = []
    path = os.path.join(directory, "game_logs.jsonl")
    if not os.path.isfile(path):
        return logs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def _col_names(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())
