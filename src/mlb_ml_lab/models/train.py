"""Baseline model training with walk-forward validation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mlb_ml_lab.models.evaluate import classification_metrics


@dataclass
class WalkForwardSplit:
    """Expanding-window time series split for temporally dependent data.

    Each fold expands the training window forward while keeping the test
    window as a contiguous block of games after the training cut-off.
    An optional ``gap`` (in days) between train and test prevents
    information from the very next day leaking into training.

    Args:
        n_splits: Number of folds.
        min_train_size: Minimum number of games required in the first
                        training window.
        gap: Number of days to skip between the last training game and
             the first test game.
    """

    n_splits: int = 5
    min_train_size: int = 30
    gap: int = 0

    def split(
        self, dates: list[date]
    ) -> list[tuple[list[int], list[int]]]:
        """Generate train/test index pairs for each fold.

        Args:
            dates: Per-row game dates, **must already be sorted
                   chronologically**.

        Returns:
            List of ``(train_idx, test_idx)`` tuples, one per fold.
        """
        n = len(dates)
        fold_size = (n - self.min_train_size) // self.n_splits
        if fold_size < 1:
            raise ValueError(
                f"{n} rows is too few for {self.n_splits} folds "
                f"(need at least {self.min_train_size + self.n_splits})"
            )

        folds: list[tuple[list[int], list[int]]] = []
        for i in range(self.n_splits):
            train_end = self.min_train_size + i * fold_size
            test_start = train_end + self.gap
            test_end = test_start + fold_size
            test_end = min(test_end, n)
            if test_start >= n:
                break
            train_idx = list(range(train_end))
            test_idx = list(range(test_start, test_end))
            folds.append((train_idx, test_idx))
        return folds


def _merge_features_targets(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge features and targets on (player_id, game_pk)."""
    key_cols = ("player_id", "game_pk")
    target_map: dict[tuple, dict[str, Any]] = {}
    for t in targets:
        key = (t["player_id"], t["game_pk"])
        target_map[key] = t

    merged: list[dict[str, Any]] = []
    for row in feature_matrix:
        key = (row["player_id"], row["game_pk"])
        tgt = target_map.get(key)
        if tgt is None:
            continue
        merged_row = dict(row)
        for k, v in tgt.items():
            if k not in key_cols:
                merged_row[k] = v
        merged.append(merged_row)
    return merged


def _feature_columns(
    rows: list[dict[str, Any]],
    exclude: set[str] | None = None,
) -> list[str]:
    """Return sorted list of numeric feature column names."""
    exclude_set = {
        "player_id",
        "game_pk",
        "date",
        "hits",
        *(exclude or set()),
    }
    candidate: set[str] = set()
    for row in rows:
        candidate.update(k for k in row if k not in exclude_set)

    cols: list[str] = []
    for col in sorted(candidate):
        vals = [r.get(col) for r in rows]
        numeric = True
        for v in vals:
            if v is None:
                continue
            try:
                float(v)
            except (TypeError, ValueError):
                numeric = False
                break
        if numeric:
            cols.append(col)
    return cols


def train_baselines(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    target_col: str = "target_0.5",
    model_types: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run walk-forward validation for one or more baseline classifiers.

    Args:
        feature_matrix: Output of ``build_feature_matrix()``.
        targets: Output of ``make_targets()``.
        target_col: Which target column to predict (e.g. ``"target_0.5"``
                    or ``"target_1.5"``).
        model_types: Which models to train.  Options: ``"lr"``
                     (LogisticRegression), ``"xgb"`` (XGBClassifier).
                     Defaults to ``["lr", "xgb"]``.
        n_splits: Number of walk-forward folds.
        seed: Random seed for reproducibility.

    Returns:
        Dict with structure::

            {
                "target_col": "...",
                "models": {
                    "lr": {
                        "fold_metrics": [{fold-0 metrics}, ...],
                        "avg_accuracy": ...,
                        "avg_auc": ...,
                        "n_folds": ...,
                    },
                    "xgb": { ... },
                },
                "n_train_total": ...,
                "n_test_total": ...,
            }
    """
    model_types = model_types or ["lr", "xgb"]
    merged = _merge_features_targets(feature_matrix, targets)
    if not merged:
        return {"target_col": target_col, "models": {}, "error": "no merged rows"}

    dates = [row["date"] for row in merged]
    feat_cols = _feature_columns(merged)

    x_all = np.array(
        [[row[c] for c in feat_cols] for row in merged], dtype=np.float64
    )
    y_all = np.array(
        [row[target_col] for row in merged], dtype=np.int32
    )

    imputer = SimpleImputer(strategy="median")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        x_all = imputer.fit_transform(x_all)
    x_all = np.nan_to_num(x_all, nan=0.0)

    splitter = WalkForwardSplit(n_splits=n_splits)
    folds = splitter.split(dates)

    results: dict[str, Any] = {
        "target_col": target_col,
        "models": {},
        "n_train_total": 0,
        "n_test_total": 0,
    }

    for model_type in model_types:
        model_type = model_type.strip().lower()
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            x_train = x_all[train_idx]
            y_train = y_all[train_idx]
            x_test = x_all[test_idx]
            y_test = y_all[test_idx]

            if model_type == "lr":
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                )
            elif model_type == "xgb":
                model = XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=0,
                    eval_metric="logloss",
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)[:, 1]

            metrics = classification_metrics(
                y_test.tolist(),
                y_pred.tolist(),
                y_proba.tolist(),
            )
            metrics["fold"] = fold_idx
            metrics["n_train"] = len(train_idx)
            metrics["n_test"] = len(test_idx)
            metrics["model_type"] = model_type
            fold_metrics.append(metrics)

        avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        avg_auc = float(
            np.mean(
                [m["auc"] for m in fold_metrics if not np.isnan(m.get("auc", float("nan")))]
            )
        )

        model_entry: dict[str, Any] = {
            "fold_metrics": fold_metrics,
            "avg_accuracy": avg_acc,
            "avg_auc": avg_auc,
            "n_folds": len(fold_metrics),
        }
        results["models"][model_type] = model_entry

        total_train = sum(m["n_train"] for m in fold_metrics)
        total_test = sum(m["n_test"] for m in fold_metrics)
        results["n_train_total"] += total_train
        results["n_test_total"] += total_test

    return results
