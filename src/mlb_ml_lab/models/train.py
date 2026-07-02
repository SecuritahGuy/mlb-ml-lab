"""Model training with walk-forward validation and ensemble support."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
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
        unique_dates = sorted(set(dates))
        n_dates = len(unique_dates)
        if n_dates < self.min_train_size:
            raise ValueError(
                f"{n_dates} unique dates is too few "
                f"(min_train_size={self.min_train_size})"
            )

        fold_dates = (
            n_dates - self.min_train_size
        ) // self.n_splits
        if fold_dates < 1:
            raise ValueError(
                f"{n_dates} unique dates is too few for {self.n_splits} folds"
            )

        folds: list[tuple[list[int], list[int]]] = []
        for i in range(self.n_splits):
            train_end_date = unique_dates[
                self.min_train_size + i * fold_dates - 1
            ]
            test_start_idx = self.min_train_size + i * fold_dates
            test_start_date = unique_dates[test_start_idx]
            if self.gap:
                test_start_date += timedelta(days=self.gap)
            test_end_idx = min(
                test_start_idx + fold_dates, n_dates
            )
            test_end_date = unique_dates[test_end_idx - 1]

            train_idx = [
                j for j, d in enumerate(dates) if d <= train_end_date
            ]
            test_idx = [
                j for j, d in enumerate(dates)
                if test_start_date <= d <= test_end_date
            ]
            if not test_idx:
                break
            folds.append((train_idx, test_idx))
        return folds


def _merge_features_targets(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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


_ALWAYS_EXCLUDE: set[str] = {
    "player_id", "game_pk", "date", "hits",
    "target_0.5", "target_1.5",
}


def _feature_columns(
    rows: list[dict[str, Any]],
    exclude: set[str] | None = None,
) -> list[str]:
    exclude_set = {
        *_ALWAYS_EXCLUDE,
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


def _build_model(
    model_type: str, seed: int
) -> Any:
    if model_type == "lr":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if model_type == "xgb":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "lgb":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
                deterministic=True,
            )
    raise ValueError(f"Unknown model type: {model_type}")


MODEL_HELP = {
    "lr": "LogisticRegression",
    "xgb": "XGBoost",
    "rf": "RandomForest",
    "lgb": "LightGBM",
}


def _run_fold(
    model_type: str, x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray, fold_idx: int, seed: int,
) -> dict[str, float]:
    model = _build_model(model_type, seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(
        y_test.tolist(), y_pred.tolist(), y_proba.tolist(),
    )
    metrics["fold"] = fold_idx
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    metrics["model_type"] = model_type
    return metrics


def _run_ensemble_fold(
    eval_models: list[str], x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray, fold_idx: int, seed: int,
) -> dict[str, float]:
    probas: list[np.ndarray] = []
    for mt in eval_models:
        model = _build_model(mt, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(x_train, y_train)
            probas.append(model.predict_proba(x_test)[:, 1])
    avg_proba = np.mean(probas, axis=0)
    avg_pred = (avg_proba > 0.5).astype(np.int32)
    metrics = classification_metrics(
        y_test.tolist(), avg_pred.tolist(), avg_proba.tolist(),
    )
    metrics["fold"] = fold_idx
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    metrics["model_type"] = "ensemble"
    return metrics


def train_baselines(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    target_col: str = "target_0.5",
    model_types: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    model_types = model_types or ["lr", "xgb", "rf", "lgb", "ensemble"]
    merged = _merge_features_targets(feature_matrix, targets)
    if not merged:
        return {"target_col": target_col, "models": {}, "error": "no merged rows"}
    merged.sort(key=lambda r: r["date"])

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

    eval_models = [m for m in model_types if m != "ensemble"]
    do_ensemble = "ensemble" in model_types

    for model_type in model_types:
        if model_type == "ensemble":
            continue
        model_type = model_type.strip().lower()
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            metrics = _run_fold(
                model_type, x_all[train_idx], y_all[train_idx],
                x_all[test_idx], y_all[test_idx], fold_idx, seed,
            )
            fold_metrics.append(metrics)

        avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
        avg_auc = float(
            np.mean(
                [m["auc"] for m in fold_metrics
                 if not np.isnan(m.get("auc", float("nan")))]
            )
        )
        results["models"][model_type] = {
            "fold_metrics": fold_metrics,
            "avg_accuracy": avg_acc,
            "avg_auc": avg_auc,
            "n_folds": len(fold_metrics),
        }
        results["n_train_total"] += sum(m["n_train"] for m in fold_metrics)
        results["n_test_total"] += sum(m["n_test"] for m in fold_metrics)

    if do_ensemble:
        ensemble_metrics = [
            _run_ensemble_fold(
                eval_models, x_all[train_idx], y_all[train_idx],
                x_all[test_idx], y_all[test_idx], fold_idx, seed,
            )
            for fold_idx, (train_idx, test_idx) in enumerate(folds)
        ]
        results["models"]["ensemble"] = {
            "fold_metrics": ensemble_metrics,
            "avg_accuracy": float(np.mean([m["accuracy"] for m in ensemble_metrics])),
            "avg_auc": float(
                np.mean(
                    [m["auc"] for m in ensemble_metrics
                     if not np.isnan(m.get("auc", float("nan")))]
                )
            ),
            "n_folds": len(ensemble_metrics),
        }

    return results
