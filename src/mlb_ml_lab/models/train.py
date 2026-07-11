"""Model training with walk-forward validation and ensemble support."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mlb_ml_lab.models.evaluate import classification_metrics
from mlb_ml_lab.models.mlx_nn import MlxNNClassifier, save_mlx_model, load_mlx_model


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


_BASE_PARAMS: dict[str, dict[str, Any]] = {
    "lr": {"max_iter": 1000},
    "xgb": {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
        "eval_metric": "logloss", "verbosity": 0,
    },
    "rf": {"n_estimators": 300, "max_depth": 8},
    "lgb": {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
        "verbosity": -1, "deterministic": True,
    },
    "mlx": {
        "hidden_dims": (256, 128, 64),
        "dropout_prob": 0.3,
        "use_batch_norm": True,
        "class_weight": "balanced",
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 256,
        "early_stop_patience": 10,
        "l2_reg": 1e-5,
    },
}

_MODEL_CLASSES: dict[str, Any] = {
    "lr": LogisticRegression,
    "xgb": XGBClassifier,
    "rf": RandomForestClassifier,
    "lgb": LGBMClassifier,
}


def _build_model(
    model_type: str, seed: int, params: dict[str, Any] | None = None,
) -> Any:
    if model_type == "mlx":
        kwargs = dict(_BASE_PARAMS.get("mlx", {}))
        if params:
            kwargs.update(params)
        return MlxNNClassifier(seed=seed, **kwargs)

    model_cls = _MODEL_CLASSES.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    kwargs = _BASE_PARAMS.get(model_type, {}).copy()
    kwargs["random_state"] = seed
    kwargs["n_jobs"] = -1
    if params:
        kwargs.update(params)
    if model_type == "lgb":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return model_cls(**kwargs)
    return model_cls(**kwargs)


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
        vals: list[float] = []
        numeric = True
        for v in (r.get(col) for r in rows):
            if v is None:
                vals.append(float("nan"))
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                numeric = False
                break
        if not numeric:
            continue
        # Drop columns that are all NaN, all zeros, or constant
        good = [v for v in vals if not np.isnan(v)]
        if not good or min(good) == max(good):
            continue
        cols.append(col)
    return cols


MODEL_HELP = {
    "lr": "LogisticRegression",
    "xgb": "XGBoost",
    "rf": "RandomForest",
    "lgb": "LightGBM",
    "mlx": "MLX-MLP",
}

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "mlx": {
        "learning_rate": [0.001, 0.003, 0.01],
        "dropout_prob": [0.1, 0.2, 0.3],
        "batch_size": [32, 64],
    },
    "lr": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs", "liblinear"],
    },
    "xgb": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "rf": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "lgb": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "num_leaves": [15, 31, 63],
    },
}


def tune_hyperparameters(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    target_col: str = "target_0.5",
    model_type: str = "xgb",
    param_grid: dict[str, list[Any]] | None = None,
    n_trials: int = 20,
    n_splits: int = 5,
    seed: int = 42,
    metric: str = "auc",
) -> dict[str, Any]:
    """Random search over hyperparameters inside walk-forward validation.

    Args:
        feature_matrix: Output from ``build_feature_matrix()``.
        targets: Output from ``make_targets()``.
        target_col: Which target column to predict.
        model_type: Classifier type (``lr``, ``xgb``, ``rf``, ``lgb``).
        param_grid: Dict mapping param names to lists of candidate values.
                    Defaults to ``DEFAULT_PARAM_GRIDS[model_type]``.
        n_trials: Number of random parameter combinations to evaluate.
        n_splits: Number of walk-forward folds.
        seed: Random seed for reproducibility.
        metric: Metric to optimise (``auc`` or ``log_loss``).

    Returns:
        Dict with keys ``best_params``, ``best_score``, ``best_std``,
        ``trials`` (list of per-trial results), ``target_col``,
        ``model_type``, ``metric``, ``n_trials``, ``n_splits``.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS.get(model_type)
        if param_grid is None:
            raise ValueError(
                f"No default param grid for '{model_type}'. "
                f"Supply a ``param_grid`` explicitly."
            )

    merged = _merge_features_targets(feature_matrix, targets)
    if not merged:
        return {"error": "no merged rows", "best_params": {}, "best_score": 0.0,
                "best_std": 0.0, "trials": []}
    merged.sort(key=lambda r: r["date"])
    dates = [row["date"] for row in merged]

    feat_cols = _feature_columns(merged)
    x_all = np.array(
        [[row[c] for c in feat_cols] for row in merged], dtype=np.float64
    )
    y_all = np.array([row[target_col] for row in merged], dtype=np.int32)

    imputer = SimpleImputer(strategy="median")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        x_all = imputer.fit_transform(x_all)
    x_all = np.nan_to_num(x_all, nan=0.0)

    splitter = WalkForwardSplit(n_splits=n_splits)
    folds = splitter.split(dates)
    if not folds:
        return {"error": "no folds generated", "best_params": {}, "best_score": 0.0,
                "best_std": 0.0, "trials": []}

    # Build a list of all parameter keys and their candidate lists.
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]

    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = []
    best_score = -float("inf") if metric == "auc" else float("inf")
    best_params: dict[str, Any] = {}
    best_std = 0.0

    for trial in range(n_trials):
        combo: dict[str, Any] = {}
        for k, vals in zip(param_keys, param_values):
            idx = rng.integers(len(vals))
            combo[k] = vals[int(idx)]

        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            metrics = _run_fold(
                model_type, x_all[train_idx], y_all[train_idx],
                x_all[test_idx], y_all[test_idx], fold_idx, seed,
                params=combo,
            )
            fold_metrics.append(metrics)

        scores = [m[metric] for m in fold_metrics if not np.isnan(m.get(metric, float("nan")))]
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0

        trials.append({
            "params": dict(combo),
            metric: mean_score,
            f"{metric}_std": std_score,
            "fold_metrics": fold_metrics,
        })

        better = mean_score > best_score if metric == "auc" else mean_score < best_score
        if better:
            best_score = mean_score
            best_params = dict(combo)
            best_std = std_score

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_std": best_std,
        "trials": trials,
        "target_col": target_col,
        "model_type": model_type,
        "metric": metric,
        "n_trials": n_trials,
        "n_splits": n_splits,
    }


def _run_fold(
    model_type: str, x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray, fold_idx: int, seed: int,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    model = _build_model(model_type, seed, params=params)
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


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_model(
    model: Any,
    feature_cols: list[str],
    imputer: SimpleImputer,
    directory: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a trained model + its preprocessors to disk.

    Creates *directory* (if needed) and writes::

        {directory}/
            model.joblib          — fitted classifier
            feature_cols.joblib   — list of feature column names
            imputer.joblib        — fitted ``SimpleImputer``
            metadata.json         — training metadata

    Args:
        model: Fitted sklearn-compatible classifier.
        feature_cols: Feature column names in the order expected by
                      *model*.
        imputer: Fitted ``SimpleImputer`` used during training.
        directory: Output directory.
        metadata: Optional extra metadata (e.g. ``target_col``,
                  ``feature_count``).

    Returns:
        The *directory* path.
    """
    os.makedirs(directory, exist_ok=True)
    if isinstance(model, MlxNNClassifier):
        save_mlx_model(model, directory, metadata)
    else:
        joblib.dump(model, os.path.join(directory, "model.joblib"))
    joblib.dump(feature_cols, os.path.join(directory, "feature_cols.joblib"))
    joblib.dump(imputer, os.path.join(directory, "imputer.joblib"))
    if metadata and not isinstance(model, MlxNNClassifier):
        meta_path = os.path.join(directory, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    return directory


def load_model(
    directory: str,
) -> tuple[Any, list[str], SimpleImputer, dict[str, Any]]:
    """Load a model saved by ``save_model()``.

    Returns:
        ``(model, feature_cols, imputer, metadata)`` tuple.
    """
    config_path = os.path.join(directory, "config.json")
    if os.path.isfile(config_path):
        model = load_mlx_model(directory)
        metadata: dict[str, Any] = {}
        meta_path = os.path.join(directory, "config.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
    else:
        model = joblib.load(os.path.join(directory, "model.joblib"))
        metadata = {}
        meta_path = os.path.join(directory, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
    feature_cols: list[str] = joblib.load(
        os.path.join(directory, "feature_cols.joblib")
    )
    imputer: SimpleImputer = joblib.load(
        os.path.join(directory, "imputer.joblib")
    )
    return model, feature_cols, imputer, metadata


# ---------------------------------------------------------------------------
# Train final model on all available data (no validation split)
# ---------------------------------------------------------------------------


def train_final(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    target_col: str = "target_0.5",
    model_type: str = "lgb",
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a single model on ALL data and return it with preprocessors.

    This is the function to call *after* walk-forward validation.  It
    uses the same preprocessing pipeline as ``train_baselines`` but
    fits on every row (no folds).

    Returns a dict suitable for ``save_model()``::

        {
            "model": <fitted classifier>,
            "feature_cols": [...],
            "imputer": <fitted SimpleImputer>,
            "metadata": {"target_col": ..., "model_type": ...},
        }
    """
    merged = _merge_features_targets(feature_matrix, targets)
    if not merged:
        return {"model": None, "feature_cols": [], "imputer": None,
                "metadata": {"error": "no merged rows"}}
    merged.sort(key=lambda r: r["date"])

    feat_cols = _feature_columns(merged)
    x = np.array(
        [[row[c] for c in feat_cols] for row in merged], dtype=np.float64
    )
    y = np.array([row[target_col] for row in merged], dtype=np.int32)

    imputer = SimpleImputer(strategy="median")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        x = imputer.fit_transform(x)
    x = np.nan_to_num(x, nan=0.0)

    model = _build_model(model_type, seed, params=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(x, y)

    return {
        "model": model,
        "feature_cols": feat_cols,
        "imputer": imputer,
        "metadata": {
            "target_col": target_col,
            "model_type": model_type,
            "n_rows": len(merged),
            "n_features": len(feat_cols),
        },
    }
