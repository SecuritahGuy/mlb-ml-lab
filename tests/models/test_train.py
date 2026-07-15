"""Tests for model training module."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from mlb_ml_lab.models.train import (
    DEFAULT_PARAM_GRIDS,
    MODEL_HELP,
    WalkForwardSplit,
    _build_model,
    tune_hyperparameters,
)


# ---------------------------------------------------------------------------
# _build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_default_params_produce_fittable_model(self):
        # MLX needs more data and is slower — test separately
        fast_models = [k for k in MODEL_HELP if k != "mlx"]
        for mt in fast_models:
            model = _build_model(mt, seed=42)
            assert model is not None
            import numpy as np

            x = np.array([[0.0], [1.0]])
            y = np.array([0, 1])
            model.fit(x, y)
            preds = model.predict(x)
            assert len(preds) == 2

    def test_mlx_build_and_smoke(self):
        # MLX needs more data for meaningful training
        pytest.importorskip("mlx.core", reason="MLX requires Apple Silicon (macOS)")
        import numpy as np

        model = _build_model("mlx", seed=42, params={"epochs": 5, "hidden_dims": (4,)})
        rng = np.random.RandomState(0)
        x = rng.randn(20, 3).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int32)
        model.fit(x, y)
        preds = model.predict(x)
        assert len(preds) == 20
        assert set(preds).issubset({0, 1})

    def test_custom_params_override_defaults(self):
        model = _build_model("lr", seed=42, params={"C": 0.01})
        assert model.C == 0.01  # type: ignore[attr-defined]

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            _build_model("nonexistent", seed=42)


# ---------------------------------------------------------------------------
# WalkForwardSplit
# ---------------------------------------------------------------------------


class TestWalkForwardSplit:
    def _dates(self, n: int) -> list[date]:
        start = date(2025, 4, 1)
        return [start + timedelta(days=i) for i in range(n)]

    def test_at_least_one_fold(self):
        dates = self._dates(60)
        splitter = WalkForwardSplit(n_splits=3, min_train_size=30)
        folds = splitter.split(dates)
        assert len(folds) >= 1

    def test_too_few_dates_raises(self):
        dates = self._dates(5)
        splitter = WalkForwardSplit(min_train_size=30)
        with pytest.raises(ValueError):
            splitter.split(dates)


# ---------------------------------------------------------------------------
# tune_hyperparameters
# ---------------------------------------------------------------------------


def _feature_row(
    player_id: int,
    game_pk: int,
    d: str,
    feat_val: float = 0.5,
) -> dict:
    return {
        "player_id": player_id,
        "game_pk": game_pk,
        "date": date.fromisoformat(d),
        "rolling_avg": feat_val,
    }


def _target_row(
    player_id: int,
    game_pk: int,
    d: str,
    hits: int = 0,
) -> dict:
    return {
        "player_id": player_id,
        "game_pk": game_pk,
        "date": date.fromisoformat(d),
        "hits": hits,
        "target_0.5": 1 if hits > 0 else 0,
        "target_1.5": 1 if hits > 1 else 0,
    }


def _make_dates(n: int) -> list[date]:
    start = date(2025, 4, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _interspersed_feature_targets(
    n: int,
    noise: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """Generate feature/target rows with interspersed classes."""
    feat = []
    tgt = []
    for i, d in enumerate(_make_dates(n)):
        # Alternate positive/negative so any fold has both classes
        is_positive = i % 2 == 0
        feat_val = 0.5 + (noise if is_positive else -noise)
        feat.append(_feature_row(1, 100 + i, d.isoformat(), feat_val=feat_val))
        tgt.append(_target_row(1, 100 + i, d.isoformat(), hits=1 if is_positive else 0))
    return feat, tgt


class TestTuneHyperparameters:
    def test_returns_best_params(self):
        feat, tgt = _interspersed_feature_targets(60)
        grid = {"C": [0.1, 1.0, 10.0]}
        result = tune_hyperparameters(
            feat,
            tgt,
            target_col="target_0.5",
            model_type="lr",
            param_grid=grid,
            n_trials=3,
            n_splits=2,
        )
        assert "best_params" in result
        assert "best_score" in result
        assert "trials" in result
        assert result["best_score"] > 0.0
        assert len(result["trials"]) == 3

    def test_returns_proper_structure(self):
        feat, tgt = _interspersed_feature_targets(60)
        grid = {"C": [0.1, 1.0]}
        result = tune_hyperparameters(
            feat,
            tgt,
            target_col="target_0.5",
            model_type="lr",
            param_grid=grid,
            n_trials=2,
            n_splits=2,
        )
        assert set(result.keys()) >= {
            "best_params",
            "best_score",
            "best_std",
            "trials",
            "target_col",
            "model_type",
            "metric",
            "n_trials",
            "n_splits",
        }
        assert result["target_col"] == "target_0.5"
        assert result["model_type"] == "lr"
        assert result["metric"] == "auc"

    def test_can_optimise_log_loss(self):
        feat, tgt = _interspersed_feature_targets(60)
        grid = {"C": [0.1, 1.0]}
        result = tune_hyperparameters(
            feat,
            tgt,
            target_col="target_0.5",
            model_type="lr",
            param_grid=grid,
            n_trials=2,
            n_splits=2,
            metric="log_loss",
        )
        assert result["metric"] == "log_loss"
        assert result["best_score"] > 0.0

    def test_empty_inputs(self):
        result = tune_hyperparameters([], [], target_col="target_0.5")
        assert "error" in result

    def test_default_grid_available_for_known_models(self):
        for mt in MODEL_HELP:
            grid = DEFAULT_PARAM_GRIDS.get(mt)
            assert grid is not None, f"No default grid for {mt}"
            assert len(grid) >= 1

    def test_explicit_grid_for_xgb(self):
        feat, tgt = _interspersed_feature_targets(60)
        grid = {"n_estimators": [50, 100], "max_depth": [3, 5]}
        result = tune_hyperparameters(
            feat,
            tgt,
            target_col="target_0.5",
            model_type="xgb",
            param_grid=grid,
            n_trials=2,
            n_splits=2,
        )
        assert result["model_type"] == "xgb"
        assert "n_estimators" in result["best_params"]
        assert "max_depth" in result["best_params"]
