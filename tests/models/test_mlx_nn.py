"""Tests for MLX neural network classifier."""

from __future__ import annotations

import tempfile

import numpy as np

from mlb_ml_lab.models.mlx_nn import (
    HitPredictor,
    MlxNNClassifier,
    _flatten_params,
    load_mlx_model,
    save_mlx_model,
)


# ---------------------------------------------------------------------------
# HitPredictor
# ---------------------------------------------------------------------------


class TestHitPredictor:
    def test_forward_pass_shape(self):
        model = HitPredictor(n_features=10, hidden_dims=(8, 4))
        import mlx.core as mx
        x = mx.random.normal((32, 10))
        y = model(x)
        assert y.shape == (32, 1)

    def test_train_eval_mode(self):
        model = HitPredictor(n_features=5)
        model.train()
        assert model.training is True
        model.eval()
        assert model.training is False

    def test_has_parameters(self):
        model = HitPredictor(n_features=5, hidden_dims=(8, 4))
        params = model.parameters()
        flat = _flatten_params(params)
        # 3 Linear layers: 5→8, 8→4, 4→1
        weight_keys = [k for k in flat if "weight" in k]
        bias_keys = [k for k in flat if "bias" in k]
        assert len(weight_keys) == 3
        assert len(bias_keys) == 3


# ---------------------------------------------------------------------------
# MlxNNClassifier
# ---------------------------------------------------------------------------


def _synthetic_data(
    n: int = 200, n_features: int = 5, seed: int = 42, bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] + bias) > 0).astype(np.int32)
    return X, y


class TestMlxNNClassifier:
    def test_fit_and_predict(self):
        X, y = _synthetic_data(200)
        clf = MlxNNClassifier(
            hidden_dims=(16, 8), epochs=30, batch_size=32, seed=42,
        )
        clf.fit(X, y)
        assert clf.model_ is not None
        assert clf.scaler_ is not None

        preds = clf.predict(X)
        assert preds.shape == (200,)
        assert preds.dtype == np.int32
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape_and_range(self):
        X, y = _synthetic_data(200)
        clf = MlxNNClassifier(hidden_dims=(16,), epochs=30, seed=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert probs.shape == (200, 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        # Positive-class probabilities should sum with negative to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_sklearn_is_fitted(self):
        clf = MlxNNClassifier()
        assert clf.__sklearn_is_fitted__() is False
        X, y = _synthetic_data(100)
        clf.fit(X, y)
        assert clf.__sklearn_is_fitted__() is True

    def test_save_load_round_trip(self):
        X, y = _synthetic_data(200)
        clf = MlxNNClassifier(hidden_dims=(16, 8), epochs=30, seed=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            save_mlx_model(clf, tmp)
            loaded = load_mlx_model(tmp)
            probs2 = loaded.predict_proba(X)

        np.testing.assert_allclose(probs, probs2, atol=1e-6)

    def test_reproducible_seed(self):
        X, y = _synthetic_data(200)
        clf1 = MlxNNClassifier(hidden_dims=(16,), epochs=20, seed=42)
        clf2 = MlxNNClassifier(hidden_dims=(16,), epochs=20, seed=42)
        clf1.fit(X, y)
        clf2.fit(X, y)
        np.testing.assert_allclose(
            clf1.predict_proba(X), clf2.predict_proba(X), atol=1e-5,
        )

    def test_different_seed_different_probs(self):
        X, y = _synthetic_data(200)
        clf1 = MlxNNClassifier(hidden_dims=(16,), epochs=20, seed=1)
        clf2 = MlxNNClassifier(hidden_dims=(16,), epochs=20, seed=999)
        clf1.fit(X, y)
        clf2.fit(X, y)
        # Very unlikely to produce identical probabilities
        probs1 = clf1.predict_proba(X)
        probs2 = clf2.predict_proba(X)
        assert not np.allclose(probs1, probs2, atol=1e-3)


# ---------------------------------------------------------------------------
# _flatten_params
# ---------------------------------------------------------------------------


class TestFlattenParams:
    def test_flattens_nested_dict(self):
        import mlx.core as mx
        params = {
            "net": {
                "layers": [
                    {"weight": mx.array([[1.0, 2.0]])},
                ],
            },
        }
        flat = _flatten_params(params)
        assert "net.layers.0.weight" in flat
        assert flat["net.layers.0.weight"].shape == (1, 2)
