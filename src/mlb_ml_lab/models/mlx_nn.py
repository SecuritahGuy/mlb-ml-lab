"""MLX-based neural network for MLB hit prediction.

Uses Apple's MLX framework (Apple Silicon only). Provides an sklearn-compatible
classifier that plugs into the existing walk-forward training pipeline.
"""

from __future__ import annotations

from typing import Any

import joblib
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from sklearn.preprocessing import StandardScaler


class HitPredictor(nn.Module):
    """MLP for binary classification of MLB hit outcomes.

    Architecture::

        Input → Linear → BatchNorm → LeakyReLU → Dropout → ... → Linear → logits
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout_prob: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        dims = (n_features, *hidden_dims, 1)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm(dims[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class MlxNNClassifier:
    """Sklearn-compatible binary classifier backed by an MLX MLP.

    Wraps ``HitPredictor`` with ``fit`` / ``predict`` / ``predict_proba``
    so it can be used as a drop-in model type in the walk-forward pipeline.

    Parameters
    ----------
    hidden_dims:
        Sizes of hidden layers.
    dropout_prob:
        Dropout probability applied after each hidden layer during training.
    use_batch_norm:
        Whether to insert batch normalisation after each hidden Linear layer.
    class_weight:
        Weighting for the positive class.  ``None`` (no weighting),
        ``"balanced"`` (auto-compute from ``y``), or a ``float`` multiplier.
    learning_rate:
        Adam learning rate (initial value when using cosine schedule).
    epochs:
        Maximum number of training epochs.
    batch_size:
        Mini-batch size for SGD.
    early_stop_patience:
        Number of epochs with no improvement before stopping.
    l2_reg:
        L2 regularisation strength (applied to all weights).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout_prob: float = 0.3,
        use_batch_norm: bool = True,
        class_weight: str | float | None = None,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        early_stop_patience: int = 10,
        l2_reg: float = 1e-5,
        seed: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.l2_reg = l2_reg
        self.seed = seed

        # Set by fit()
        self.model_: HitPredictor | None = None
        self.scaler_: StandardScaler | None = None
        self._pos_weight_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MlxNNClassifier:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        mx.random.seed(self.seed)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        self.model_ = HitPredictor(
            n_features=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout_prob=self.dropout_prob,
            use_batch_norm=self.use_batch_norm,
        )

        # ── Class weighting ──────────────────────────────────────────
        if self.class_weight == "balanced":
            n_pos = float(y.sum())
            n_neg = float(len(y) - n_pos)
            self._pos_weight_ = n_neg / n_pos if n_pos > 0 else 1.0
        elif isinstance(self.class_weight, (int, float)):
            self._pos_weight_ = float(self.class_weight)
        else:
            self._pos_weight_ = 1.0

        # ── Optimiser + cosine LR schedule ───────────────────────────
        n_samples = X_scaled.shape[0]
        steps_per_epoch = max(1, n_samples // self.batch_size)
        total_steps = self.epochs * steps_per_epoch
        lr_schedule = optim.cosine_decay(
            self.learning_rate,
            total_steps,
            end=self.learning_rate * 0.01,
        )
        optimizer = optim.Adam(learning_rate=lr_schedule)

        def loss_fn(x_batch: mx.array, y_batch: mx.array) -> mx.array:
            logits = self.model_(x_batch)
            losses = nn.losses.binary_cross_entropy(logits, y_batch)
            # Positive-class weighting
            if self._pos_weight_ != 1.0:
                weights = mx.where(
                    y_batch > 0.5,
                    self._pos_weight_,
                    1.0,
                )
                losses = losses * weights
            base_loss = losses.mean()
            # L2 regularisation
            l2 = 0.0
            for _, p in tree_flatten(self.model_.parameters()):
                l2 += (p * p).sum()
            return base_loss + self.l2_reg * l2

        loss_and_grad_fn = nn.value_and_grad(self.model_, loss_fn)

        best_loss = float("inf")
        patience = 0
        epoch_losses: list[float] = []

        for epoch in range(self.epochs):
            self.model_.train()

            perm = np.random.RandomState(self.seed + epoch).permutation(n_samples)
            X_shuffled = X_scaled[perm]
            y_shuffled = y[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                x_batch = mx.array(X_shuffled[start:end])
                y_batch = mx.array(y_shuffled[start:end, np.newaxis])

                loss, grads = loss_and_grad_fn(x_batch, y_batch)

                # Gradient clipping (element-wise)
                grads = tree_unflatten(
                    [(k, mx.clip(v, -5.0, 5.0)) for k, v in tree_flatten(grads)]
                )

                optimizer.update(self.model_, grads)
                mx.eval(self.model_.parameters(), optimizer.state)

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            epoch_losses.append(avg_loss)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape ``(n, 2)`` — sklearn convention.

        Column 0 is ``1 - p``, column 1 is ``p`` (positive-class prob).
        """
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.model_.eval()
        logits = self.model_(mx.array(X_scaled))
        p = np.asarray(mx.sigmoid(logits)).reshape(-1, 1)
        return np.concatenate([1.0 - p, p], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int32)

    def __sklearn_is_fitted__(self) -> bool:
        return self.model_ is not None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _flatten_params(params: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten a nested MLX parameter dict into ``{"path.to.key": array}``."""
    flat: dict[str, mx.array] = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_params(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                flat.update(_flatten_params({str(i): item}, key))
        elif isinstance(v, mx.array):
            flat[key] = v
    return flat


def save_mlx_model(
    classifier: MlxNNClassifier,
    directory: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a fitted ``MlxNNClassifier`` to disk.

    Writes::

        {directory}/
            model.safetensors   — MLX model weights (flat key format)
            scaler.joblib       — fitted StandardScaler
            config.json         — hyperparameters + metadata
    """
    import json
    import os

    os.makedirs(directory, exist_ok=True)

    weights = _flatten_params(classifier.model_.parameters())
    mx.save_safetensors(os.path.join(directory, "model.safetensors"), weights)
    joblib.dump(classifier.scaler_, os.path.join(directory, "scaler.joblib"))

    config = {
        "n_features": classifier.scaler_.n_features_in_,
        "hidden_dims": list(classifier.hidden_dims),
        "dropout_prob": classifier.dropout_prob,
        "use_batch_norm": classifier.use_batch_norm,
        "class_weight": classifier.class_weight,
        "learning_rate": classifier.learning_rate,
        "epochs": classifier.epochs,
        "batch_size": classifier.batch_size,
        "early_stop_patience": classifier.early_stop_patience,
        "l2_reg": classifier.l2_reg,
        "seed": classifier.seed,
    }
    if metadata:
        config.update(metadata)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return directory


def load_mlx_model(directory: str) -> MlxNNClassifier:
    """Load a ``MlxNNClassifier`` saved by ``save_mlx_model``."""
    import json
    import os

    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    classifier = MlxNNClassifier(
        hidden_dims=tuple(config["hidden_dims"]),
        dropout_prob=config["dropout_prob"],
        use_batch_norm=config.get("use_batch_norm", True),
        class_weight=config.get("class_weight"),
        learning_rate=config.get("learning_rate", 1e-3),
        epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size", 256),
        early_stop_patience=config.get("early_stop_patience", 10),
        l2_reg=config.get("l2_reg", 1e-5),
        seed=config.get("seed", 42),
    )

    n_features = config["n_features"]
    classifier.model_ = HitPredictor(
        n_features=n_features,
        hidden_dims=classifier.hidden_dims,
        dropout_prob=classifier.dropout_prob,
        use_batch_norm=classifier.use_batch_norm,
    )
    classifier.model_.load_weights(
        os.path.join(directory, "model.safetensors"),
        strict=False,
    )
    classifier.scaler_ = joblib.load(os.path.join(directory, "scaler.joblib"))

    return classifier
