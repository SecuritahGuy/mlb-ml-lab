"""Sequence model for MLB hit prediction.

Builds fixed-length windows of recent game stat lines and feeds them
through a GRU → MLP head to predict next-game hit probability.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

SEQUENCE_LEN = 15

STAT_FEATURES = [
    "at_bats", "hits", "walks", "strikeouts",
    "doubles", "triples", "home_runs", "runs", "rbi",
]

N_STATS = len(STAT_FEATURES) + 1  # stats + is_home


def _feat_vec(log: Any) -> list[float]:
    """Build normalised feature vector from a game-log row (dict or object)."""
    vals: list[float] = []
    for k in STAT_FEATURES:
        if isinstance(log, dict):
            v = float(log.get(k, 0))
        else:
            v = float(getattr(log, k, 0))
        vals.append(v)
    if isinstance(log, dict):
        is_home = bool(log.get("is_home", False))
    else:
        is_home = log.is_home
    vals.append(1.0 if is_home else 0.0)
    return vals


def build_sequences(
    game_logs: list[Any],
    targets: list[dict[str, Any]] | None = None,
    seq_len: int = SEQUENCE_LEN,
    stats_mean: np.ndarray | None = None,
    stats_std: np.ndarray | None = None,
    target_col: str = "target_0.5",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Build fixed-length stat sequences from game logs.

    Args:
        game_logs: Chronologically-ordered list of ``PlayerGameLog``.
        targets: Optional parallel list of target dicts (must be same
                 length and order as *game_logs*).
        seq_len: Number of previous games to look back.
        stats_mean: Normalisation mean (shape ``[N_STATS]``).  Computed
                    from data if ``None``.
        stats_std: Normalisation std  (shape ``[N_STATS]``).  Computed
                   from data if ``None``.
        target_col: Which target key to use from targets dicts.

    Returns:
        ``(X, y, mask)`` where:
        - *X* has shape ``(n_sequences, seq_len, N_STATS)``
        - *y* has shape ``(n_sequences,)`` or ``None``
        - *mask* has shape ``(n_sequences, seq_len)`` (1 = real, 0 = pad)
    """
    # Group by (player_id, season), sort by date
    grouped: dict[tuple[int, str], list[tuple[int, Any]]] = defaultdict(list)
    for i, log in enumerate(game_logs):
        key = (log.player_id, str(log.season))
        grouped[key].append((i, log))

    raw_seqs: list[list[list[float]]] = []
    targets_out: list[int] = []
    masks: list[list[float]] = []

    for (pid, season), entries in grouped.items():
        entries.sort(key=lambda e: e[1].date)
        indices = [e[0] for e in entries]
        vecs = [_feat_vec(e[1]) for e in entries]

        # For each game at position `pos`, the sequence is the *previous*
        # ``seq_len`` games (padded for the first few games of the season).
        for pos in range(len(vecs)):
            if pos == 0:
                # No previous games — skip
                continue
            if pos < seq_len:
                seq = [vecs[0]] * (seq_len - pos) + vecs[:pos]
                mask = [0.0] * (seq_len - pos) + [1.0] * pos
            else:
                seq = vecs[pos - seq_len : pos]
                mask = [1.0] * seq_len

            raw_seqs.append(seq)
            masks.append(mask)

            if targets is not None:
                tgt = targets[indices[pos]]
                targets_out.append(tgt.get(target_col, 0))
            else:
                targets_out.append(0)

    # Convert to numpy
    X = np.array(raw_seqs, dtype=np.float32)
    mask_arr = np.array(masks, dtype=np.float32)

    # Normalise per-stat dimension
    flat = X.reshape(-1, N_STATS)
    if stats_mean is None or stats_std is None:
        stats_mean = flat.mean(axis=0)
        stats_std = flat.std(axis=0) + 1e-8
    flat = (flat - stats_mean) / stats_std
    X = flat.reshape(-1, seq_len, N_STATS)

    y = np.array(targets_out, dtype=np.int32) if targets_out else None

    return X, y, mask_arr, stats_mean, stats_std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SequenceHitPredictor(nn.Module):
    """GRU over recent game stat lines → sigmoid output.

    Architecture::

        [B, T, F] → GRU → [B, H] → Linear → [B, 1]
    """

    def __init__(
        self,
        n_stats: int = N_STATS,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(n_stats, hidden_dim, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.gru(x)          # [B, T, H]
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_sequence_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    early_stop_patience: int = 8,
    l2_reg: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[SequenceHitPredictor, dict[str, Any]]:
    """Train a ``SequenceHitPredictor``.

    Args:
        X_train: Shape ``(n, seq_len, n_stats)``.
        y_train: Shape ``(n,)`` binary labels.
        hidden_dim: GRU hidden dimension.
        n_layers: Number of GRU layers.
        dropout: Dropout probability.
        learning_rate: Initial Adam LR (cosine decay to 1%).
        epochs: Maximum epochs.
        batch_size: Mini-batch size.
        early_stop_patience: Epochs without improvement before stopping.
        l2_reg: L2 regularisation strength.
        seed: Random seed.
        verbose: Print epoch losses.

    Returns:
        ``(model, metadata)`` tuple.
    """
    mx.random.seed(seed)
    n_samples = X_train.shape[0]
    n_stats = X_train.shape[2]
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = epochs * steps_per_epoch

    model = SequenceHitPredictor(
        n_stats=n_stats,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    lr_schedule = optim.cosine_decay(
        learning_rate, total_steps, end=learning_rate * 0.01,
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    # Class weighting
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    def loss_fn(x_batch: mx.array, y_batch: mx.array) -> mx.array:
        logits = model(x_batch)
        losses = nn.losses.binary_cross_entropy(logits, y_batch)
        weights = mx.where(y_batch > 0.5, pos_weight, 1.0)
        losses = losses * weights
        base_loss = losses.mean()
        l2 = sum((p * p).sum() for _, p in tree_flatten(model.parameters()))
        return base_loss + l2_reg * l2

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_loss = float("inf")
    patience = 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_samples)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = mx.array(X_shuf[start:end])
            y_batch = mx.array(y_shuf[start:end, np.newaxis])

            loss, grads = loss_and_grad_fn(x_batch, y_batch)
            grads = tree_unflatten([
                (k, mx.clip(v, -5.0, 5.0))
                for k, v in tree_flatten(grads)
            ])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    metadata = {
        "n_stats": n_stats,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs_trained": epoch + 1,
        "batch_size": batch_size,
        "l2_reg": l2_reg,
        "pos_weight": pos_weight,
        "n_train": n_samples,
    }
    return model, metadata


def predict_sequence_model(
    model: SequenceHitPredictor,
    X: np.ndarray,
) -> np.ndarray:
    """Return positive-class probabilities."""
    model.eval()
    n = X.shape[0]
    batch_size = 1024
    all_probas: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = mx.array(X[start:end])
        logits = model(x_batch)
        probas = mx.sigmoid(logits)
        all_probas.append(np.asarray(probas).reshape(-1))

    return np.concatenate(all_probas)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_sequence_model(
    model: SequenceHitPredictor,
    directory: str,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a trained ``SequenceHitPredictor`` to disk.

    Writes::

        {directory}/
            model.safetensors
            stats_mean.npy
            stats_std.npy
            config.json
    """
    os.makedirs(directory, exist_ok=True)

    weights = _flatten_params(model.parameters())
    mx.save_safetensors(os.path.join(directory, "model.safetensors"), weights)
    np.save(os.path.join(directory, "stats_mean.npy"), stats_mean)
    np.save(os.path.join(directory, "stats_std.npy"), stats_std)

    config = {
        "arch": "SequenceHitPredictor",
        "n_stats": N_STATS,
        "hidden_dim": model.gru.hidden_size,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return directory


def load_sequence_model(directory: str) -> tuple[SequenceHitPredictor, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a model saved by ``save_sequence_model``.

    Returns:
        ``(model, stats_mean, stats_std, config)``.
    """
    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)

    n_layers = config.get("n_layers", 2)
    model = SequenceHitPredictor(
        n_stats=config.get("n_stats", N_STATS),
        hidden_dim=config.get("hidden_dim", 64),
        n_layers=n_layers,
        dropout=config.get("dropout", 0.3),
    )
    model.load_weights(os.path.join(directory, "model.safetensors"), strict=False)
    stats_mean = np.load(os.path.join(directory, "stats_mean.npy"))
    stats_std = np.load(os.path.join(directory, "stats_std.npy"))
    return model, stats_mean, stats_std, config


def _flatten_params(params: dict, prefix: str = "") -> dict[str, mx.array]:
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
