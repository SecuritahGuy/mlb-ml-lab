"""Sequence model for MLB hit prediction.

Builds fixed-length windows of recent game stat lines and feeds them
through a GRU → MLP head to predict next-game hit probability.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    _MLX_AVAILABLE = True
    _NNModuleBase: type = nn.Module
except ImportError:  # pragma: no cover
    _MLX_AVAILABLE = False
    _NNModuleBase = object  # type: ignore[assignment,misc]

SEQUENCE_LEN = 15

STAT_FEATURES = [
    "at_bats",
    "hits",
    "walks",
    "strikeouts",
    "doubles",
    "triples",
    "home_runs",
    "runs",
    "rbi",
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

    for (_, _), entries in grouped.items():
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


class SequenceHitPredictor(_NNModuleBase):
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
        out = self.gru(x)  # [B, T, H]
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
    batch_size: int = 2048,
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
        learning_rate,
        total_steps,
        end=learning_rate * 0.01,
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
            grads = tree_unflatten(
                [(k, mx.clip(v, -5.0, 5.0)) for k, v in tree_flatten(grads)]
            )
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


def save_hybrid_model(
    model: HybridHitPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a trained model to disk.

    Writes::

        {directory}/
            model.safetensors
            stats_mean.npy, stats_std.npy
            feat_mean.npy, feat_std.npy
            config.json
    """
    os.makedirs(directory, exist_ok=True)

    weights = _flatten_params(model.parameters())
    mx.save_safetensors(os.path.join(directory, "model.safetensors"), weights)

    for name, arr in [
        ("stats_mean", stats_mean),
        ("stats_std", stats_std),
        ("feat_mean", feat_mean),
        ("feat_std", feat_std),
    ]:
        if arr is not None:
            np.save(os.path.join(directory, f"{name}.npy"), arr)

    config = {
        "arch": "HybridHitPredictor",
        "n_stats": N_STATS,
        "hidden_dim": model.gru.hidden_size,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return directory


def load_hybrid_model(
    directory: str,
) -> tuple[
    HybridHitPredictor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
    """Load a model saved by ``save_hybrid_model``.

    Returns:
        ``(model, stats_mean, stats_std, feat_mean, feat_std, config)``.
    """
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    model = HybridHitPredictor(
        n_stats=config.get("n_stats", N_STATS),
        n_context=config.get("n_context", 64),
        hidden_dim=config.get("hidden_dim", 64),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
    )
    model.load_weights(os.path.join(directory, "model.safetensors"), strict=False)

    def _load_arr(name: str) -> np.ndarray:
        path = os.path.join(directory, name)
        return np.load(path) if os.path.isfile(path) else np.array([])

    stats_mean = _load_arr("stats_mean.npy")
    stats_std = _load_arr("stats_std.npy")
    feat_mean = _load_arr("feat_mean.npy")
    feat_std = _load_arr("feat_std.npy")
    return model, stats_mean, stats_std, feat_mean, feat_std, config


def load_sequence_model(
    directory: str,
) -> tuple[SequenceHitPredictor, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a model saved by ``save_sequence_model``.

    Returns:
        ``(model, stats_mean, stats_std, config)``.
    """
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
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


# ---------------------------------------------------------------------------
# Hybrid model: GRU + context features
# ---------------------------------------------------------------------------


class HybridHitPredictor(_NNModuleBase):
    """GRU over stat sequences + MLP over context features → binary prediction.

    Architecture::

        [B,T,F] → GRU → [B,H]  ─┐
                                 ├→ Concat → [B,2H] → Linear → [B,1]
        [B,C] → [Gate] → Linear+ReLU ────┘

    When ``use_gating=True``, a learned sigmoid gate is applied to context
    features before the MLP, performing soft feature selection.
    """

    def __init__(
        self,
        n_stats: int = N_STATS,
        n_context: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        use_gating: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(n_stats, hidden_dim, n_layers)
        self.use_gating = use_gating
        if use_gating:
            self.feature_gate = nn.Linear(n_context, n_context)
        self.context_net = nn.Sequential(
            nn.Linear(n_context, hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def __call__(self, seq: mx.array, ctx: mx.array) -> mx.array:
        out = self.gru(seq)
        last = out[:, -1, :]
        last = self.dropout(last)
        if self.use_gating:
            gates = mx.sigmoid(self.feature_gate(ctx))
            ctx = ctx * gates
        ctx_emb = self.context_net(ctx)
        combined = mx.concatenate([last, ctx_emb], axis=-1)
        return self.head(combined)


def build_hybrid_sequences(
    game_logs: list[Any],
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    seq_len: int = SEQUENCE_LEN,
    stats_mean: np.ndarray | None = None,
    stats_std: np.ndarray | None = None,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
    target_col: str = "target_0.5",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sequences + context features.

    Returns:
        ``(X_seq, X_ctx, y, stats_mean, stats_std, feat_mean, feat_std)``.
    """
    # Index feature rows by (player_id, game_pk)
    feat_index: dict[tuple[int, int], dict[str, Any]] = {}
    for fr in feature_matrix:
        feat_index[(fr["player_id"], fr["game_pk"])] = fr

    # Build a target index by (player_id, game_pk)
    target_index: dict[tuple[int, int], dict[str, Any]] = {}
    for t in targets:
        target_index[(t["player_id"], t["game_pk"])] = t

    # Group logs by (player_id, season), sort by date
    grouped: dict[tuple[int, str], list[tuple[int, Any]]] = defaultdict(list)
    for i, log in enumerate(game_logs):
        pid = log.player_id if hasattr(log, "player_id") else log["player_id"]
        season = (
            str(log.season) if hasattr(log, "season") else str(log.get("season", ""))
        )
        grouped[(pid, season)].append((i, log))

    seq_list: list[np.ndarray] = []
    ctx_list: list[np.ndarray] = []
    y_list: list[int] = []

    # Determine numeric context feature columns (check first 100 rows)
    ctx_cols: list[str] | None = None
    _excluded = {"player_id", "game_pk", "date"}
    _numeric_types = (int, float)

    def _ctx_is_numeric(fr: dict, k: str) -> bool:
        v = fr.get(k)
        return isinstance(v, _numeric_types) or v is None

    # Scan first 100 feature rows to identify numeric columns
    _sample_cols = None
    for fr in feature_matrix[:100]:
        if _sample_cols is None:
            _sample_cols = set(fr.keys()) - _excluded
        _sample_cols = {k for k in _sample_cols if k in fr and _ctx_is_numeric(fr, k)}
    ctx_cols = sorted(_sample_cols) if _sample_cols else []

    for (_, _), entries in grouped.items():
        entries.sort(key=lambda e: e[1].date if hasattr(e[1], "date") else e[1]["date"])
        indices = [e[0] for e in entries]
        vecs = [_feat_vec(e[1]) for e in entries]

        for pos in range(seq_len, len(vecs)):
            idx = indices[pos]
            log = game_logs[idx]
            log_pid = log.player_id if hasattr(log, "player_id") else log["player_id"]
            log_gpk = log.game_pk if hasattr(log, "game_pk") else log["game_pk"]

            feat_row = feat_index.get((log_pid, log_gpk))
            if feat_row is None:
                continue

            target_row = target_index.get((log_pid, log_gpk))
            if target_row is None:
                continue

            seq = vecs[pos - seq_len : pos]
            seq_list.append(np.array(seq, dtype=np.float32))

            ctx_vec = np.array([feat_row[c] or 0.0 for c in ctx_cols], dtype=np.float32)
            ctx_list.append(ctx_vec)

            y_list.append(target_row.get(target_col, 0))

    X_seq = np.stack(seq_list)
    X_ctx = np.stack(ctx_list)
    y = np.array(y_list, dtype=np.int32)

    # Normalise stat features
    flat_seq = X_seq.reshape(-1, N_STATS)
    if stats_mean is None:
        stats_mean = flat_seq.mean(axis=0)
        stats_std = flat_seq.std(axis=0) + 1e-8
    flat_seq = (flat_seq - stats_mean) / stats_std
    X_seq = flat_seq.reshape(-1, seq_len, N_STATS)

    # Normalise context features
    if feat_mean is None:
        feat_mean = X_ctx.mean(axis=0)
        feat_std = X_ctx.std(axis=0) + 1e-8
        feat_std[feat_std == 0] = 1.0  # avoid division by zero
    X_ctx = (X_ctx - feat_mean) / feat_std
    X_ctx = np.nan_to_num(X_ctx, nan=0.0)

    return X_seq, X_ctx, y, stats_mean, stats_std, feat_mean, feat_std


def train_hybrid_model(
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
    use_gating: bool = False,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 2048,
    early_stop_patience: int = 8,
    l2_reg: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[HybridHitPredictor, dict[str, Any]]:
    """Train a ``HybridHitPredictor``."""
    mx.random.seed(seed)
    n_samples = X_seq.shape[0]
    n_stats = X_seq.shape[2]
    n_context = X_ctx.shape[1]
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = epochs * steps_per_epoch

    model = HybridHitPredictor(
        n_stats=n_stats,
        n_context=n_context,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        use_gating=use_gating,
    )

    lr_schedule = optim.cosine_decay(
        learning_rate,
        total_steps,
        end=learning_rate * 0.01,
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    def loss_fn(xs: mx.array, xc: mx.array, yb: mx.array) -> mx.array:
        logits = model(xs, xc)
        losses = nn.losses.binary_cross_entropy(logits, yb)
        weights = mx.where(yb > 0.5, pos_weight, 1.0)
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
        Xs_shuf = X_seq[perm]
        Xc_shuf = X_ctx[perm]
        y_shuf = y[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xs_batch = mx.array(Xs_shuf[start:end])
            xc_batch = mx.array(Xc_shuf[start:end])
            y_batch = mx.array(y_shuf[start:end, np.newaxis])

            loss, grads = loss_and_grad_fn(xs_batch, xc_batch, y_batch)
            grads = tree_unflatten(
                [(k, mx.clip(v, -5.0, 5.0)) for k, v in tree_flatten(grads)]
            )
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
        "arch": "HybridHitPredictor",
        "n_stats": n_stats,
        "n_context": n_context,
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


def predict_hybrid_model(
    model: HybridHitPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> np.ndarray:
    """Return positive-class probabilities."""
    model.eval()
    n = X_seq.shape[0]
    batch_size = 1024
    all_probas: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = mx.array(X_seq[start:end])
        xc = mx.array(X_ctx[start:end])
        logits = model(xs, xc)
        probas = mx.sigmoid(logits)
        all_probas.append(np.asarray(probas).reshape(-1))

    return np.concatenate(all_probas)


# ---------------------------------------------------------------------------
# Multi-task model: shared encoder, two heads
# ---------------------------------------------------------------------------


class MultiTaskHybridPredictor(_NNModuleBase):
    """Shared GRU+MLP encoder with two output heads.

    Architecture::

        [B,T,F] → GRU → [B,H]  ─┐
                                 ├→ Concat → head_05 → [B,1]
        [B,C] → Linear+ReLU ────┘
                                           head_15 → [B,1]
    """

    def __init__(
        self,
        n_stats: int = N_STATS,
        n_context: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        context_depth: int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(n_stats, hidden_dim, n_layers)
        if context_depth >= 2:
            self.context_net = nn.Sequential(
                nn.Linear(n_context, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )
        else:
            self.context_net = nn.Sequential(
                nn.Linear(n_context, hidden_dim),
                nn.ReLU(),
            )
        self.dropout = nn.Dropout(dropout)
        self.head_05 = nn.Linear(hidden_dim * 2, 1)
        self.head_15 = nn.Linear(hidden_dim * 2, 1)

    def __call__(self, seq: mx.array, ctx: mx.array) -> tuple[mx.array, mx.array]:
        out = self.gru(seq)
        last = out[:, -1, :]
        last = self.dropout(last)
        ctx_emb = self.context_net(ctx)
        combined = mx.concatenate([last, ctx_emb], axis=-1)
        return self.head_05(combined), self.head_15(combined)


def build_hybrid_mt_sequences(
    game_logs: list[Any],
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    seq_len: int = SEQUENCE_LEN,
    stats_mean: np.ndarray | None = None,
    stats_std: np.ndarray | None = None,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build sequences + context features for multi-task learning.

    Returns:
        ``(X_seq, X_ctx, y_05, y_15, stats_mean, stats_std, feat_mean,
        feat_std)``.
    """
    feat_index: dict[tuple[int, int], dict[str, Any]] = {}
    for fr in feature_matrix:
        feat_index[(fr["player_id"], fr["game_pk"])] = fr

    target_index: dict[tuple[int, int], dict[str, Any]] = {}
    for t in targets:
        target_index[(t["player_id"], t["game_pk"])] = t

    grouped: dict[tuple[int, str], list[tuple[int, Any]]] = defaultdict(list)
    for i, log in enumerate(game_logs):
        pid = log.player_id if hasattr(log, "player_id") else log["player_id"]
        season = (
            str(log.season) if hasattr(log, "season") else str(log.get("season", ""))
        )
        grouped[(pid, season)].append((i, log))

    seq_list: list[np.ndarray] = []
    ctx_list: list[np.ndarray] = []
    y05_list: list[int] = []
    y15_list: list[int] = []

    ctx_cols: list[str] | None = None
    _excluded = {"player_id", "game_pk", "date"}
    _numeric_types = (int, float)

    def _ctx_is_numeric(fr: dict, k: str) -> bool:
        v = fr.get(k)
        return isinstance(v, _numeric_types) or v is None

    _sample_cols = None
    for fr in feature_matrix[:100]:
        if _sample_cols is None:
            _sample_cols = set(fr.keys()) - _excluded
        _sample_cols = {k for k in _sample_cols if k in fr and _ctx_is_numeric(fr, k)}
    ctx_cols = sorted(_sample_cols) if _sample_cols else []

    for (_, _), entries in grouped.items():
        entries.sort(key=lambda e: e[1].date if hasattr(e[1], "date") else e[1]["date"])
        indices = [e[0] for e in entries]
        vecs = [_feat_vec(e[1]) for e in entries]

        for pos in range(seq_len, len(vecs)):
            idx = indices[pos]
            log = game_logs[idx]
            log_pid = log.player_id if hasattr(log, "player_id") else log["player_id"]
            log_gpk = log.game_pk if hasattr(log, "game_pk") else log["game_pk"]

            feat_row = feat_index.get((log_pid, log_gpk))
            if feat_row is None:
                continue

            target_row = target_index.get((log_pid, log_gpk))
            if target_row is None:
                continue

            seq = vecs[pos - seq_len : pos]
            seq_list.append(np.array(seq, dtype=np.float32))
            ctx_vec = np.array([feat_row[c] or 0.0 for c in ctx_cols], dtype=np.float32)
            ctx_list.append(ctx_vec)
            y05_list.append(target_row.get("target_0.5", 0))
            y15_list.append(target_row.get("target_1.5", 0))

    X_seq = np.stack(seq_list)
    X_ctx = np.stack(ctx_list)
    y_05 = np.array(y05_list, dtype=np.int32)
    y_15 = np.array(y15_list, dtype=np.int32)

    flat_seq = X_seq.reshape(-1, N_STATS)
    if stats_mean is None:
        stats_mean = flat_seq.mean(axis=0)
        stats_std = flat_seq.std(axis=0) + 1e-8
    flat_seq = (flat_seq - stats_mean) / stats_std
    X_seq = flat_seq.reshape(-1, seq_len, N_STATS)

    if feat_mean is None:
        feat_mean = X_ctx.mean(axis=0)
        feat_std = X_ctx.std(axis=0) + 1e-8
        feat_std[feat_std == 0] = 1.0
    X_ctx = (X_ctx - feat_mean) / feat_std
    X_ctx = np.nan_to_num(X_ctx, nan=0.0)

    return X_seq, X_ctx, y_05, y_15, stats_mean, stats_std, feat_mean, feat_std


def train_multi_task_model(
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
    y_05: np.ndarray,
    y_15: np.ndarray,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
    context_depth: int = 1,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    early_stop_patience: int = 8,
    l2_reg: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[MultiTaskHybridPredictor, dict[str, Any]]:
    """Train a ``MultiTaskHybridPredictor`` with two losses."""
    mx.random.seed(seed)
    n_samples = X_seq.shape[0]
    n_stats = X_seq.shape[2]
    n_context = X_ctx.shape[1]
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = epochs * steps_per_epoch

    model = MultiTaskHybridPredictor(
        n_stats=n_stats,
        n_context=n_context,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        context_depth=context_depth,
    )

    lr_schedule = optim.cosine_decay(
        learning_rate,
        total_steps,
        end=learning_rate * 0.01,
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    # Per-target class weights
    n_pos_05 = int(y_05.sum())
    n_neg_05 = int(len(y_05) - n_pos_05)
    pw_05 = n_neg_05 / n_pos_05 if n_pos_05 > 0 else 1.0

    n_pos_15 = int(y_15.sum())
    n_neg_15 = int(len(y_15) - n_pos_15)
    pw_15 = n_neg_15 / n_pos_15 if n_pos_15 > 0 else 1.0

    def loss_fn(
        xs: mx.array,
        xc: mx.array,
        y05: mx.array,
        y15: mx.array,
    ) -> mx.array:
        logits_05, logits_15 = model(xs, xc)
        loss_05 = nn.losses.binary_cross_entropy(logits_05, y05)
        loss_05 = loss_05 * mx.where(y05 > 0.5, pw_05, 1.0)
        loss_15 = nn.losses.binary_cross_entropy(logits_15, y15)
        loss_15 = loss_15 * mx.where(y15 > 0.5, pw_15, 1.0)
        base_loss = loss_05.mean() + loss_15.mean()
        l2 = sum((p * p).sum() for _, p in tree_flatten(model.parameters()))
        return base_loss + l2_reg * l2

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_loss = float("inf")
    patience = 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_samples)
        Xs_shuf = X_seq[perm]
        Xc_shuf = X_ctx[perm]
        y05_shuf = y_05[perm]
        y15_shuf = y_15[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xs_batch = mx.array(Xs_shuf[start:end])
            xc_batch = mx.array(Xc_shuf[start:end])
            y05_batch = mx.array(y05_shuf[start:end, np.newaxis])
            y15_batch = mx.array(y15_shuf[start:end, np.newaxis])

            loss, grads = loss_and_grad_fn(xs_batch, xc_batch, y05_batch, y15_batch)
            grads = tree_unflatten(
                [(k, mx.clip(v, -5.0, 5.0)) for k, v in tree_flatten(grads)]
            )
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
        "arch": "MultiTaskHybridPredictor",
        "n_stats": n_stats,
        "n_context": n_context,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "context_depth": context_depth,
        "learning_rate": learning_rate,
        "epochs_trained": epoch + 1,
        "batch_size": batch_size,
        "l2_reg": l2_reg,
        "n_train": n_samples,
    }
    return model, metadata


def predict_multi_task_model(
    model: MultiTaskHybridPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (proba_05, proba_15) for the multi-task model."""
    model.eval()
    n = X_seq.shape[0]
    batch_size = 1024
    p05_list: list[np.ndarray] = []
    p15_list: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = mx.array(X_seq[start:end])
        xc = mx.array(X_ctx[start:end])
        logits_05, logits_15 = model(xs, xc)
        p05_list.append(np.asarray(mx.sigmoid(logits_05)).reshape(-1))
        p15_list.append(np.asarray(mx.sigmoid(logits_15)).reshape(-1))

    return np.concatenate(p05_list), np.concatenate(p15_list)


def save_multi_task_model(
    model: MultiTaskHybridPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a trained multi-task model to disk."""
    os.makedirs(directory, exist_ok=True)
    weights = _flatten_params(model.parameters())
    mx.save_safetensors(os.path.join(directory, "model.safetensors"), weights)
    for name, arr in [
        ("stats_mean", stats_mean),
        ("stats_std", stats_std),
        ("feat_mean", feat_mean),
        ("feat_std", feat_std),
    ]:
        if arr is not None:
            np.save(os.path.join(directory, f"{name}.npy"), arr)
    config = {
        "arch": "MultiTaskHybridPredictor",
        "n_stats": N_STATS,
        "hidden_dim": model.gru.hidden_size,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return directory


def load_multi_task_model(
    directory: str,
) -> tuple[
    MultiTaskHybridPredictor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Load a multi-task model saved by ``save_multi_task_model``."""
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    model = MultiTaskHybridPredictor(
        n_stats=config.get("n_stats", N_STATS),
        n_context=config.get("n_context", 64),
        hidden_dim=config.get("hidden_dim", 64),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
        context_depth=config.get("context_depth", 1),
    )
    model.load_weights(os.path.join(directory, "model.safetensors"), strict=False)

    def _load_arr(name: str) -> np.ndarray:
        path = os.path.join(directory, name)
        return np.load(path) if os.path.isfile(path) else np.array([])

    stats_mean = _load_arr("stats_mean.npy")
    stats_std = _load_arr("stats_std.npy")
    feat_mean = _load_arr("feat_mean.npy")
    feat_std = _load_arr("feat_std.npy")
    return model, stats_mean, stats_std, feat_mean, feat_std, config


# ---------------------------------------------------------------------------
# DCN (Deep & Cross Network) model
# ---------------------------------------------------------------------------


class CrossNetwork(_NNModuleBase):
    """Cross network for explicit bounded-degree feature interactions.

    Each layer::

        x_{l+1} = x_0 ⊙ (W_l @ x_l + b_l) + x_l
    """

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self._num = num_layers
        for i in range(num_layers):
            setattr(self, f"cross_{i}", nn.Linear(dim, dim, bias=True))

    def __call__(self, x: mx.array) -> mx.array:
        x_0 = x
        x_l = x_0
        for i in range(self._num):
            w = getattr(self, f"cross_{i}")
            x_l = x_0 * w(x_l) + x_l
        return x_l


class DCNMultiTaskPredictor(_NNModuleBase):
    """GRU over sequences + lightweight DCN over context → two heads.

    Projects 91-dim context down to *cross_dim* first, then applies a
    small cross network at that dimension — no deep tower.  This keeps
    parameter count comparable to the original Linear+ReLU context path
    while adding explicit feature interactions.

    Architecture::

        [B,T,F] → GRU → [B,H]  ─┐
                                 ├→ Concat → head_05 → [B,1]
        [B,C] → Proj(cross_dim) ─┤
              → CrossNet ────────┘
                                           head_15 → [B,1]
    """

    def __init__(
        self,
        n_stats: int = N_STATS,
        n_context: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        cross_dim: int = 32,
        num_cross_layers: int = 2,
    ):
        super().__init__()
        self.gru = nn.GRU(n_stats, hidden_dim, n_layers)
        self.ctx_proj = nn.Linear(n_context, cross_dim)
        self.cross_net = CrossNetwork(cross_dim, num_cross_layers)
        self.dropout = nn.Dropout(dropout)
        self.head_05 = nn.Linear(hidden_dim + cross_dim, 1)
        self.head_15 = nn.Linear(hidden_dim + cross_dim, 1)

    def __call__(
        self,
        seq: mx.array,
        ctx: mx.array,
    ) -> tuple[mx.array, mx.array]:
        out = self.gru(seq)
        last = out[:, -1, :]
        last = self.dropout(last)
        ctx_emb = self.cross_net(self.ctx_proj(ctx))
        combined = mx.concatenate([last, ctx_emb], axis=-1)
        return self.head_05(combined), self.head_15(combined)


def train_dcn_multi_task_model(
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
    y_05: np.ndarray,
    y_15: np.ndarray,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.3,
    cross_dim: int = 32,
    num_cross_layers: int = 2,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    early_stop_patience: int = 8,
    l2_reg: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[DCNMultiTaskPredictor, dict[str, Any]]:
    """Train a ``DCNMultiTaskPredictor`` with two losses."""
    mx.random.seed(seed)
    n_samples = X_seq.shape[0]
    n_stats = X_seq.shape[2]
    n_context = X_ctx.shape[1]
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = epochs * steps_per_epoch

    model = DCNMultiTaskPredictor(
        n_stats=n_stats,
        n_context=n_context,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        cross_dim=cross_dim,
        num_cross_layers=num_cross_layers,
    )

    lr_schedule = optim.cosine_decay(
        learning_rate,
        total_steps,
        end=learning_rate * 0.01,
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    n_pos_05 = int(y_05.sum())
    n_neg_05 = int(len(y_05) - n_pos_05)
    pw_05 = n_neg_05 / n_pos_05 if n_pos_05 > 0 else 1.0

    n_pos_15 = int(y_15.sum())
    n_neg_15 = int(len(y_15) - n_pos_15)
    pw_15 = n_neg_15 / n_pos_15 if n_pos_15 > 0 else 1.0

    def loss_fn(
        xs: mx.array,
        xc: mx.array,
        y05: mx.array,
        y15: mx.array,
    ) -> mx.array:
        logits_05, logits_15 = model(xs, xc)
        loss_05 = nn.losses.binary_cross_entropy(logits_05, y05)
        loss_05 = loss_05 * mx.where(y05 > 0.5, pw_05, 1.0)
        loss_15 = nn.losses.binary_cross_entropy(logits_15, y15)
        loss_15 = loss_15 * mx.where(y15 > 0.5, pw_15, 1.0)
        base_loss = loss_05.mean() + loss_15.mean()
        l2 = sum((p * p).sum() for _, p in tree_flatten(model.parameters()))
        return base_loss + l2_reg * l2

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_loss = float("inf")
    patience = 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_samples)
        Xs_shuf = X_seq[perm]
        Xc_shuf = X_ctx[perm]
        y05_shuf = y_05[perm]
        y15_shuf = y_15[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xs_batch = mx.array(Xs_shuf[start:end])
            xc_batch = mx.array(Xc_shuf[start:end])
            y05_batch = mx.array(y05_shuf[start:end, np.newaxis])
            y15_batch = mx.array(y15_shuf[start:end, np.newaxis])

            loss, grads = loss_and_grad_fn(xs_batch, xc_batch, y05_batch, y15_batch)
            grads = tree_unflatten(
                [(k, mx.clip(v, -5.0, 5.0)) for k, v in tree_flatten(grads)]
            )
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
        "arch": "DCNMultiTaskPredictor",
        "n_stats": n_stats,
        "n_context": n_context,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "cross_dim": cross_dim,
        "num_cross_layers": num_cross_layers,
        "learning_rate": learning_rate,
        "epochs_trained": epoch + 1,
        "batch_size": batch_size,
        "l2_reg": l2_reg,
        "n_train": n_samples,
    }
    return model, metadata


def predict_dcn_multi_task_model(
    model: DCNMultiTaskPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (proba_05, proba_15)."""
    model.eval()
    n = X_seq.shape[0]
    batch_size = 1024
    p05_list: list[np.ndarray] = []
    p15_list: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = mx.array(X_seq[start:end])
        xc = mx.array(X_ctx[start:end])
        logits_05, logits_15 = model(xs, xc)
        p05_list.append(np.asarray(mx.sigmoid(logits_05)).reshape(-1))
        p15_list.append(np.asarray(mx.sigmoid(logits_15)).reshape(-1))

    return np.concatenate(p05_list), np.concatenate(p15_list)


def save_dcn_model(
    model: DCNMultiTaskPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a trained DCN multi-task model to disk."""
    os.makedirs(directory, exist_ok=True)
    weights = _flatten_params(model.parameters())
    mx.save_safetensors(os.path.join(directory, "model.safetensors"), weights)
    for name, arr in [
        ("stats_mean", stats_mean),
        ("stats_std", stats_std),
        ("feat_mean", feat_mean),
        ("feat_std", feat_std),
    ]:
        if arr is not None:
            np.save(os.path.join(directory, f"{name}.npy"), arr)
    config = {
        "arch": "DCNMultiTaskPredictor",
        "n_stats": N_STATS,
        "hidden_dim": model.gru.hidden_size,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return directory


def load_dcn_model(
    directory: str,
) -> tuple[
    DCNMultiTaskPredictor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Load a DCN multi-task model saved by ``save_dcn_model``."""
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    model = DCNMultiTaskPredictor(
        n_stats=config.get("n_stats", N_STATS),
        n_context=config.get("n_context", 64),
        hidden_dim=config.get("hidden_dim", 64),
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
        cross_dim=config.get("cross_dim", 32),
        num_cross_layers=config.get("num_cross_layers", 2),
    )
    model.load_weights(os.path.join(directory, "model.safetensors"), strict=False)

    def _load_arr(name: str) -> np.ndarray:
        path = os.path.join(directory, name)
        return np.load(path) if os.path.isfile(path) else np.array([])

    stats_mean = _load_arr("stats_mean.npy")
    stats_std = _load_arr("stats_std.npy")
    feat_mean = _load_arr("feat_mean.npy")
    feat_std = _load_arr("feat_std.npy")
    return model, stats_mean, stats_std, feat_mean, feat_std, config


# ---------------------------------------------------------------------------
# Transformer encoder model (replaces GRU)
# ---------------------------------------------------------------------------


class PositionalEncoding(_NNModuleBase):
    """Learned positional encoding for a fixed-length sequence."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = mx.random.normal((max_len, d_model)) * 0.02

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.embedding[: x.shape[1]]


class TransformerEncoder(_NNModuleBase):
    """Lightweight transformer encoder over game stat sequences.

    Args:
        n_stats: Number of stat features per game.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_stats: int = N_STATS,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self._num_layers = num_layers
        self.input_proj = nn.Linear(n_stats, d_model)
        self.pos_enc = PositionalEncoding(SEQUENCE_LEN, d_model)
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            setattr(self, f"attn_{i}", nn.MultiHeadAttention(d_model, nhead))
            setattr(self, f"norm1_{i}", nn.LayerNorm(d_model))
            setattr(
                self,
                f"ffn_{i}",
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model),
                ),
            )
            setattr(self, f"norm2_{i}", nn.LayerNorm(d_model))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for i in range(self._num_layers):
            attn = getattr(self, f"attn_{i}")
            norm1 = getattr(self, f"norm1_{i}")
            ffn = getattr(self, f"ffn_{i}")
            norm2 = getattr(self, f"norm2_{i}")

            residual = x
            x = attn(x, x, x)
            x = norm1(residual + x)

            residual = x
            x = ffn(x)
            x = norm2(residual + x)

        return x.mean(axis=1)


class TransformerMultiTaskPredictor(_NNModuleBase):
    """Transformer encoder over sequences + MLP over context → two heads.

    Architecture::

        [B,T,F] → Transformer → [B,D]  ─┐
                                         ├→ Concat → head_05 → [B,1]
        [B,C] → Linear+ReLU ────────────┘
                                                   head_15 → [B,1]
    """

    def __init__(
        self,
        n_stats: int = N_STATS,
        n_context: int = 64,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.transformer = TransformerEncoder(
            n_stats=n_stats,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.context_net = nn.Sequential(
            nn.Linear(n_context, d_model),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head_05 = nn.Linear(d_model * 2, 1)
        self.head_15 = nn.Linear(d_model * 2, 1)

    def __call__(
        self,
        seq: mx.array,
        ctx: mx.array,
    ) -> tuple[mx.array, mx.array]:
        seq_emb = self.transformer(seq)
        ctx_emb = self.context_net(ctx)
        combined = mx.concatenate([seq_emb, ctx_emb], axis=-1)
        combined = self.dropout(combined)
        return self.head_05(combined), self.head_15(combined)


def train_transformer_multi_task_model(
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
    y_05: np.ndarray,
    y_15: np.ndarray,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    early_stop_patience: int = 8,
    l2_reg: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[TransformerMultiTaskPredictor, dict[str, Any]]:
    """Train a ``TransformerMultiTaskPredictor`` with two losses."""
    mx.random.seed(seed)
    n_samples = X_seq.shape[0]
    n_stats = X_seq.shape[2]
    n_context = X_ctx.shape[1]
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = epochs * steps_per_epoch

    model = TransformerMultiTaskPredictor(
        n_stats=n_stats,
        n_context=n_context,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )

    lr_schedule = optim.cosine_decay(
        learning_rate,
        total_steps,
        end=learning_rate * 0.01,
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    n_pos_05 = int(y_05.sum())
    n_neg_05 = int(len(y_05) - n_pos_05)
    pw_05 = n_neg_05 / n_pos_05 if n_pos_05 > 0 else 1.0
    n_pos_15 = int(y_15.sum())
    n_neg_15 = int(len(y_15) - n_pos_15)
    pw_15 = n_neg_15 / n_pos_15 if n_pos_15 > 0 else 1.0

    def loss_fn(
        xs: mx.array,
        xc: mx.array,
        y05: mx.array,
        y15: mx.array,
    ) -> mx.array:
        logits_05, logits_15 = model(xs, xc)
        loss_05 = nn.losses.binary_cross_entropy(logits_05, y05)
        loss_05 = loss_05 * mx.where(y05 > 0.5, pw_05, 1.0)
        loss_15 = nn.losses.binary_cross_entropy(logits_15, y15)
        loss_15 = loss_15 * mx.where(y15 > 0.5, pw_15, 1.0)
        base_loss = loss_05.mean() + loss_15.mean()
        l2 = sum((p * p).sum() for _, p in tree_flatten(model.parameters()))
        return base_loss + l2_reg * l2

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_loss = float("inf")
    patience = 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_samples)
        Xs_shuf = X_seq[perm]
        Xc_shuf = X_ctx[perm]
        y05_shuf = y_05[perm]
        y15_shuf = y_15[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xs_batch = mx.array(Xs_shuf[start:end])
            xc_batch = mx.array(Xc_shuf[start:end])
            y05_batch = mx.array(y05_shuf[start:end, np.newaxis])
            y15_batch = mx.array(y15_shuf[start:end, np.newaxis])

            loss, grads = loss_and_grad_fn(xs_batch, xc_batch, y05_batch, y15_batch)
            grads = tree_unflatten(
                [(k, mx.clip(v, -5.0, 5.0)) for k, v in tree_flatten(grads)]
            )
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
        "arch": "TransformerMultiTaskPredictor",
        "n_stats": n_stats,
        "n_context": n_context,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs_trained": epoch + 1,
        "batch_size": batch_size,
        "l2_reg": l2_reg,
        "n_train": n_samples,
    }
    return model, metadata


def predict_transformer_multi_task_model(
    model: TransformerMultiTaskPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (proba_05, proba_15)."""
    model.eval()
    n = X_seq.shape[0]
    batch_size = 1024
    p05_list: list[np.ndarray] = []
    p15_list: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = mx.array(X_seq[start:end])
        xc = mx.array(X_ctx[start:end])
        logits_05, logits_15 = model(xs, xc)
        p05_list.append(np.asarray(mx.sigmoid(logits_05)).reshape(-1))
        p15_list.append(np.asarray(mx.sigmoid(logits_15)).reshape(-1))

    return np.concatenate(p05_list), np.concatenate(p15_list)


def save_transformer_model(
    model: TransformerMultiTaskPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a trained transformer multi-task model to disk."""
    os.makedirs(directory, exist_ok=True)
    weights = _flatten_params(model.parameters())
    mx.save_safetensors(os.path.join(directory, "model.safetensors"), weights)
    for name, arr in [
        ("stats_mean", stats_mean),
        ("stats_std", stats_std),
        ("feat_mean", feat_mean),
        ("feat_std", feat_std),
    ]:
        if arr is not None:
            np.save(os.path.join(directory, f"{name}.npy"), arr)
    config = {
        "arch": "TransformerMultiTaskPredictor",
        "n_stats": N_STATS,
        "d_model": model.transformer.input_proj.weight.shape[0],
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return directory


def load_transformer_model(
    directory: str,
) -> tuple[
    TransformerMultiTaskPredictor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Load a transformer multi-task model saved by ``save_transformer_model``."""
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    model = TransformerMultiTaskPredictor(
        n_stats=config.get("n_stats", N_STATS),
        n_context=config.get("n_context", 64),
        d_model=config.get("d_model", 32),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
    )
    model.load_weights(os.path.join(directory, "model.safetensors"), strict=False)

    def _load_arr(name: str) -> np.ndarray:
        path = os.path.join(directory, name)
        return np.load(path) if os.path.isfile(path) else np.array([])

    stats_mean = _load_arr("stats_mean.npy")
    stats_std = _load_arr("stats_std.npy")
    feat_mean = _load_arr("feat_mean.npy")
    feat_std = _load_arr("feat_std.npy")
    return model, stats_mean, stats_std, feat_mean, feat_std, config
