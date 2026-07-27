"""Sequence model for MLB hit prediction.

Builds fixed-length windows of recent game stat lines and feeds them
through a GRU / Transformer → MLP head to predict next-game hit probability.
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _feat_vec(log: Any) -> list[float]:
    vals: list[float] = []
    for k in STAT_FEATURES:
        v = float(log[k]) if isinstance(log, dict) else float(getattr(log, k, 0))
        vals.append(v)
    is_home = bool(log.get("is_home", False)) if isinstance(log, dict) else log.is_home
    vals.append(1.0 if is_home else 0.0)
    return vals


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
# Shared training loop
# ---------------------------------------------------------------------------


def _train_mlx_model(
    model: nn.Module,
    X_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    loss_fn_builder: Any,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 2048,
    early_stop_patience: int = 8,
    l2_reg: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    mx.random.seed(seed)
    n_samples = X_arrays[0].shape[0]
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = epochs * steps_per_epoch

    lr_schedule = optim.cosine_decay(
        learning_rate, total_steps, end=learning_rate * 0.01
    )
    optimizer = optim.Adam(learning_rate=lr_schedule)

    loss_fn = loss_fn_builder(model, y_arrays, l2_reg)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    best_loss = float("inf")
    patience = 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_samples)
        shuf = [a[perm] for a in X_arrays] + [a[perm] for a in y_arrays]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batches = [mx.array(s[start:end]) for s in shuf]

            loss, grads = loss_and_grad_fn(*batches)
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

    return {
        "epochs_trained": epoch + 1,
        "batch_size": batch_size,
        "l2_reg": l2_reg,
        "n_train": n_samples,
        "learning_rate": learning_rate,
    }


# ---------------------------------------------------------------------------
# Shared predict loop
# ---------------------------------------------------------------------------


def _predict_mlx_model(
    model: nn.Module,
    X_arrays: list[np.ndarray],
    n_outputs: int = 1,
    batch_size: int = 1024,
) -> list[np.ndarray]:
    model.eval()
    n = X_arrays[0].shape[0]
    outputs: list[list[np.ndarray]] = [[] for _ in range(n_outputs)]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches = [mx.array(a[start:end]) for a in X_arrays]

        result = model(*batches)
        if n_outputs == 1:
            result = (result,)
        for i in range(n_outputs):
            probas = mx.sigmoid(result[i])
            outputs[i].append(np.asarray(probas).reshape(-1))

    return [np.concatenate(o) for o in outputs]


# ---------------------------------------------------------------------------
# Shared persistence
# ---------------------------------------------------------------------------


def _save_model(
    model: nn.Module,
    directory: str,
    arrays: dict[str, np.ndarray | None],
    config: dict[str, Any],
) -> str:
    os.makedirs(directory, exist_ok=True)
    weights = _flatten_params(model.parameters())
    mx.save_safetensors(os.path.join(directory, "model.safetensors"), weights)
    for name, arr in arrays.items():
        if arr is not None:
            np.save(os.path.join(directory, f"{name}.npy"), arr)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return directory


def _load_model_arrays(directory: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in ("stats_mean", "stats_std", "feat_mean", "feat_std"):
        path = os.path.join(directory, f"{name}.npy")
        result[name] = np.load(path) if os.path.isfile(path) else np.array([])
    return result


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_context_columns(feature_matrix: list[dict[str, Any]]) -> list[str]:
    _excluded = {"player_id", "game_pk", "date"}
    _numeric_types = (int, float)
    sample_cols: set[str] | None = None
    for fr in feature_matrix[:100]:
        if sample_cols is None:
            sample_cols = set(fr.keys()) - _excluded
        sample_cols = {
            k
            for k in sample_cols
            if k in fr and isinstance(fr.get(k), (_numeric_types, type(None)))
        }
    return sorted(sample_cols) if sample_cols else []


def _build_hybrid_sequences(
    game_logs: list[Any],
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    seq_len: int = SEQUENCE_LEN,
    stats_mean: np.ndarray | None = None,
    stats_std: np.ndarray | None = None,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
    target_cols: list[str] | None = None,
) -> tuple:
    if target_cols is None:
        target_cols = ["target_0.5"]
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
    y_lists: list[list[int]] = [[] for _ in target_cols]
    ctx_cols = _build_context_columns(feature_matrix)

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
            for yi, col in enumerate(target_cols):
                y_lists[yi].append(target_row.get(col, 0))

    X_seq = np.stack(seq_list)
    X_ctx = np.stack(ctx_list)
    y_arrays = [np.array(y, dtype=np.int32) for y in y_lists]

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

    return (X_seq, X_ctx, *y_arrays, stats_mean, stats_std, feat_mean, feat_std)


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
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    result = _build_hybrid_sequences(
        game_logs,
        feature_matrix,
        targets,
        seq_len,
        stats_mean,
        stats_std,
        feat_mean,
        feat_std,
        target_cols=[target_col],
    )
    X_seq, X_ctx, y, stats_mean, stats_std, feat_mean, feat_std = (
        result[0],
        result[1],
        result[2],
        result[-4],
        result[-3],
        result[-2],
        result[-1],
    )
    return X_seq, X_ctx, y, stats_mean, stats_std, feat_mean, feat_std


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
    result = _build_hybrid_sequences(
        game_logs,
        feature_matrix,
        targets,
        seq_len,
        stats_mean,
        stats_std,
        feat_mean,
        feat_std,
        target_cols=["target_0.5", "target_1.5"],
    )
    return (
        result[0],
        result[1],
        result[2],
        result[3],
        result[-4],
        result[-3],
        result[-2],
        result[-1],
    )


# ---------------------------------------------------------------------------
# Model: HybridHitPredictor
# ---------------------------------------------------------------------------


class HybridHitPredictor(_NNModuleBase):
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


def _hybrid_loss(model, y_arrays, l2_reg):
    y = y_arrays[0]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pw = n_neg / n_pos if n_pos > 0 else 1.0

    def loss_fn(xs, xc, yb):
        logits = model(xs, xc)
        losses = nn.losses.binary_cross_entropy(logits, yb)
        losses = losses * mx.where(yb > 0.5, pw, 1.0)
        base = losses.mean()
        l2 = sum((p * p).sum() for _, p in tree_flatten(model.parameters()))
        return base + l2_reg * l2

    return loss_fn


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
    model = HybridHitPredictor(
        n_stats=X_seq.shape[2],
        n_context=X_ctx.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        use_gating=use_gating,
    )
    meta = _train_mlx_model(
        model,
        [X_seq, X_ctx],
        [y],
        loss_fn_builder=_hybrid_loss,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        l2_reg=l2_reg,
        seed=seed,
        verbose=verbose,
    )
    meta.update(
        {
            "arch": "HybridHitPredictor",
            "n_stats": X_seq.shape[2],
            "n_context": X_ctx.shape[1],
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
        }
    )
    return model, meta


def predict_hybrid_model(
    model: HybridHitPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> np.ndarray:
    return _predict_mlx_model(model, [X_seq, X_ctx], n_outputs=1)[0]


def save_hybrid_model(
    model: HybridHitPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    config = {
        "arch": "HybridHitPredictor",
        "n_stats": N_STATS,
        "hidden_dim": model.gru.hidden_size,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    return _save_model(
        model,
        directory,
        {
            "stats_mean": stats_mean,
            "stats_std": stats_std,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        },
        config,
    )


def load_hybrid_model(
    directory: str,
) -> tuple[
    HybridHitPredictor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
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
    arrs = _load_model_arrays(directory)
    return (
        model,
        arrs["stats_mean"],
        arrs["stats_std"],
        arrs["feat_mean"],
        arrs["feat_std"],
        config,
    )


# ---------------------------------------------------------------------------
# Model: MultiTaskHybridPredictor
# ---------------------------------------------------------------------------


class MultiTaskHybridPredictor(_NNModuleBase):
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


def _mt_loss(model, y_arrays, l2_reg):
    y_05, y_15 = y_arrays
    pw_05 = (len(y_05) - int(y_05.sum())) / max(int(y_05.sum()), 1)
    pw_15 = (len(y_15) - int(y_15.sum())) / max(int(y_15.sum()), 1)

    def loss_fn(xs, xc, y05b, y15b):
        logits_05, logits_15 = model(xs, xc)
        l05 = nn.losses.binary_cross_entropy(logits_05, y05b) * mx.where(
            y05b > 0.5, pw_05, 1.0
        )
        l15 = nn.losses.binary_cross_entropy(logits_15, y15b) * mx.where(
            y15b > 0.5, pw_15, 1.0
        )
        base = l05.mean() + l15.mean()
        l2 = sum((p * p).sum() for _, p in tree_flatten(model.parameters()))
        return base + l2_reg * l2

    return loss_fn


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
    model = MultiTaskHybridPredictor(
        n_stats=X_seq.shape[2],
        n_context=X_ctx.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        context_depth=context_depth,
    )
    meta = _train_mlx_model(
        model,
        [X_seq, X_ctx],
        [y_05, y_15],
        loss_fn_builder=_mt_loss,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        l2_reg=l2_reg,
        seed=seed,
        verbose=verbose,
    )
    meta.update(
        {
            "arch": "MultiTaskHybridPredictor",
            "n_stats": X_seq.shape[2],
            "n_context": X_ctx.shape[1],
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "context_depth": context_depth,
        }
    )
    return model, meta


def predict_multi_task_model(
    model: MultiTaskHybridPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return tuple(_predict_mlx_model(model, [X_seq, X_ctx], n_outputs=2))  # type: ignore[return-value]


def save_multi_task_model(
    model: MultiTaskHybridPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    config = {
        "arch": "MultiTaskHybridPredictor",
        "n_stats": N_STATS,
        "hidden_dim": model.gru.hidden_size,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    return _save_model(
        model,
        directory,
        {
            "stats_mean": stats_mean,
            "stats_std": stats_std,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        },
        config,
    )


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
    arrs = _load_model_arrays(directory)
    return (
        model,
        arrs["stats_mean"],
        arrs["stats_std"],
        arrs["feat_mean"],
        arrs["feat_std"],
        config,
    )


# ---------------------------------------------------------------------------
# Model: DCNMultiTaskPredictor
# ---------------------------------------------------------------------------


class CrossNetwork(_NNModuleBase):
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

    def __call__(self, seq: mx.array, ctx: mx.array) -> tuple[mx.array, mx.array]:
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
    model = DCNMultiTaskPredictor(
        n_stats=X_seq.shape[2],
        n_context=X_ctx.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        cross_dim=cross_dim,
        num_cross_layers=num_cross_layers,
    )
    meta = _train_mlx_model(
        model,
        [X_seq, X_ctx],
        [y_05, y_15],
        loss_fn_builder=_mt_loss,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        l2_reg=l2_reg,
        seed=seed,
        verbose=verbose,
    )
    meta.update(
        {
            "arch": "DCNMultiTaskPredictor",
            "n_stats": X_seq.shape[2],
            "n_context": X_ctx.shape[1],
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "cross_dim": cross_dim,
            "num_cross_layers": num_cross_layers,
        }
    )
    return model, meta


def predict_dcn_multi_task_model(
    model: DCNMultiTaskPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return tuple(_predict_mlx_model(model, [X_seq, X_ctx], n_outputs=2))  # type: ignore[return-value]


def save_dcn_model(
    model: DCNMultiTaskPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    config = {
        "arch": "DCNMultiTaskPredictor",
        "n_stats": N_STATS,
        "hidden_dim": model.gru.hidden_size,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    return _save_model(
        model,
        directory,
        {
            "stats_mean": stats_mean,
            "stats_std": stats_std,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        },
        config,
    )


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
    arrs = _load_model_arrays(directory)
    return (
        model,
        arrs["stats_mean"],
        arrs["stats_std"],
        arrs["feat_mean"],
        arrs["feat_std"],
        config,
    )


# ---------------------------------------------------------------------------
# Model: TransformerMultiTaskPredictor
# ---------------------------------------------------------------------------


class PositionalEncoding(_NNModuleBase):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = mx.random.normal((max_len, d_model)) * 0.02

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.embedding[: x.shape[1]]


class TransformerEncoder(_NNModuleBase):
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

    def __call__(self, seq: mx.array, ctx: mx.array) -> tuple[mx.array, mx.array]:
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
    model = TransformerMultiTaskPredictor(
        n_stats=X_seq.shape[2],
        n_context=X_ctx.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )
    meta = _train_mlx_model(
        model,
        [X_seq, X_ctx],
        [y_05, y_15],
        loss_fn_builder=_mt_loss,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        l2_reg=l2_reg,
        seed=seed,
        verbose=verbose,
    )
    meta.update(
        {
            "arch": "TransformerMultiTaskPredictor",
            "n_stats": X_seq.shape[2],
            "n_context": X_ctx.shape[1],
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    )
    return model, meta


def predict_transformer_multi_task_model(
    model: TransformerMultiTaskPredictor,
    X_seq: np.ndarray,
    X_ctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return tuple(_predict_mlx_model(model, [X_seq, X_ctx], n_outputs=2))  # type: ignore[return-value]


def save_transformer_model(
    model: TransformerMultiTaskPredictor,
    directory: str,
    stats_mean: np.ndarray | None,
    stats_std: np.ndarray | None,
    feat_mean: np.ndarray | None,
    feat_std: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    config = {
        "arch": "TransformerMultiTaskPredictor",
        "n_stats": N_STATS,
        "d_model": model.transformer.input_proj.out_features,
        "dropout": 0.3,
    }
    if metadata:
        config.update(metadata)
    return _save_model(
        model,
        directory,
        {
            "stats_mean": stats_mean,
            "stats_std": stats_std,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        },
        config,
    )


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
    arrs = _load_model_arrays(directory)
    return (
        model,
        arrs["stats_mean"],
        arrs["stats_std"],
        arrs["feat_mean"],
        arrs["feat_std"],
        config,
    )
