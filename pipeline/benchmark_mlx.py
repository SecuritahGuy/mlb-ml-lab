"""Benchmark MLX training speed — full dataset."""

from __future__ import annotations

import time

import numpy as np

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.sequence import (
    build_hybrid_sequences,
    train_hybrid_model,
)

CACHED_DATASET = "data/datasets/full_2016_2026_30teams"
N_EPOCHS = 10


def benchmark(
    label: str,
    batch_size: int,
    hidden_dim: int = 64,
    n_layers: int = 2,
) -> float:
    print(f"\n{'=' * 50}")
    print(f"Config: {label}")
    t0 = time.time()
    model, meta = train_hybrid_model(
        Xs_train, Xc_train, y_train,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=N_EPOCHS,
        batch_size=batch_size,
        verbose=False,
    )
    elapsed = time.time() - t0
    avg = elapsed / N_EPOCHS
    print(f"  Total: {elapsed:.1f}s, avg epoch: {avg:.2f}s")
    return avg


# ── Load data ──────────────────────────────────────────────────────
print("Loading full dataset...")
raw_logs = load_game_logs(CACHED_DATASET)
feature_matrix, targets_list, meta = load_feature_data(CACHED_DATASET)
print(f"  {len(raw_logs)} logs, {len(feature_matrix)} feature rows")

game_logs: list[PlayerGameLog] = []
for d in raw_logs:
    game_logs.append(PlayerGameLog(**{
        k: v for k, v in d.items()
        if k in PlayerGameLog.__dataclass_fields__
    }))

print("Building sequences (full dataset)...")
Xs_train, Xc_train, y_train, sm, ss, fm, fs = build_hybrid_sequences(
    game_logs, feature_matrix, targets_list, target_col="target_0.5",
)
print(f"Sequences: {len(Xs_train)}")

# ── Benchmarks ────────────────────────────────────────────────────
results: dict[str, float] = {}

results["batch=512 (baseline)"] = benchmark("batch=512 (baseline)", batch_size=512)
results["batch=2048"] = benchmark("batch=2048", batch_size=2048)

# If not much difference, skip further
baseline = results["batch=512 (baseline)"]
bs2048 = results["batch=2048"]
speedup = baseline / bs2048

if speedup >= 1.15:
    results["batch=4096"] = benchmark("batch=4096", batch_size=4096)

# Bigger model test
results["batch=512 + hidden=128"] = benchmark(
    "batch=512 + hidden=128", batch_size=512, hidden_dim=128,
)
results["batch=512 + hidden=128 + L3"] = benchmark(
    "batch=512 + hidden=128 + L3", batch_size=512, hidden_dim=128, n_layers=3,
)

print(f"\n{'=' * 50}")
print("Summary (10 epochs, avg epoch time):")
for label, avg in results.items():
    sp = baseline / avg
    print(f"  {label:40s}  {avg:.2f}s  ({sp:.1f}x vs baseline)")
