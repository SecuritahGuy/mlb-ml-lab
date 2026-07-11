"""Train the sequence model (GRU over game log sequences).

Loads cached game logs + targets, builds sequences per season,
and runs walk-forward validation across season boundaries.

Usage:
    poetry run python pipeline/train_seq.py
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.evaluate import classification_metrics
from mlb_ml_lab.models.sequence import (
    SEQUENCE_LEN,
    build_sequences,
    predict_sequence_model,
    save_sequence_model,
    train_sequence_model,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
MODEL_DIR = "data/models/sequence"


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────
    print(f"Loading data from {CACHED_DATASET}...")
    raw_logs = load_game_logs(CACHED_DATASET)
    _, targets_list, meta = load_feature_data(CACHED_DATASET)
    print(f"  {len(raw_logs)} game logs, {len(targets_list)} targets")

    # Convert dict logs to PlayerGameLog objects for _feat_vec compatibility
    game_logs: list[PlayerGameLog] = []
    for d in raw_logs:
        game_logs.append(PlayerGameLog(**{
            k: v for k, v in d.items()
            if k in PlayerGameLog.__dataclass_fields__
        }))

    targets: list[dict[str, Any]] = targets_list
    assert len(game_logs) == len(targets), "Logs/targets length mismatch"

    # ── Walk-forward by season boundary ───────────────────────────
    train_seasons = TRAIN_SEASONS

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n{'=' * 60}")
        print(f"Walk-forward — {target_col}")
        print(f"{'=' * 60}")

        fold_metrics: list[dict[str, float]] = []

        for fold_idx in range(len(train_seasons) - 1):
            train_cutoff = train_seasons[fold_idx]
            test_season = train_seasons[fold_idx + 1]

            train_logs = [lg for lg, t in zip(game_logs, targets)
                          if int(t["date"][:4]) <= train_cutoff]
            test_logs = [lg for lg, t in zip(game_logs, targets)
                         if int(t["date"][:4]) == test_season]
            train_tgt = [t for t in targets if int(t["date"][:4]) <= train_cutoff]
            test_tgt = [t for t in targets if int(t["date"][:4]) == test_season]

            print(f"\n  Fold {fold_idx + 1}: train ≤{train_cutoff}, test={test_season}")
            print(f"    Train: {len(train_logs)} logs, Test: {len(test_logs)} logs")

            # Build sequences
            X_train, y_train, _, stats_mean, stats_std = build_sequences(
                train_logs, train_tgt, seq_len=SEQUENCE_LEN,
                target_col=target_col,
            )
            X_test, y_test, _, _, _ = build_sequences(
                test_logs, test_tgt, seq_len=SEQUENCE_LEN,
                stats_mean=stats_mean, stats_std=stats_std,
                target_col=target_col,
            )

            if len(X_train) == 0 or len(X_test) == 0:
                print("    Skipping fold — no sequences")
                continue

            print(f"    Train sequences: {len(X_train)}, "
                  f"Test sequences: {len(X_test)}")

            model, _ = train_sequence_model(
                X_train, y_train,
                hidden_dim=128,
                n_layers=2,
                dropout=0.2,
                learning_rate=1e-3,
                epochs=60,
                batch_size=512,
                verbose=True,
            )

            y_proba = predict_sequence_model(model, X_test)
            y_pred = (y_proba > 0.5).astype(np.int32)

            metrics = classification_metrics(
                y_test.tolist(), y_pred.tolist(), y_proba.tolist(),
            )
            metrics["fold"] = fold_idx
            metrics["n_train"] = len(X_train)
            metrics["n_test"] = len(X_test)
            fold_metrics.append(metrics)

            print(f"    Fold AUC: {metrics.get('auc', 'N/A'):.4f}  "
                  f"Acc: {metrics['accuracy']:.4f}  "
                  f"Brier: {metrics.get('brier', 'N/A'):.4f}")

        # Report aggregated
        if fold_metrics:
            avg_auc = float(np.mean([m.get("auc", 0) for m in fold_metrics
                                     if not np.isnan(m.get("auc", float("nan")))]))
            avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
            print(f"\n  === {target_col} Results ===")
            print(f"    Avg AUC:       {avg_auc:.4f}")
            print(f"    Avg Accuracy:  {avg_acc:.4f}")
            print(f"    Folds:         {len(fold_metrics)}")

    # ── Train final model on all data ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training final sequence model on all seasons")
    print(f"{'=' * 60}")

    X_all, y_all, _, stats_mean, stats_std = build_sequences(
        game_logs, targets, seq_len=SEQUENCE_LEN,
        target_col="target_0.5",
    )
    print(f"  {len(X_all)} sequences, {len(y_all)} targets")

    model, metadata = train_sequence_model(
        X_all, y_all,
        hidden_dim=128,
        n_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=80,
        batch_size=512,
        verbose=True,
    )

    save_sequence_model(
        model, MODEL_DIR, stats_mean, stats_std,
        metadata={**metadata, "seasons": TRAIN_SEASONS, "target": "target_0.5"},
    )
    print(f"  Model saved to {MODEL_DIR}")

    # Evaluate on training data
    y_proba = predict_sequence_model(model, X_all)
    y_pred = (y_proba > 0.5).astype(np.int32)
    metrics = classification_metrics(y_all.tolist(), y_pred.tolist(), y_proba.tolist())
    print(f"  Training AUC: {metrics.get('auc', 'N/A'):.4f}")
    print(f"  Training Acc: {metrics['accuracy']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
