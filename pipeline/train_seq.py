"""Train the hybrid sequence+context model.

Loads cached game logs, feature matrix, and targets, builds sequences
per season, and runs walk-forward validation.

Usage:
    poetry run python pipeline/train_seq.py
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.evaluate import classification_metrics
from mlb_ml_lab.models.sequence import (
    build_hybrid_sequences,
    predict_hybrid_model,
    save_hybrid_model,
    train_hybrid_model,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
MODEL_DIR = "data/models/hybrid"


def _run_fold(
    train_logs: list,
    train_tgt: list[dict],
    train_feat: list[dict],
    test_logs: list,
    test_tgt: list[dict],
    test_feat: list[dict],
    target_col: str,
    fold_idx: int,
) -> dict:
    """Train hybrid model on one fold and return metrics."""
    Xs_tr, Xc_tr, y_tr, sm, ss, fm, fs = build_hybrid_sequences(
        train_logs, train_feat, train_tgt,
        target_col=target_col,
    )
    Xs_te, Xc_te, y_te, _, _, _, _ = build_hybrid_sequences(
        test_logs, test_feat, test_tgt,
        stats_mean=sm, stats_std=ss,
        feat_mean=fm, feat_std=fs,
        target_col=target_col,
    )

    if len(Xs_tr) == 0 or len(Xs_te) == 0:
        return {}

    print(f"    Train: {len(Xs_tr)} samples, Test: {len(Xs_te)} samples")

    model, _ = train_hybrid_model(
        Xs_tr, Xc_tr, y_tr,
        hidden_dim=64,
        n_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=60,
        batch_size=512,
        verbose=True,
    )

    y_proba = predict_hybrid_model(model, Xs_te, Xc_te)
    y_pred = (y_proba > 0.5).astype(np.int32)
    metrics = classification_metrics(
        y_te.tolist(), y_pred.tolist(), y_proba.tolist(),
    )
    metrics["fold"] = fold_idx
    metrics["n_train"] = len(Xs_tr)
    metrics["n_test"] = len(Xs_te)
    return metrics


def main() -> None:
    print(f"Loading data from {CACHED_DATASET}...")
    raw_logs = load_game_logs(CACHED_DATASET)
    feature_matrix, targets_list, meta = load_feature_data(CACHED_DATASET)
    print(f"  {len(raw_logs)} game logs, {len(feature_matrix)} feature rows, "
          f"{len(targets_list)} targets")

    game_logs: list = []
    for d in raw_logs:
        game_logs.append(PlayerGameLog(**{
            k: v for k, v in d.items()
            if k in PlayerGameLog.__dataclass_fields__
        }))

    targets: list[dict] = targets_list
    features: list[dict] = feature_matrix

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n{'=' * 60}")
        print(f"Walk-forward — {target_col}")
        print(f"{'=' * 60}")

        fold_metrics: list[dict] = []

        for fold_idx in range(len(TRAIN_SEASONS) - 1):
            train_cutoff = TRAIN_SEASONS[fold_idx]
            test_season = TRAIN_SEASONS[fold_idx + 1]

            train_logs = [lg for lg, t in zip(game_logs, targets)
                          if int(t["date"][:4]) <= train_cutoff]
            test_logs = [lg for lg, t in zip(game_logs, targets)
                         if int(t["date"][:4]) == test_season]
            train_tgt = [t for t in targets if int(t["date"][:4]) <= train_cutoff]
            test_tgt = [t for t in targets if int(t["date"][:4]) == test_season]
            train_feat = [f for f, t in zip(features, targets)
                          if int(t["date"][:4]) <= train_cutoff]
            test_feat = [f for f, t in zip(features, targets)
                         if int(t["date"][:4]) == test_season]

            print(f"\n  Fold {fold_idx + 1}: train ≤{train_cutoff}, test={test_season}")
            print(f"    Train: {len(train_logs)} logs, {len(train_feat)} features")
            print(f"    Test:  {len(test_logs)} logs, {len(test_feat)} features")

            metrics = _run_fold(
                train_logs, train_tgt, train_feat,
                test_logs, test_tgt, test_feat,
                target_col, fold_idx,
            )
            if not metrics:
                print("    Skipping fold — no sequences")
                continue

            fold_metrics.append(metrics)
            print(f"    Fold AUC: {metrics.get('auc', 'N/A'):.4f}  "
                  f"Acc: {metrics['accuracy']:.4f}")

        if fold_metrics:
            avg_auc = float(np.mean([m.get("auc", 0) for m in fold_metrics
                                     if not np.isnan(m.get("auc", float("nan")))]))
            avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
            print(f"\n  === {target_col} Results ===")
            print(f"    Avg AUC:       {avg_auc:.4f}")
            print(f"    Avg Accuracy:  {avg_acc:.4f}")
            print(f"    Folds:         {len(fold_metrics)}")

    # ── Train final hybrid model ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training final hybrid model on all seasons")
    print(f"{'=' * 60}")

    Xs, Xc, y, sm, ss, fm, fs = build_hybrid_sequences(
        game_logs, features, targets, target_col="target_0.5",
    )
    print(f"  {len(Xs)} samples")

    model, metadata = train_hybrid_model(
        Xs, Xc, y,
        hidden_dim=64,
        n_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=80,
        batch_size=512,
        verbose=True,
    )

    save_hybrid_model(
        model, MODEL_DIR, sm, ss, fm, fs,
        metadata={**metadata, "seasons": TRAIN_SEASONS, "target": "target_0.5"},
    )
    print(f"  Model saved to {MODEL_DIR}")

    y_proba = predict_hybrid_model(model, Xs, Xc)
    y_pred = (y_proba > 0.5).astype(np.int32)
    metrics = classification_metrics(y.tolist(), y_pred.tolist(), y_proba.tolist())
    print(f"  Training AUC: {metrics.get('auc', 'N/A'):.4f}")
    print(f"  Training Acc: {metrics['accuracy']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
