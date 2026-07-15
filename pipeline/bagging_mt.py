"""Bagged multi-task ensemble (5 seeds, averaged predictions).

Usage:
    poetry run python pipeline/bagging_mt.py
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Any

import numpy as np

try:
    import mlx.core as mx  # pylint: disable=import-error
except ImportError:  # pragma: no cover
    pass
from sklearn.exceptions import ConvergenceWarning

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.evaluate import classification_metrics
from mlb_ml_lab.models.sequence import (
    build_hybrid_mt_sequences,
    load_multi_task_model,
    predict_multi_task_model,
    save_multi_task_model,
    train_multi_task_model,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
BAGGING_SEEDS = [42, 43, 44, 45, 46]
MODEL_DIR = "data/models/bagging_mt"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _build_hybrid(
    logs: list,
    feats: list[dict],
    tgts: list[dict],
    sm: np.ndarray | None = None,
    ss: np.ndarray | None = None,
    fm: np.ndarray | None = None,
    fs: np.ndarray | None = None,
) -> tuple:
    return build_hybrid_mt_sequences(
        logs,
        feats,
        tgts,
        stats_mean=sm,
        stats_std=ss,
        feat_mean=fm,
        feat_std=fs,
    )


def main() -> None:
    print(f"Loading data from {CACHED_DATASET}...")
    raw_logs = load_game_logs(CACHED_DATASET)
    feature_matrix, targets_list, _meta = load_feature_data(CACHED_DATASET)
    print(
        f"  {len(raw_logs)} game logs, {len(feature_matrix)} feature rows, "
        f"{len(targets_list)} targets"
    )

    game_logs: list[PlayerGameLog] = []
    for d in raw_logs:
        game_logs.append(
            PlayerGameLog(
                **{
                    k: v
                    for k, v in d.items()
                    if k in PlayerGameLog.__dataclass_fields__
                }
            )
        )

    targets: list[dict] = targets_list
    features: list[dict] = feature_matrix

    feat_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for f in features:
        feat_by_key[(f["player_id"], f["game_pk"])] = f
    tgt_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for t in targets:
        tgt_by_key[(t["player_id"], t["game_pk"])] = t

    aligned_logs: list[PlayerGameLog] = []
    aligned_feats: list[dict] = []
    aligned_tgts: list[dict] = []
    for lg in game_logs:
        key = (lg.player_id, lg.game_pk)
        fr = feat_by_key.get(key)
        tr = tgt_by_key.get(key)
        if fr is not None and tr is not None:
            aligned_logs.append(lg)
            aligned_feats.append(fr)
            aligned_tgts.append(tr)
    print(f"  Aligned: {len(aligned_logs)} games")

    # ── Walk-forward bagging ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Bagged multi-task walk-forward (5 seeds)")
    print(f"{'=' * 60}")

    all_results: list[dict] = []

    for fold_idx in range(len(TRAIN_SEASONS) - 1):
        train_cutoff = TRAIN_SEASONS[fold_idx]
        test_season = TRAIN_SEASONS[fold_idx + 1]

        train_logs = [
            lg
            for lg, t in zip(aligned_logs, aligned_tgts)
            if int(t["date"][:4]) <= train_cutoff
        ]
        test_logs = [
            lg
            for lg, t in zip(aligned_logs, aligned_tgts)
            if int(t["date"][:4]) == test_season
        ]
        train_tgt = [t for t in aligned_tgts if int(t["date"][:4]) <= train_cutoff]
        test_tgt = [t for t in aligned_tgts if int(t["date"][:4]) == test_season]
        train_feat = [
            f
            for f, t in zip(aligned_feats, aligned_tgts)
            if int(t["date"][:4]) <= train_cutoff
        ]
        test_feat = [
            f
            for f, t in zip(aligned_feats, aligned_tgts)
            if int(t["date"][:4]) == test_season
        ]

        print(f"\n  Fold {fold_idx + 1}: train ≤{train_cutoff}, test={test_season}")
        print(f"    Train: {len(train_logs)} logs, Test: {len(test_logs)} logs")

        Xs_tr, Xc_tr, y05_tr, y15_tr, sm, ss, fm, fs = _build_hybrid(
            train_logs,
            train_feat,
            train_tgt,
        )
        Xs_te, Xc_te, y05_te, y15_te, _, _, _, _ = _build_hybrid(
            test_logs,
            test_feat,
            test_tgt,
            sm,
            ss,
            fm,
            fs,
        )

        if len(Xs_tr) == 0 or len(Xs_te) == 0:
            print("    Skipping — no sequences")
            continue
        print(f"    Train: {len(Xs_tr)} seqs, Test: {len(Xs_te)} seqs")

        bag_p05 = np.zeros_like(y05_te, dtype=np.float32)
        bag_p15 = np.zeros_like(y15_te, dtype=np.float32)

        for seed_idx, seed in enumerate(BAGGING_SEEDS):
            _set_seed(seed)
            print(f"    Seed {seed} ({seed_idx + 1}/{len(BAGGING_SEEDS)})...")
            model, _ = train_multi_task_model(
                Xs_tr,
                Xc_tr,
                y05_tr,
                y15_tr,
                hidden_dim=64,
                n_layers=2,
                dropout=0.2,
                learning_rate=1e-3,
                epochs=60,
                batch_size=512,
                verbose=False,
            )
            p05, p15 = predict_multi_task_model(model, Xs_te, Xc_te)
            bag_p05 += p05 / len(BAGGING_SEEDS)
            bag_p15 += p15 / len(BAGGING_SEEDS)

        m05 = classification_metrics(
            y05_te.tolist(),
            (bag_p05 > 0.5).astype(np.int32).tolist(),
            bag_p05.tolist(),
        )
        m15 = classification_metrics(
            y15_te.tolist(),
            (bag_p15 > 0.5).astype(np.int32).tolist(),
            bag_p15.tolist(),
        )

        row = {
            "fold": fold_idx + 1,
            "auc_05": m05.get("auc", float("nan")),
            "acc_05": m05["accuracy"],
            "auc_15": m15.get("auc", float("nan")),
            "acc_15": m15["accuracy"],
            "n_train": len(Xs_tr),
            "n_test": len(Xs_te),
        }
        all_results.append(row)
        print(
            f"    Bagged target_0.5 AUC: {row['auc_05']:.4f}  Acc: {row['acc_05']:.4f}"
        )
        print(
            f"    Bagged target_1.5 AUC: {row['auc_15']:.4f}  Acc: {row['acc_15']:.4f}"
        )

    if all_results:
        print("\n  === Bagged walk-forward results ===")
        for r in all_results:
            print(
                f"    Fold {r['fold']}: 0.5 AUC={r['auc_05']:.4f}  "
                f"1.5 AUC={r['auc_15']:.4f}"
            )
        avg_05 = float(np.mean([r["auc_05"] for r in all_results]))
        avg_15 = float(np.mean([r["auc_15"] for r in all_results]))
        print(f"    Avg target_0.5 AUC: {avg_05:.4f}")
        print(f"    Avg target_1.5 AUC: {avg_15:.4f}")
        print(
            f"    Gain over single multi-task (0.702/0.679): "
            f"+{avg_05 - 0.702:+.3f} / +{avg_15 - 0.679:+.3f}"
        )

    # ── Train final bagged ensemble on all data ───────────────────
    print(f"\n{'=' * 60}")
    print("Training final bagged ensemble on all seasons")
    print(f"{'=' * 60}")

    Xs, Xc, y05, y15, sm, ss, fm, fs = _build_hybrid(
        aligned_logs,
        aligned_feats,
        aligned_tgts,
    )
    print(f"  {len(Xs)} samples")

    for seed_idx, seed in enumerate(BAGGING_SEEDS):
        _set_seed(seed)
        print(f"  Model {seed_idx + 1}/{len(BAGGING_SEEDS)} (seed={seed})...")
        model, metadata = train_multi_task_model(
            Xs,
            Xc,
            y05,
            y15,
            hidden_dim=64,
            n_layers=2,
            dropout=0.2,
            learning_rate=1e-3,
            epochs=90,
            batch_size=512,
            verbose=True,
        )
        model_dir = os.path.join(MODEL_DIR, f"seed_{seed}")
        save_multi_task_model(
            model,
            model_dir,
            sm,
            ss,
            fm,
            fs,
            metadata={**metadata, "bagging_seed": seed, "seasons": TRAIN_SEASONS},
        )

    # Training set performance (bagged average)
    bag_p05 = np.zeros_like(y05, dtype=np.float32)
    bag_p15 = np.zeros_like(y15, dtype=np.float32)
    for seed in BAGGING_SEEDS:
        m, _, _, _, _, _ = load_multi_task_model(
            os.path.join(MODEL_DIR, f"seed_{seed}")
        )
        p05, p15 = predict_multi_task_model(m, Xs, Xc)
        bag_p05 += p05 / len(BAGGING_SEEDS)
        bag_p15 += p15 / len(BAGGING_SEEDS)

    m05 = classification_metrics(
        y05.tolist(),
        (bag_p05 > 0.5).astype(np.int32).tolist(),
        bag_p05.tolist(),
    )
    m15 = classification_metrics(
        y15.tolist(),
        (bag_p15 > 0.5).astype(np.int32).tolist(),
        bag_p15.tolist(),
    )
    print(f"  Bagged training AUC (0.5): {m05.get('auc', 'N/A'):.4f}")
    print(f"  Bagged training Acc (0.5): {m05['accuracy']:.4f}")
    print(f"  Bagged training AUC (1.5): {m15.get('auc', 'N/A'):.4f}")
    print(f"  Bagged training Acc (1.5): {m15['accuracy']:.4f}")
    print(f"\n  Models saved to {MODEL_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
