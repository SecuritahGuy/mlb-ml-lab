"""Ensemble Transformer sequence model + tuned XGB on walk-forward splits.

Usage:
    poetry run python pipeline/transformer_ensemble.py
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.evaluate import classification_metrics
from mlb_ml_lab.models.sequence import (
    SEQUENCE_LEN,
    build_hybrid_sequences,
    predict_transformer_multi_task_model,
    save_transformer_model,
    train_transformer_multi_task_model,
)
from mlb_ml_lab.models.train import _build_model, _feature_columns

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
ENSEMBLE_DIR = "data/models/transformer_ensemble"

_TUNED_XGB = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
}

_TRANSFORMER_HP = {
    "d_model": 32,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": 40,
    "batch_size": 256,
    "l2_reg": 1e-5,
}


def _merge_rows(
    feat_rows: list[dict],
    tgt_rows: list[dict],
    target_col: str,
) -> list[dict]:
    merged: list[dict] = []
    for fr, tr in zip(feat_rows, tgt_rows):
        row = dict(fr)
        row[target_col] = tr[target_col]
        merged.append(row)
    return merged


def _extract_xgb(
    merged_rows: list[dict],
    target_col: str,
    cols: list[str] | None = None,
    imputer: SimpleImputer | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], SimpleImputer]:
    if cols is None:
        cols = [c for c in _feature_columns(merged_rows) if c != target_col]
    y = np.array([r[target_col] for r in merged_rows], dtype=np.int32)
    x = np.zeros((len(merged_rows), len(cols)), dtype=np.float64)
    for i, r in enumerate(merged_rows):
        for j, c in enumerate(cols):
            v = r.get(c)
            x[i, j] = float(v) if v is not None else float("nan")
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        x = imputer.fit_transform(x)
    else:
        x = imputer.transform(x)
    x = np.nan_to_num(x, nan=0.0)
    return x, y, cols, imputer


def main() -> None:
    # ── Load data ────────────────────────────────────────────────
    print("Loading data...")
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

    # Align: keep only rows present in both features and targets
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
    print(f"  Aligned: {len(aligned_logs)} games with both features + targets")

    # ── Walk-forward for both targets ────────────────────────────
    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n{'=' * 60}")
        print(f"Transformer walk-forward — {target_col}")
        print(f"{'=' * 60}")

        summaries: list[dict[str, float]] = []

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
            print(f"    Train: {len(train_logs)} logs, {len(train_feat)} features")
            print(f"    Test:  {len(test_logs)} logs, {len(test_feat)} features")

            Xs_tr, Xc_tr, y_tr, sm, ss, fm, fs = build_hybrid_sequences(
                train_logs,
                train_feat,
                train_tgt,
                target_col=target_col,
            )
            Xs_te, Xc_te, y_te, _, _, _, _ = build_hybrid_sequences(
                test_logs,
                test_feat,
                test_tgt,
                stats_mean=sm,
                stats_std=ss,
                feat_mean=fm,
                feat_std=fs,
                target_col=target_col,
            )
            if len(Xs_tr) == 0 or len(Xs_te) == 0:
                print("    Skipping — no sequences")
                continue

            train_merged = _merge_rows(train_feat, train_tgt, target_col)
            test_merged = _merge_rows(test_feat, test_tgt, target_col)

            # XGB
            x_train, y_tr_xgb, xgb_cols, xgb_imputer = _extract_xgb(
                train_merged,
                target_col,
            )
            x_test, y_te_xgb, _, _ = _extract_xgb(
                test_merged,
                target_col,
                cols=xgb_cols,
                imputer=xgb_imputer,
            )
            xgb_model = _build_model("xgb", SEED, params=_TUNED_XGB)
            xgb_model.fit(x_train, y_tr_xgb)
            xgb_te_proba = xgb_model.predict_proba(x_test)[:, 1]
            xgb_pred = (xgb_te_proba > 0.5).astype(np.int32)
            xgb_metrics = classification_metrics(
                y_te_xgb.tolist(),
                xgb_pred.tolist(),
                xgb_te_proba.tolist(),
            )

            # Transformer (multi-task — use target_0.5 output)
            tf_model, _ = train_transformer_multi_task_model(
                Xs_tr,
                Xc_tr,
                y_tr,
                y_tr,  # same target for both heads for single-target mode
                **_TRANSFORMER_HP,
                seed=SEED,
                verbose=False,
            )
            tf_te_proba_05, tf_te_proba_15 = predict_transformer_multi_task_model(
                tf_model,
                Xs_te,
                Xc_te,
            )
            tf_proba = tf_te_proba_05 if target_col == "target_0.5" else tf_te_proba_15
            tf_pred = (tf_proba > 0.5).astype(np.int32)
            tf_metrics = classification_metrics(
                y_te.tolist(),
                tf_pred.tolist(),
                tf_proba.tolist(),
            )

            # Build sharded prediction maps for alignment
            tf_proba_map: dict[tuple[int, int], float] = {}
            for i, row in enumerate(test_merged):
                if i < len(tf_proba):
                    tf_proba_map[(row["player_id"], row["game_pk"])] = float(
                        tf_proba[i]
                    )

            xgb_proba_map: dict[tuple[int, int], float] = {}
            for i, row in enumerate(test_merged):
                xgb_proba_map[(row["player_id"], row["game_pk"])] = float(
                    xgb_te_proba[i]
                )

            # Ensemble
            _keys_te: list[tuple[int, int]] = []
            _grouped: dict[tuple[int, str], list[tuple[int, Any]]] = defaultdict(list)
            for i, lg in enumerate(test_logs):
                _grouped[(lg.player_id, str(lg.season))].append((i, lg))

            for (pid, _season), entries in _grouped.items():
                entries.sort(key=lambda e: e[1].date)
                indices = [e[0] for e in entries]
                for pos in range(SEQUENCE_LEN, len(indices)):
                    idx = indices[pos]
                    key = (test_logs[idx].player_id, test_logs[idx].game_pk)
                    if key in xgb_proba_map:
                        _keys_te.append(key)

            ens_y = []
            ens_preds = []
            for pid, gpk in _keys_te:
                hp = tf_proba_map.get((pid, gpk), 0.5)
                xp = xgb_proba_map.get((pid, gpk), 0.5)
                ens_preds.append((hp + xp) / 2.0)
                for t in test_tgt:
                    if t["player_id"] == pid and t["game_pk"] == gpk:
                        ens_y.append(t[target_col])
                        break

            if len(ens_y) > 0:
                ens_y = np.array(ens_y)
                ens_preds = np.array(ens_preds)
                ens_pred_bin = (ens_preds > 0.5).astype(np.int32)
                ens_metrics = classification_metrics(
                    ens_y.tolist(),
                    ens_pred_bin.tolist(),
                    ens_preds.tolist(),
                )
                ens_auc = ens_metrics["auc"]
                ens_acc = ens_metrics["accuracy"]
            else:
                ens_auc = 0.5
                ens_acc = 0.5

            print(
                f"    XGB       AUC: {xgb_metrics['auc']:.4f}  "
                f"Acc: {xgb_metrics['accuracy']:.4f}"
            )
            print(
                f"    Transform AUC: {tf_metrics['auc']:.4f}  Acc: {tf_metrics['accuracy']:.4f}"
            )
            print(f"    Ensemble  AUC: {ens_auc:.4f}  Acc: {ens_acc:.4f}")

            summaries.append(
                {
                    "fold": fold_idx + 1,
                    "xgb_auc": xgb_metrics["auc"],
                    "tf_auc": tf_metrics["auc"],
                    "ens_auc": ens_auc,
                }
            )

        # Summary
        print(f"\n  === {target_col} Summary ===")
        for s in summaries:
            print(
                f"    Fold {s['fold']}: XGB={s['xgb_auc']:.4f}  "
                f"Transformer={s['tf_auc']:.4f}  Ensemble={s['ens_auc']:.4f}"
            )
        if summaries:
            avg_xgb = np.mean([s["xgb_auc"] for s in summaries])
            avg_tf = np.mean([s["tf_auc"] for s in summaries])
            avg_ens = np.mean([s["ens_auc"] for s in summaries])
            print(f"    Avg XGB:      {avg_xgb:.4f}")
            print(f"    Avg Transform:{avg_tf:.4f}")
            print(f"    Avg Ensemble: {avg_ens:.4f}")

    # ── Final training on all data ───────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training final Transformer on all data")
    print(f"{'=' * 60}")

    target_col = "target_0.5"
    merged_all = _merge_rows(aligned_feats, aligned_tgts, target_col)
    x_all, y_all, xgb_cols_all, _ = _extract_xgb(merged_all, target_col)

    xgb_final = _build_model("xgb", SEED, params=_TUNED_XGB)
    xgb_final.fit(x_all, y_all)

    Xs_all, Xc_all, y_all_seq, sm_f, ss_f, fm_f, fs_f = build_hybrid_sequences(
        aligned_logs,
        aligned_feats,
        aligned_tgts,
        target_col=target_col,
    )
    print(f"  Transformer train: {len(Xs_all)} sequences")

    tf_final, tf_meta = train_transformer_multi_task_model(
        Xs_all,
        Xc_all,
        y_all_seq,
        y_all_seq,
        **_TRANSFORMER_HP,
        seed=SEED,
        verbose=True,
    )

    # Save
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    save_transformer_model(
        tf_final,
        os.path.join(ENSEMBLE_DIR, "transformer"),
        sm_f,
        ss_f,
        fm_f,
        fs_f,
        tf_meta,
    )
    joblib.dump(xgb_final, os.path.join(ENSEMBLE_DIR, "xgb_model.joblib"))

    # Save XGB cols + imputer
    path_xgb_cols = os.path.join(ENSEMBLE_DIR, "xgb_cols.json")
    with open(path_xgb_cols, "w", encoding="utf-8") as f:
        json.dump(xgb_cols_all, f)
    np.save(
        os.path.join(ENSEMBLE_DIR, "xgb_imputer_mean.npy"),
        _extract_xgb(merged_all, target_col)[3].statistics_,
    )

    # Save ensemble config
    config = {
        "arch": "Transformer + XGB Ensemble",
        "target_col": target_col,
        "transformer_config": _TRANSFORMER_HP,
        "xgb_config": _TUNED_XGB,
        "n_features": len(xgb_cols_all),
        "n_seq": len(Xs_all),
    }
    with open(
        os.path.join(ENSEMBLE_DIR, "ensemble_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)

    print(f"  Ensemble saved to {ENSEMBLE_DIR}")
    print(f"  XGB rows: {len(x_all)}, Transformer seqs: {len(Xs_all)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
