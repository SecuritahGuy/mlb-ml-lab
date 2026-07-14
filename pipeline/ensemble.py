"""Ensemble hybrid sequence model + tuned XGB on walk-forward splits.

Usage:
    poetry run python pipeline/ensemble.py
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
    _feat_vec,
    build_hybrid_sequences,
    predict_hybrid_model,
    save_hybrid_model,
    train_hybrid_model,
)
from mlb_ml_lab.models.train import _build_model, _feature_columns

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2016_2026_30teams"
TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
SEED = 42
ENSEMBLE_DIR = "data/models/ensemble"

_TUNED_XGB = {
    "n_estimators": 500, "max_depth": 5, "learning_rate": 0.01,
    "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1,
}


def _merge_rows(
    feat_rows: list[dict],
    tgt_rows: list[dict],
    target_col: str,
) -> list[dict]:
    """Merge feature rows with their target values into single dicts."""
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
    """Convert merged rows to XGB-ready (X, y, cols, imputer)."""
    if cols is None:
        # Exclude target from feature columns (it's already in merged rows)
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
    print(f"Loading data from {CACHED_DATASET}...")
    raw_logs = load_game_logs(CACHED_DATASET)
    feature_matrix, targets_list, meta = load_feature_data(CACHED_DATASET)
    print(f"  {len(raw_logs)} game logs, {len(feature_matrix)} feature rows, "
          f"{len(targets_list)} targets")

    game_logs: list[PlayerGameLog] = []
    for d in raw_logs:
        game_logs.append(PlayerGameLog(**{
            k: v for k, v in d.items()
            if k in PlayerGameLog.__dataclass_fields__
        }))

    targets: list[dict] = targets_list
    features: list[dict] = feature_matrix

    # Build (player_id, game_pk) index for alignment
    feat_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for f in features:
        feat_by_key[(f["player_id"], f["game_pk"])] = f
    tgt_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for t in targets:
        tgt_by_key[(t["player_id"], t["game_pk"])] = t

    # Align logs to games that have both feature and target rows
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

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n{'=' * 60}")
        print(f"Ensemble walk-forward — {target_col}")
        print(f"{'=' * 60}")

        all_results: list[dict] = []

        for fold_idx in range(len(TRAIN_SEASONS) - 1):
            train_cutoff = TRAIN_SEASONS[fold_idx]
            test_season = TRAIN_SEASONS[fold_idx + 1]

            # Filter aligned lists by target date
            train_logs = [lg for lg, t in zip(aligned_logs, aligned_tgts)
                          if int(t["date"][:4]) <= train_cutoff]
            test_logs = [lg for lg, t in zip(aligned_logs, aligned_tgts)
                         if int(t["date"][:4]) == test_season]
            train_tgt = [t for t in aligned_tgts if int(t["date"][:4]) <= train_cutoff]
            test_tgt = [t for t in aligned_tgts if int(t["date"][:4]) == test_season]
            train_feat = [f for f, t in zip(aligned_feats, aligned_tgts)
                          if int(t["date"][:4]) <= train_cutoff]
            test_feat = [f for f, t in zip(aligned_feats, aligned_tgts)
                         if int(t["date"][:4]) == test_season]

            print(f"\n  Fold {fold_idx + 1}: train ≤{train_cutoff}, test={test_season}")
            print(f"    Train: {len(train_logs)} logs, {len(train_feat)} features")
            print(f"    Test:  {len(test_logs)} logs, {len(test_feat)} features")

            # ── Build hybrid sequences ──────────────────────────────────────
            Xs_tr, Xc_tr, y_tr, sm, ss, fm, fs = build_hybrid_sequences(
                train_logs, train_feat, train_tgt, target_col=target_col,
            )
            Xs_te, Xc_te, y_te, _, _, _, _ = build_hybrid_sequences(
                test_logs, test_feat, test_tgt,
                stats_mean=sm, stats_std=ss, feat_mean=fm, feat_std=fs,
                target_col=target_col,
            )
            if len(Xs_tr) == 0 or len(Xs_te) == 0:
                print("    Skipping — no sequences")
                continue
            print(f"    Hybrid train: {len(Xs_tr)}, test: {len(Xs_te)}")

            # ── Prepare XGB data ────────────────────────────────────────────
            train_merged = _merge_rows(train_feat, train_tgt, target_col)
            test_merged = _merge_rows(test_feat, test_tgt, target_col)

            x_train, y_tr_xgb, _xgb_cols, xgb_imputer = _extract_xgb(
                train_merged, target_col,
            )
            x_test, y_te_xgb, _, _ = _extract_xgb(
                test_merged, target_col, cols=_xgb_cols, imputer=xgb_imputer,
            )

            print(f"    XGB train: {len(x_train)}, test: {len(x_test)}")

            # ── Train XGB ───────────────────────────────────────────────────
            xgb_model = _build_model("xgb", SEED, params=_TUNED_XGB)
            xgb_model.fit(x_train, y_tr_xgb)
            xgb_te_proba = xgb_model.predict_proba(x_test)[:, 1]

            # ── Train hybrid ────────────────────────────────────────────────
            hybrid_model, _ = train_hybrid_model(
                Xs_tr, Xc_tr, y_tr,
                hidden_dim=64,
                n_layers=2,
                dropout=0.2,
                learning_rate=1e-3,
                epochs=60,
                batch_size=512,
                verbose=False,
            )
            hybrid_te_proba = predict_hybrid_model(hybrid_model, Xs_te, Xc_te)

            # ── Evaluate individually ───────────────────────────────────────
            xgb_pred = (xgb_te_proba > 0.5).astype(np.int32)
            hybrid_pred = (hybrid_te_proba > 0.5).astype(np.int32)

            xgb_metrics = classification_metrics(
                y_te_xgb.tolist(), xgb_pred.tolist(), xgb_te_proba.tolist(),
            )
            hybrid_metrics = classification_metrics(
                y_te.tolist(), hybrid_pred.tolist(), hybrid_te_proba.tolist(),
            )

            # ── Ensemble: average probabilities ─────────────────────────────
            # Hybrid only covers games with 15+ prior games (a subset of XGB).
            # Map XGB predictions by (player_id, game_pk) then look up each
            # hybrid test row.
            xgb_proba_map: dict[tuple[int, int], float] = {}
            for i, row in enumerate(test_merged):
                xgb_proba_map[(row["player_id"], row["game_pk"])] = float(xgb_te_proba[i])

            # Capture (pid, gpk) for each hybrid test sequence
            _keys_te: list[tuple[int, int]] = []
            _feat_idx: dict[tuple[int, int], dict] = {}
            for f in test_feat:
                _feat_idx[(f["player_id"], f["game_pk"])] = f

            _grouped: dict[tuple[int, str], list[tuple[int, Any]]] = defaultdict(list)
            for i, lg in enumerate(test_logs):
                _grouped[(lg.player_id, str(lg.season))].append((i, lg))

            for (pid, season), entries in _grouped.items():
                entries.sort(key=lambda e: e[1].date)
                indices = [e[0] for e in entries]
                vecs = [_feat_vec(e[1]) for e in entries]
                for pos in range(SEQUENCE_LEN, len(vecs)):
                    idx = indices[pos]
                    lg = test_logs[idx]
                    if _feat_idx.get((lg.player_id, lg.game_pk)):
                        _keys_te.append((lg.player_id, lg.game_pk))

            _ensemble_probas: list[float] = []
            for i, (pid, gpk) in enumerate(_keys_te):
                hp = float(hybrid_te_proba[i])
                xp = xgb_proba_map.get((pid, gpk), hp)
                _ensemble_probas.append((hp + xp) / 2.0)

            ensemble_pred = (np.array(_ensemble_probas) > 0.5).astype(np.int32)
            ens_metrics = classification_metrics(
                y_te.tolist(), ensemble_pred.tolist(), _ensemble_probas,
            )

            fold_result = {
                "fold": fold_idx + 1,
                "xgb_auc": xgb_metrics.get("auc", float("nan")),
                "xgb_acc": xgb_metrics["accuracy"],
                "hybrid_auc": hybrid_metrics.get("auc", float("nan")),
                "hybrid_acc": hybrid_metrics["accuracy"],
                "ensemble_auc": ens_metrics.get("auc", float("nan")),
                "ensemble_acc": ens_metrics["accuracy"],
                "n_train": len(Xs_tr),
                "n_test": len(Xs_te),
            }
            all_results.append(fold_result)

            print(f"    XGB     AUC: {fold_result['xgb_auc']:.4f}  "
                  f"Acc: {fold_result['xgb_acc']:.4f}")
            print(f"    Hybrid  AUC: {fold_result['hybrid_auc']:.4f}  "
                  f"Acc: {fold_result['hybrid_acc']:.4f}")
            print(f"    Ensemble AUC: {fold_result['ensemble_auc']:.4f}  "
                  f"Acc: {fold_result['ensemble_acc']:.4f}")

        if all_results:
            print(f"\n  === {target_col} Summary ===")
            for r in all_results:
                print(f"    Fold {r['fold']}: XGB={r['xgb_auc']:.4f}  "
                      f"Hybrid={r['hybrid_auc']:.4f}  "
                      f"Ensemble={r['ensemble_auc']:.4f}")
            avg_xgb = float(np.mean([r["xgb_auc"] for r in all_results]))
            avg_hybrid = float(np.mean([r["hybrid_auc"] for r in all_results]))
            avg_ens = float(np.mean([r["ensemble_auc"] for r in all_results]))
            print(f"    Avg XGB:      {avg_xgb:.4f}")
            print(f"    Avg Hybrid:   {avg_hybrid:.4f}")
            print(f"    Avg Ensemble: {avg_ens:.4f}")

    # ── Train final ensemble on all data ──────────────────────────
    print(f"\n{'=' * 60}")
    print("Training final ensemble on all data")
    print(f"{'=' * 60}")

    target_col = "target_0.5"
    all_merged = _merge_rows(aligned_feats, aligned_tgts, target_col)
    x_all, y_all, xgb_cols, xgb_imputer = _extract_xgb(all_merged, target_col)
    Xs, Xc, y, sm, ss, fm, fs = build_hybrid_sequences(
        aligned_logs, aligned_feats, aligned_tgts, target_col=target_col,
    )

    print(f"  XGB train: {len(x_all)}, Hybrid train: {len(Xs)}")

    final_xgb = _build_model("xgb", SEED, params=_TUNED_XGB)
    final_xgb.fit(x_all, y_all)

    final_hybrid, hybrid_meta = train_hybrid_model(
        Xs, Xc, y,
        hidden_dim=64, n_layers=2, dropout=0.2,
        learning_rate=1e-3, epochs=90, verbose=True,
    )

    # Save both models + ensemble config
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    joblib.dump(final_xgb, os.path.join(ENSEMBLE_DIR, "xgb_model.joblib"))
    joblib.dump(xgb_cols, os.path.join(ENSEMBLE_DIR, "xgb_cols.joblib"))
    joblib.dump(xgb_imputer, os.path.join(ENSEMBLE_DIR, "xgb_imputer.joblib"))

    save_hybrid_model(
        final_hybrid, ENSEMBLE_DIR, sm, ss, fm, fs,
        metadata=hybrid_meta,
    )

    config = {
        "arch": "EnsembleXgbHybrid",
        "models": ["xgb", "hybrid"],
        "weight": 0.5,
        "hybrid_seq_len": SEQUENCE_LEN,
        "target_col": target_col,
        "train_seasons": TRAIN_SEASONS,
    }
    with open(os.path.join(ENSEMBLE_DIR, "ensemble.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Ensemble saved to {ENSEMBLE_DIR}")
    print(f"  XGB rows: {len(x_all)}, Hybrid seqs: {len(Xs)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
