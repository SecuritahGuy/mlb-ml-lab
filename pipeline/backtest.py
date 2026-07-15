"""Walk-forward backtesting with flat-stake betting simulation.

Runs the XGB+hybrid ensemble walk-forward, saves per-game predictions,
then simulates flat $1 bets at -110 odds across multiple confidence
thresholds.

Usage:
    poetry run python pipeline/backtest.py
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.sequence import (
    SEQUENCE_LEN,
    _feat_vec,
    build_hybrid_sequences,
    predict_hybrid_model,
    train_hybrid_model,
)
from mlb_ml_lab.models.train import _build_model, _feature_columns

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
ODDS = -110  # standard betting odds
BREAKEVEN = abs(ODDS) / (abs(ODDS) + 100)  # 0.524 for -110
DECIMAL_ODDS = 1 + 100 / abs(ODDS)  # 1.909 for -110

_TUNED_XGB = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
}


def _merge_rows(
    feat_rows: list[dict], tgt_rows: list[dict], target_col: str
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


def backtest(
    predictions: list[dict],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Run flat-stake betting simulation on walk-forward predictions.

    Each prediction dict must have:
        - ``prob``: predicted probability
        - ``actual``: ground truth (0 or 1)
        - ``target``: e.g. "target_0.5" or "target_1.5"

    Returns a list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    results: list[dict] = []
    for thresh in thresholds:
        filtered = [p for p in predictions if p["prob"] >= thresh]
        if not filtered:
            results.append(
                {
                    "threshold": thresh,
                    "n_bets": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "roi": 0.0,
                    "avg_prob": 0.0,
                    "ev_per_bet": 0.0,
                }
            )
            continue

        n = len(filtered)
        wins = sum(1 for p in filtered if p["actual"] == 1)
        losses = n - wins
        win_rate = wins / n

        net_pnl = wins * (DECIMAL_ODDS - 1) - losses
        roi = net_pnl / n * 100

        ev_per_bet = net_pnl / n

        avg_prob = float(np.mean([p["prob"] for p in filtered]))

        results.append(
            {
                "threshold": thresh,
                "n_bets": n,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 4),
                "breakeven": BREAKEVEN,
                "total_pnl": round(net_pnl, 2),
                "roi": round(roi, 2),
                "avg_prob": round(avg_prob, 4),
                "ev_per_bet": round(ev_per_bet, 4),
            }
        )

    return results


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
    print(f"  Aligned: {len(aligned_logs)} games with both features + targets")

    all_predictions: list[dict] = []

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n{'=' * 60}")
        print(f"Walk-forward ensemble — {target_col}")
        print(f"{'=' * 60}")

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

            x_train, y_tr_xgb, xgb_cols, xgb_imputer = _extract_xgb(
                train_merged,
                target_col,
            )
            x_test, _y_te_xgb, _, _ = _extract_xgb(
                test_merged,
                target_col,
                cols=xgb_cols,
                imputer=xgb_imputer,
            )

            # Train XGB
            xgb_model = _build_model("xgb", SEED, params=_TUNED_XGB)
            xgb_model.fit(x_train, y_tr_xgb)
            xgb_te_proba = xgb_model.predict_proba(x_test)[:, 1]

            # Train hybrid
            hybrid_model, _ = train_hybrid_model(
                Xs_tr,
                Xc_tr,
                y_tr,
                hidden_dim=64,
                n_layers=2,
                dropout=0.2,
                learning_rate=1e-3,
                epochs=60,
                batch_size=512,
                verbose=False,
            )
            hybrid_te_proba = predict_hybrid_model(hybrid_model, Xs_te, Xc_te)

            # Map XGB probabilities to hybrid keys
            xgb_proba_map: dict[tuple[int, int], float] = {}
            for i, row in enumerate(test_merged):
                xgb_proba_map[(row["player_id"], row["game_pk"])] = float(
                    xgb_te_proba[i]
                )

            # Build ensemble predictions for each hybrid test row
            _keys_te: list[tuple[int, int]] = []
            _feat_idx: dict[tuple[int, int], dict] = {}
            for f in test_feat:
                _feat_idx[(f["player_id"], f["game_pk"])] = f

            _grouped: dict[tuple[int, str], list[tuple[int, Any]]] = defaultdict(list)
            for i, lg in enumerate(test_logs):
                _grouped[(lg.player_id, str(lg.season))].append((i, lg))

            for (pid, _season), entries in _grouped.items():
                entries.sort(key=lambda e: e[1].date)
                indices = [e[0] for e in entries]
                vecs = [_feat_vec(e[1]) for e in entries]
                for pos in range(SEQUENCE_LEN, len(vecs)):
                    idx = indices[pos]
                    lg = test_logs[idx]
                    if _feat_idx.get((lg.player_id, lg.game_pk)):
                        _keys_te.append((lg.player_id, lg.game_pk))

            for i, (pid, gpk) in enumerate(_keys_te):
                hp = float(hybrid_te_proba[i])
                xp = xgb_proba_map.get((pid, gpk), hp)
                ensemble_prob = (hp + xp) / 2.0
                all_predictions.append(
                    {
                        "player_id": pid,
                        "game_pk": gpk,
                        "target": target_col,
                        "prob": ensemble_prob,
                        "actual": int(y_te[i]),
                        "fold": fold_idx + 1,
                        "test_season": test_season,
                    }
                )

            n_ens = len(_keys_te)
            fold_auc = float(np.nan) if n_ens == 0 else float(np.nan)
            if n_ens > 0:
                fold_auc = roc_auc_score(
                    [p["actual"] for p in all_predictions[-n_ens:]],
                    [p["prob"] for p in all_predictions[-n_ens:]],
                )
            print(f"    Ensemble AUC: {fold_auc:.4f}  ({n_ens} games)")

    # ── Backtest simulation ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Backtest simulation (flat $1 bets at -110 odds)")
    print(f"{'=' * 60}")

    for target_col in ("target_0.5", "target_1.5"):
        preds = [p for p in all_predictions if p["target"] == target_col]
        print(f"\n  --- {target_col} ({len(preds)} predictions across all folds) ---")

        results = backtest(preds)
        print(
            f"  {'Thresh':>6}  {'Bets':>6}  {'WinRate':>8}  {'BE':>5}  "
            f"{'P&L':>8}  {'ROI':>6}  {'AvgProb':>8}"
        )
        print(
            f"  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 5}  "
            f"{'-' * 8}  {'-' * 6}  {'-' * 8}"
        )
        for r in results:
            if r["n_bets"] == 0:
                continue
            print(
                f"  {r['threshold']:>6.2f}  {r['n_bets']:>6}  "
                f"{r['win_rate']:>8.4f}  {r['breakeven']:>5.3f}  "
                f"{r['total_pnl']:>8.2f}  {r['roi']:>6.2f}%  "
                f"{r['avg_prob']:>8.4f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
