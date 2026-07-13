"""Aggregate player-level hit predictions to team-game level.

Walk-forward ensemble → sum individual hit probabilities by team-game → 
compare to actual team hit totals. Evaluates whether aggregating players
captures game-level signal that individual props miss.

Usage:
    poetry run python pipeline/game_aggregation.py
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.sequence import (
    SEQUENCE_LEN,
    _feat_vec,
    build_hybrid_sequences,
    predict_hybrid_model,
    train_hybrid_model,
)
from mlb_ml_lab.models.train import _build_model, _feature_columns

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
ODDS = -110
BREAKEVEN = abs(ODDS) / (abs(ODDS) + 100)
DECIMAL_ODDS = 1 + 100 / abs(ODDS)

_TUNED_XGB = {
    "n_estimators": 500, "max_depth": 5, "learning_rate": 0.01,
    "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1,
}


def _merge_rows(feat_rows: list[dict], tgt_rows: list[dict], target_col: str) -> list[dict]:
    merged: list[dict] = []
    for fr, tr in zip(feat_rows, tgt_rows):
        row = dict(fr)
        row[target_col] = tr[target_col]
        merged.append(row)
    return merged


def _extract_xgb(
    merged_rows: list[dict], target_col: str, cols: list[str] | None = None,
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

    target_col = "target_0.5"
    team_game_predictions: dict[tuple[int, int], dict] = {}

    print(f"\n{'=' * 60}")
    print("Walk-forward aggregation to team-game level")
    print(f"{'=' * 60}")

    for fold_idx in range(len(TRAIN_SEASONS) - 1):
        train_cutoff = TRAIN_SEASONS[fold_idx]
        test_season = TRAIN_SEASONS[fold_idx + 1]

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
        print(f"    Train: {len(train_logs)} logs, Test: {len(test_logs)} logs")

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

        train_merged = _merge_rows(train_feat, train_tgt, target_col)
        test_merged = _merge_rows(test_feat, test_tgt, target_col)

        x_train, y_tr_xgb, xgb_cols, xgb_imputer = _extract_xgb(
            train_merged, target_col,
        )
        x_test, y_te_xgb, _, _ = _extract_xgb(
            test_merged, target_col, cols=xgb_cols, imputer=xgb_imputer,
        )

        xgb_model = _build_model("xgb", SEED, params=_TUNED_XGB)
        xgb_model.fit(x_train, y_tr_xgb)
        xgb_te_proba = xgb_model.predict_proba(x_test)[:, 1]

        hybrid_model, _ = train_hybrid_model(
            Xs_tr, Xc_tr, y_tr,
            hidden_dim=64, n_layers=2, dropout=0.2,
            learning_rate=1e-3, epochs=60, batch_size=512,
            verbose=False,
        )
        hybrid_te_proba = predict_hybrid_model(hybrid_model, Xs_te, Xc_te)

        xgb_proba_map: dict[tuple[int, int], float] = {}
        for i, row in enumerate(test_merged):
            xgb_proba_map[(row["player_id"], row["game_pk"])] = float(xgb_te_proba[i])

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

        for i, (pid, gpk) in enumerate(_keys_te):
            hp = float(hybrid_te_proba[i])
            xp = xgb_proba_map.get((pid, gpk), hp)
            ensemble_prob = (hp + xp) / 2.0

            # Find the PlayerGameLog for this player+game
            for lg in test_logs:
                if lg.player_id == pid and lg.game_pk == gpk:
                    key = (lg.team_id, gpk)
                    if key not in team_game_predictions:
                        team_game_predictions[key] = {
                            "team_id": lg.team_id,
                            "game_pk": gpk,
                            "date": lg.date,
                            "opponent_id": lg.opponent_id,
                            "is_home": lg.is_home,
                            "players": [],
                            "actual_team_hits": 0,
                        }
                    team_game_predictions[key]["players"].append({
                        "player_id": pid,
                        "prob": ensemble_prob,
                        "actual_hits": lg.hits,
                    })
                    team_game_predictions[key]["actual_team_hits"] += lg.hits
                    break

    # ── Team-level evaluation ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Team-level aggregation results")
    print(f"{'=' * 60}")

    rows = list(team_game_predictions.values())
    print(f"\n  {len(rows)} team-game slots with predictions")

    # Per team-game: expected hits = sum of each player's hit probability
    # Actual hits from game logs
    expected_total = []
    actual_total = []
    n_players_list = []
    for r in rows:
        expected = sum(p["prob"] for p in r["players"])
        actual = r["actual_team_hits"]
        expected_total.append(expected)
        actual_total.append(actual)
        n_players_list.append(len(r["players"]))

    expected_total = np.array(expected_total)
    actual_total = np.array(actual_total)
    n_players = np.array(n_players_list)

    # Overall stats
    total_actual = actual_total.sum()
    total_expected = expected_total.sum()
    mae = np.abs(actual_total - expected_total).mean()
    rmse = np.sqrt(((actual_total - expected_total) ** 2).mean())
    corr = np.corrcoef(expected_total, actual_total)[0, 1]

    print(f"\n  Total expected hits: {total_expected:.0f}")
    print(f"  Total actual hits:   {total_actual}")
    print(f"  Bias: {total_expected - total_actual:.1f} hits "
          f"({((total_expected - total_actual) / total_actual) * 100:+.2f}%)")
    print(f"  MAE: {mae:.2f} hits/team-game")
    print(f"  RMSE: {rmse:.2f} hits/team-game")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Avg players/team-game: {n_players.mean():.1f}")

    # By number of players predicted
    print("\n  Accuracy by roster coverage:")
    for bracket in [(3, 5), (5, 7), (7, 9), (9, 99)]:
        lo, hi = bracket
        mask = (n_players >= lo) & (n_players < hi)
        if mask.sum() == 0:
            continue
        e, a = expected_total[mask], actual_total[mask]
        bias = (e.sum() - a.sum()) / a.sum() * 100
        print(f"    {lo}-{hi} players ({mask.sum():>5} games): "
              f"MAE={np.abs(a - e).mean():.2f}  "
              f"bias={bias:+.2f}%  "
              f"corr={np.corrcoef(e, a)[0, 1]:.4f}")

    # ── Team-level betting simulation ────────────────────────────
    print("\n  --- Flat-stake betting on team total hits ---")
    print(f"  Odds: -110 (breakeven {BREAKEVEN:.1%})")
    print(f"  {'Thresh':>8}  {'Bets':>6}  {'WinRate':>8}  {'P&L':>8}  {'ROI':>7}")

    # Strategy: Over X team hits when predicted total > threshold
    for thresh in np.arange(3.5, 10.5, 0.5):
        bets = 0
        wins = 0
        for e, a in zip(expected_total, actual_total):
            if e >= thresh:
                bets += 1
                if a > thresh:
                    wins += 1
        if bets < 50:
            continue
        win_rate = wins / bets
        net_pnl = wins * (DECIMAL_ODDS - 1) - (bets - wins)
        roi = net_pnl / bets * 100
        print(f"  Over {thresh:>4.1f}  {bets:>6}  {win_rate:>8.4f}  "
              f"{net_pnl:>8.2f}  {roi:>7.2f}%")

    # Strategy: Under X team hits when predicted total < threshold
    print()
    for thresh in np.arange(3.5, 10.5, 0.5):
        bets = 0
        wins = 0
        for e, a in zip(expected_total, actual_total):
            if e <= thresh:
                bets += 1
                if a < thresh:
                    wins += 1
        if bets < 50:
            continue
        win_rate = wins / bets
        net_pnl = wins * (DECIMAL_ODDS - 1) - (bets - wins)
        roi = net_pnl / bets * 100
        print(f"  Under {thresh:>4.1f}  {bets:>6}  {win_rate:>8.4f}  "
              f"{net_pnl:>8.2f}  {roi:>7.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
