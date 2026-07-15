"""Wire real SBR moneyline odds into the walk-forward ensemble for a true +EV backtest.

Two evaluations:

1. Moneyline +EV — the ensemble predicts per-player P(hits >= 1). We sum those
   to a team's expected hits, take the difference vs the opponent, and bridge
   that to a win probability with a walk-forward logistic fit on *real* team
   hit-differences (no future leakage). We then compare the model's win
   probability to the market's vig-free implied probability from SBR
   moneylines, bet when the edge is positive, and settle on the actual game
   result. Flat-stake and Kelly simulations report ROI, win rate, and max
   drawdown, plus a calibration check on the moneyline model itself.

2. Player-prop calibration — the model's core output is P(hits >= k) for
   k in {1, 2}. We bucket those probabilities against realized frequencies and
   report ECE (expected calibration error), for both the raw ensemble and a
   per-season isotonic recalibration (cross-fit within each season). SBR's free
   page does not expose player hit-prop odds to compare against directly.

Usage:
    poetry run python pipeline/odds_backtest.py
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from mlb_ml_lab import MlbClient, PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.data.odds import fetch_game_odds, load_cached_odds, save_cached_odds
from mlb_ml_lab.evaluation.backtest import (
    GamePrediction,
    calibration_buckets,
    expected_calibration_error,
)
from mlb_ml_lab.models.sequence import (
    SEQUENCE_LEN,
    build_hybrid_sequences,
    predict_hybrid_model,
    train_hybrid_model,
)
from mlb_ml_lab.models.train import _build_model, _feature_columns

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2016_2026_30teams"
TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
SEED = 42
MIN_PLAYERS = 6  # require a reasonable lineup share to bet a team-game

_TUNED_XGB = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
}


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------


def ml_to_implied_prob(ml: int | float | None) -> float | None:
    """Convert American moneyline odds to implied probability (with vig)."""
    if ml is None or abs(ml) >= 10000:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 100.0 / (ml + 100.0)


def american_to_decimal(ml: int | float) -> float:
    """Convert American odds to decimal odds (payout multiple incl. stake)."""
    if ml < 0:
        return 1.0 + 100.0 / abs(ml)
    return 1.0 + ml / 100.0


def fair_prob(ml_ours: int, ml_opp: int) -> float:
    """Vig-free implied win probability for 'our' team given both moneylines."""
    p_ours = ml_to_implied_prob(ml_ours)
    p_opp = ml_to_implied_prob(ml_opp)
    if p_ours is None or p_opp is None:
        return 0.5
    total = p_ours + p_opp
    if total <= 0:
        return 0.5
    return p_ours / total


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


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


def _build_hybrid_keys(
    logs: list[PlayerGameLog], feat_rows: list[dict], tgt_rows: list[dict]
) -> list[tuple[int, int]]:
    """Reconstruct (player_id, game_pk) order produced by build_hybrid_sequences.

    Must mirror the skip logic inside build_hybrid_sequences exactly so the
    returned keys align 1:1 with its sequence outputs.
    """
    feat_index = {(f["player_id"], f["game_pk"]): f for f in feat_rows}
    tgt_index = {(t["player_id"], t["game_pk"]): t for t in tgt_rows}
    grouped: dict[tuple[int, str], list] = defaultdict(list)
    for i, lg in enumerate(logs):
        grouped[(lg.player_id, str(lg.season))].append((i, lg))

    keys: list[tuple[int, int]] = []
    for (_pid, _season), entries in grouped.items():
        entries.sort(key=lambda e: e[1].date)
        for pos in range(SEQUENCE_LEN, len(entries)):
            idx = entries[pos][0]
            lg = logs[idx]
            if feat_index.get((lg.player_id, lg.game_pk)) is None:
                continue
            if tgt_index.get((lg.player_id, lg.game_pk)) is None:
                continue
            keys.append((lg.player_id, lg.game_pk))
    return keys


def _actual_team_games(
    logs: list[PlayerGameLog],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Aggregate actual hits + win flag per (team_id, game_pk)."""
    agg: dict[tuple[int, int], dict[str, Any]] = {}
    for lg in logs:
        key = (lg.team_id, lg.game_pk)
        if key not in agg:
            agg[key] = {
                "team_id": lg.team_id,
                "game_pk": lg.game_pk,
                "date": lg.date[:10],
                "opponent_id": lg.opponent_id,
                "hits": 0,
                "n_players": 0,
                "win": bool(lg.is_win),
            }
        agg[key]["hits"] += int(lg.hits)
        agg[key]["n_players"] += 1
    return agg


def _ensemble_map(
    keys: list[tuple[int, int]],
    hybrid_proba: np.ndarray,
    xgb_map: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Per-(player, game) ensemble probability = mean(hybrid, xgb)."""
    return {
        k: (float(hybrid_proba[i]) + xgb_map.get(k, float(hybrid_proba[i]))) / 2.0
        for i, k in enumerate(keys)
    }


def _expected_team_games(
    logs: list[PlayerGameLog],
    keys: list[tuple[int, int]],
    prob_map: dict[tuple[int, int], float],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Aggregate ensemble hit probability to expected team hits per game."""
    games: dict[tuple[int, int], dict[str, Any]] = {}
    for _i, (pid, gpk) in enumerate(keys):
        ensemble = prob_map.get((pid, gpk), 0.0)
        lg = next(row for row in logs if row.player_id == pid and row.game_pk == gpk)
        key = (lg.team_id, lg.game_pk)
        if key not in games:
            games[key] = {
                "team_id": lg.team_id,
                "game_pk": lg.game_pk,
                "date": lg.date[:10],
                "opponent_id": lg.opponent_id,
                "exp_hits": 0.0,
                "n_players": 0,
            }
        games[key]["exp_hits"] += ensemble
        games[key]["n_players"] += 1
    return games


# ---------------------------------------------------------------------------
# Bridge: expected hit-diff -> win probability (walk-forward, train only)
# ---------------------------------------------------------------------------


def _fit_win_bridge(train_logs: list[PlayerGameLog]) -> LogisticRegression:
    """Fit logistic regression mapping real team hit-diff to win.

    Trained strictly on training data (no future leakage). The same coefficient
    then maps *expected* hit-diff (from the model) to a win probability.
    """
    tg = _actual_team_games(train_logs)
    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    for _key, g in tg.items():
        opp = tg.get((g["opponent_id"], g["game_pk"]))
        if opp is None:
            continue
        diff = g["hits"] - opp["hits"]
        rows_x.append([diff])
        rows_y.append(1 if g["win"] else 0)
    clf = LogisticRegression()
    clf.fit(np.array(rows_x), np.array(rows_y))
    return clf


# ---------------------------------------------------------------------------
# Probability calibration (fix overconfidence) — walk-forward, train only
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _fit_temperature(train_p: np.ndarray, train_y: np.ndarray) -> float:
    """Grid-search the temperature T (softmax-style) minimising Brier on train."""
    best_T, best_b = 1.0, float("inf")
    for threshold in np.linspace(0.5, 2.5, 41):
        sp = 1.0 / (1.0 + np.exp(-_logit(train_p) / threshold))
        b = float(np.mean((sp - train_y) ** 2))
        if b < best_b:
            best_b, best_T = b, threshold
    return best_T


def _fit_platt(train_p: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    """Platt scaling: logistic regression on the logit of the raw probability."""
    lr = LogisticRegression()
    lr.fit(_logit(train_p).reshape(-1, 1), train_y)
    return lr


def fit_calibrator(
    train_p: np.ndarray,
    train_y: np.ndarray,
) -> tuple[str, float, LogisticRegression | None]:
    """Fit a calibrator on training predictions, choosing Platt vs temperature
    by lower Brier score. Returns ``(method, temperature, platt_model)``."""
    threshold = _fit_temperature(train_p, train_y)
    p_temp = 1.0 / (1.0 + np.exp(-_logit(train_p) / threshold))
    b_temp = float(np.mean((p_temp - train_y) ** 2))

    platt = _fit_platt(train_p, train_y)
    p_platt = platt.predict_proba(_logit(train_p).reshape(-1, 1))[:, 1]
    b_platt = float(np.mean((p_platt - train_y) ** 2))

    if b_platt <= b_temp:
        return "platt", threshold, platt
    return "temp", threshold, None


def apply_calibrator(
    p: np.ndarray,
    cal: tuple[str, float, LogisticRegression | None],
) -> np.ndarray:
    """Apply a calibrator fitted by ``fit_calibrator``."""
    method, threshold, platt = cal
    if method == "platt" and platt is not None:
        return platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    return 1.0 / (1.0 + np.exp(-_logit(p) / threshold))


def _fit_oof_calibrator(
    target_col: str,
    cal_train_logs,
    cal_train_feat,
    cal_train_tgt,
    cal_hold_logs,
    cal_hold_feat,
    cal_hold_tgt,
    tgt_by_key: dict,
) -> tuple[str, float, LogisticRegression | None] | None:
    """Fit a calibrator on OUT-OF-SAMPLE predictions.

    Trains the ensemble on ``cal_train_*`` and predicts the held-out season
    ``cal_hold_*`` — predictions the model never trained on. This avoids the
    in-sample optimism of fitting a calibrator on a model's own training data.
    """
    keys, hybrid, xgb_map, _, _, _ = _train_predict(
        cal_train_logs,
        cal_train_feat,
        cal_train_tgt,
        cal_hold_logs,
        cal_hold_feat,
        cal_hold_tgt,
        target_col,
    )
    if keys is None or len(keys) == 0:
        return None
    p = np.array(
        [
            (float(hybrid[i]) + xgb_map.get(k, float(hybrid[i]))) / 2.0
            for i, k in enumerate(keys)
        ]
    )
    y = np.array([int(tgt_by_key[k][target_col]) for k in keys])
    return fit_calibrator(p, y)


def _isotonic_cv(
    raw_p: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """Leakage-free per-season isotonic calibration via internal cross-fitting.

    The raw probabilities are already out-of-sample (the model was trained on
    earlier seasons). We recalibrate them against this season's own labels using
    K-fold cross-fitting so no player's calibrated probability is trained on its
    own target. A separate isotonic curve per season tracks temporal
    distribution drift far better than a single global Platt/temperature scaler,
    which over-corrected when applied to later seasons (ECE 0.07 -> 0.17).
    """
    raw_p = np.asarray(raw_p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(raw_p) < n_splits * 10:
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(raw_p, y)
        return ir.predict(raw_p)
    out = np.empty(len(raw_p), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in kf.split(raw_p):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(raw_p[tr], y[tr])
        out[te] = ir.predict(raw_p[te])
    return out


# ---------------------------------------------------------------------------
# Ensemble training + prediction for one target threshold
# ---------------------------------------------------------------------------


def _train_predict(
    train_logs,
    train_feat,
    train_tgt,
    test_logs,
    test_feat,
    test_tgt,
    target_col,
):
    """Train hybrid + xgb on ``target_col`` and return test ensemble per key."""
    Xs_tr, Xc_tr, y_tr, sm, ss, fm, fs = build_hybrid_sequences(
        train_logs,
        train_feat,
        train_tgt,
        target_col=target_col,
    )
    Xs_te, Xc_te, _y_te, _, _, _, _ = build_hybrid_sequences(
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
        return None, None, None

    hybrid_model, _ = train_hybrid_model(
        Xs_tr,
        Xc_tr,
        y_tr,
        hidden_dim=64,
        n_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=40,
        batch_size=512,
        verbose=False,
    )
    hybrid_proba = predict_hybrid_model(hybrid_model, Xs_te, Xc_te)

    train_merged = _merge_rows(train_feat, train_tgt, target_col)
    test_merged = _merge_rows(test_feat, test_tgt, target_col)
    x_train, y_tr_xgb, xgb_cols, xgb_imputer = _extract_xgb(train_merged, target_col)
    x_test, _, _, _ = _extract_xgb(
        test_merged,
        target_col,
        cols=xgb_cols,
        imputer=xgb_imputer,
    )
    xgb_model = _build_model("xgb", SEED, params=_TUNED_XGB)
    xgb_model.fit(x_train, y_tr_xgb)
    xgb_proba = xgb_model.predict_proba(x_test)[:, 1]
    xgb_tr_proba = xgb_model.predict_proba(x_train)[:, 1]

    xgb_map = {
        (row["player_id"], row["game_pk"]): float(p)
        for row, p in zip(test_merged, xgb_proba)
    }
    keys = _build_hybrid_keys(test_logs, test_feat, test_tgt)
    keys_tr = _build_hybrid_keys(train_logs, train_feat, train_tgt)
    hybrid_tr_proba = predict_hybrid_model(hybrid_model, Xs_tr, Xc_tr)
    xgb_tr_map = {
        (row["player_id"], row["game_pk"]): float(p)
        for row, p in zip(train_merged, xgb_tr_proba)
    }
    return keys, hybrid_proba, xgb_map, keys_tr, hybrid_tr_proba, xgb_tr_map


# ---------------------------------------------------------------------------
# Betting simulation
# ---------------------------------------------------------------------------


def _simulate(records: list[dict], mode: str, min_edge: float) -> dict[str, float]:
    """Simulate moneyline bets.

    ``records`` carry model_prob, decimal_odds, win (bool), edge. Bets fire
    when edge >= min_edge. mode='flat' stakes 1 unit; mode='kelly' uses full
    Kelly capped at 1 unit.
    """
    bets = [r for r in records if r["edge"] >= min_edge and r["decimal_odds"] > 1.0]
    n = len(bets)
    if n == 0:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "roi": 0.0,
            "max_dd": 0.0,
            "avg_edge": 0.0,
        }

    # Equity starts at `n` units (one unit per bet) so drawdown is bounded.
    equity = float(n)
    peak = float(n)
    max_dd = 0.0
    wins = 0
    total_stake = 0.0
    pnl = 0.0
    edges = []
    for r in bets:
        d = r["decimal_odds"]
        if mode == "kelly":
            f = (r["model_prob"] * d - 1.0) / (d - 1.0)
            stake = max(0.0, min(f, 1.0))
        else:
            stake = 1.0
        if stake <= 0:
            continue
        edges.append(r["edge"])
        total_stake += stake
        if r["win"]:
            profit = stake * (d - 1.0)
            wins += 1
        else:
            profit = -stake
        pnl += profit
        equity += profit
        peak = max(peak, equity)
        if peak > 1e-9:
            max_dd = max(max_dd, (peak - equity) / peak)

    return {
        "bets": n,
        "wins": wins,
        "win_rate": wins / n,
        "pnl": pnl,
        "roi": pnl / total_stake if total_stake > 0 else 0.0,
        "max_dd": max_dd,
        "avg_edge": float(np.mean(edges)),
    }


def _build_moneyline_rows(
    exp_games: dict,
    actual_test: dict,
    win_bridge: LogisticRegression,
    id_to_abbrev: dict[int, str],
    odds_index: dict,
) -> list[dict]:
    """Match team-games to SBR moneylines and emit moneyline bet records."""
    rows: list[dict] = []
    for key, g in exp_games.items():
        team_abbrev = id_to_abbrev.get(g["team_id"], "")
        opp_abbrev = id_to_abbrev.get(g["opponent_id"], "")
        opp = exp_games.get((g["opponent_id"], g["game_pk"]))
        if (
            opp is None
            or g["n_players"] < MIN_PLAYERS
            or opp["n_players"] < MIN_PLAYERS
        ):
            continue
        ds = g["date"]
        odds_row = odds_index.get((ds, opp_abbrev, team_abbrev))
        if odds_row is None:
            odds_row = odds_index.get((ds, team_abbrev, opp_abbrev))
        if odds_row is None:
            continue
        if odds_row["home_team"] == team_abbrev:
            ml_ours, ml_opp = odds_row["home_ml"], odds_row["away_ml"]
        else:
            ml_ours, ml_opp = odds_row["away_ml"], odds_row["home_ml"]
        if ml_ours is None or ml_opp is None:
            continue

        exp_diff = g["exp_hits"] - opp["exp_hits"]
        model_prob = float(win_bridge.predict_proba([[exp_diff]])[0, 1])
        fair = fair_prob(ml_ours, ml_opp)
        decimal = american_to_decimal(ml_ours)
        if decimal <= 1.0:
            continue
        rows.append(
            {
                "date": ds,
                "team": team_abbrev,
                "opp": opp_abbrev,
                "model_prob": model_prob,
                "fair_prob": fair,
                "decimal_odds": decimal,
                "edge": model_prob - fair,
                "win": bool(actual_test[key]["win"]),
                "exp_diff": exp_diff,
            }
        )
    return rows


def _print_sim(title: str, rows: list[dict], mode: str) -> None:
    print(f"\n  {title}:")
    print(
        f"  {'MinEdge':>8} {'Bets':>6} {'Win%':>7} {'P&L':>9} {'ROI':>8} "
        f"{'MaxDD':>7} {'AvgEdge':>8}"
    )
    for thr in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
        r = _simulate(rows, mode, thr)
        if r["bets"] == 0:
            continue
        print(
            f"  {thr:>8.2f} {r['bets']:>6} {r['win_rate'] * 100:>6.2f}% "
            f"{r['pnl']:>9.1f} {r['roi'] * 100:>7.2f}% "
            f"{r['max_dd'] * 100:>6.1f}% {r['avg_edge']:>8.3f}"
        )


def main() -> None:
    print("Loading data...")
    raw_logs = load_game_logs(CACHED_DATASET)
    feature_matrix, targets_list, _meta = load_feature_data(CACHED_DATASET)
    print(f"  {len(raw_logs)} game logs, {len(feature_matrix)} feature rows")

    client = MlbClient()
    teams = client.get_teams()
    id_to_abbrev: dict[int, str] = {}
    for t in teams:
        if t.get("sport", {}).get("id") == 1:
            id_to_abbrev[t["id"]] = t.get("abbreviation", "")

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

    feat_by_key: dict[tuple[int, int], dict[str, Any]] = {
        (f["player_id"], f["game_pk"]): f for f in feature_matrix
    }
    tgt_by_key: dict[tuple[int, int], dict[str, Any]] = {
        (t["player_id"], t["game_pk"]): t for t in targets_list
    }

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

    test_dates = sorted(
        set(
            d["date"][:10]
            for d in aligned_tgts
            if d["date"][:4] in ("2022", "2023", "2024", "2025")
        )
    )
    print(f"  {len(test_dates)} unique dates in test seasons (2022-2025)")

    print("\nFetching historical SBR odds (cached after first run)...")
    date_odds: dict[str, list[dict[str, Any]]] = {}
    cached = fetched = 0
    for ds in test_dates:
        odds = load_cached_odds(ds)
        if odds is not None:
            date_odds[ds] = odds
            cached += 1
        else:
            odds = fetch_game_odds(ds, sportsbook="betmgm")
            date_odds[ds] = odds
            save_cached_odds(ds, odds)
            fetched += 1
            time.sleep(0.3)
    print(f"  Odds: {cached} cached + {fetched} fetched = {len(date_odds)} dates")

    odds_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ds, odds_list in date_odds.items():
        for g in odds_list:
            odds_index[(ds, g["away_team"], g["home_team"])] = g
    print(f"  {len(odds_index)} game-odds entries indexed")

    moneyline_rows_raw: list[dict] = []
    moneyline_rows_cal: list[dict] = []
    prop_preds_05_raw: list[tuple[float, int]] = []
    prop_preds_15_raw: list[tuple[float, int]] = []
    prop_preds_05_iso: list[tuple[float, int]] = []
    prop_preds_15_iso: list[tuple[float, int]] = []

    for fold_idx in range(len(TRAIN_SEASONS) - 1):
        train_cutoff = TRAIN_SEASONS[fold_idx]
        test_season = TRAIN_SEASONS[fold_idx + 1]

        train_mask = [int(t["date"][:4]) <= train_cutoff for t in aligned_tgts]
        test_mask = [int(t["date"][:4]) == test_season for t in aligned_tgts]

        train_logs = [lg for lg, m in zip(aligned_logs, train_mask) if m]
        test_logs = [lg for lg, m in zip(aligned_logs, test_mask) if m]
        train_feat = [f for f, m in zip(aligned_feats, train_mask) if m]
        test_feat = [f for f, m in zip(aligned_feats, test_mask) if m]
        train_tgt = [t for t, m in zip(aligned_tgts, train_mask) if m]
        test_tgt = [t for t, m in zip(aligned_tgts, test_mask) if m]

        print(f"\n  Fold {fold_idx + 1}: train <= {train_cutoff}, test={test_season}")
        print(f"    Train: {len(train_logs)} logs, Test: {len(test_logs)} logs")

        keys_05, hybrid_05, xgb_05, keys_05_tr, hybrid_05_tr, xgb_05_tr = (
            _train_predict(
                train_logs,
                train_feat,
                train_tgt,
                test_logs,
                test_feat,
                test_tgt,
                "target_0.5",
            )
        )
        keys_15, hybrid_15, xgb_15, _keys_15_tr, _hybrid_15_tr, _xgb_15_tr = (
            _train_predict(
                train_logs,
                train_feat,
                train_tgt,
                test_logs,
                test_feat,
                test_tgt,
                "target_1.5",
            )
        )
        if keys_05 is None or keys_15 is None:
            print("    Skipping — no sequences")
            continue

        # Raw ensemble probability maps (mean of hybrid + xgb), per (player, game)
        raw_map_05 = _ensemble_map(keys_05, hybrid_05, xgb_05)
        raw_map_15 = _ensemble_map(keys_15, hybrid_15, xgb_15)

        # Fit calibrators on OUT-OF-SAMPLE predictions: hold out the most
        # recent train season, train on the rest, predict the hold-out season.
        cal_holdout = max(s for s in TRAIN_SEASONS if s <= train_cutoff)
        cal_train_seasons = [
            s for s in TRAIN_SEASONS if s <= train_cutoff and s != cal_holdout
        ]
        if cal_train_seasons:
            ct_mask = [int(t["date"][:4]) in cal_train_seasons for t in aligned_tgts]
            ch_mask = [int(t["date"][:4]) == cal_holdout for t in aligned_tgts]
            ct_logs = [lg for lg, m in zip(aligned_logs, ct_mask) if m]
            ct_feat = [f for f, m in zip(aligned_feats, ct_mask) if m]
            ct_tgt = [t for t, m in zip(aligned_tgts, ct_mask) if m]
            ch_logs = [lg for lg, m in zip(aligned_logs, ch_mask) if m]
            ch_feat = [f for f, m in zip(aligned_feats, ch_mask) if m]
            ch_tgt = [t for t, m in zip(aligned_tgts, ch_mask) if m]
            cal_05 = _fit_oof_calibrator(
                "target_0.5",
                ct_logs,
                ct_feat,
                ct_tgt,
                ch_logs,
                ch_feat,
                ch_tgt,
                tgt_by_key,
            )
        else:
            # No season to hold out (earliest fold) — fall back to in-sample.
            tr_p_05 = np.array(
                [
                    (float(hybrid_05_tr[i]) + xgb_05_tr.get(k, float(hybrid_05_tr[i])))
                    / 2.0
                    for i, k in enumerate(keys_05_tr)
                ]
            )
            tr_y_05 = np.array([int(tgt_by_key[k]["target_0.5"]) for k in keys_05_tr])
            cal_05 = fit_calibrator(tr_p_05, tr_y_05)

        cal_map_05 = {
            k: float(apply_calibrator(np.array([v]), cal_05)[0])
            for k, v in raw_map_05.items()
        }

        # Expected hits -> team games (test), raw and calibrated
        exp_games_raw = _expected_team_games(test_logs, keys_05, raw_map_05)
        exp_games_cal = _expected_team_games(test_logs, keys_05, cal_map_05)
        actual_test = _actual_team_games(test_logs)

        # Win bridge from training actual hit-diff
        win_bridge = _fit_win_bridge(train_logs)

        # Player-prop predictions for calibration.
        #  - RAW: the ensemble output, unchanged.
        #  - ISOTONIC: per-season isotonic recalibration (cross-fit within the
        #    season). Platt/temp is intentionally NOT used here — a single
        #    global scaler over-corrected across seasons (ECE 0.07 -> 0.17).
        for k, v in raw_map_05.items():
            prop_preds_05_raw.append((v, int(tgt_by_key[k]["target_0.5"])))
        for k, v in raw_map_15.items():
            prop_preds_15_raw.append((v, int(tgt_by_key[k]["target_1.5"])))

        iso_05 = _isotonic_cv(
            np.array(list(raw_map_05.values())),
            np.array([int(tgt_by_key[k]["target_0.5"]) for k in raw_map_05]),
        )
        for k, p in zip(raw_map_05.keys(), iso_05):
            prop_preds_05_iso.append((float(p), int(tgt_by_key[k]["target_0.5"])))
        iso_15 = _isotonic_cv(
            np.array(list(raw_map_15.values())),
            np.array([int(tgt_by_key[k]["target_1.5"]) for k in raw_map_15]),
        )
        for k, p in zip(raw_map_15.keys(), iso_15):
            prop_preds_15_iso.append((float(p), int(tgt_by_key[k]["target_1.5"])))

        # Moneyline rows, raw and calibrated
        rows_raw = _build_moneyline_rows(
            exp_games_raw, actual_test, win_bridge, id_to_abbrev, odds_index
        )
        rows_cal = _build_moneyline_rows(
            exp_games_cal, actual_test, win_bridge, id_to_abbrev, odds_index
        )
        moneyline_rows_raw.extend(rows_raw)
        moneyline_rows_cal.extend(rows_cal)

        print(
            f"    cal_05={cal_05[0]} (T={cal_05[1]:.2f}); "
            f"matched {len(rows_cal)} moneyline rows"
        )

    # ------------------------------------------------------------------
    # Moneyline +EV analysis — raw vs calibrated
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("MONEYLINE +EV BACKTEST (real SBR odds, settled on game result)")
    print(f"{'=' * 60}")

    for tag, rows in [
        ("RAW (uncalibrated)", moneyline_rows_raw),
        ("CALIBRATED (Platt/temp)", moneyline_rows_cal),
    ]:
        print(f"\n  === {tag}  (n={len(rows)}) ===")
        _print_sim("Flat stake", rows, "flat")
        _print_sim("Full Kelly (cap 1 unit)", rows, "kelly")

        print("    Moneyline model calibration (model_prob vs realized win%):")
        rows_sorted = sorted(rows, key=lambda r: r["model_prob"])
        n = len(rows_sorted)
        for lo in range(10):
            bucket = rows_sorted[int(lo * n / 10) : int((lo + 1) * n / 10)]
            if not bucket:
                continue
            mp = np.mean([b["model_prob"] for b in bucket])
            wr = np.mean([1.0 if b["win"] else 0.0 for b in bucket])
            print(f"      p~{mp:.2f}  win%={wr * 100:5.1f}  n={len(bucket)}")

    # ------------------------------------------------------------------
    # Player-prop calibration — RAW plus per-season ISOTONIC recalibration.
    # Platt/temperature scaling is NOT applied to per-player probabilities:
    # fit on one season it over-corrects when applied to the next (temporal
    # distribution shift), raising ECE from ~0.070 -> ~0.170 on target_0.5.
    # The moneyline section above still uses the calibrated ensemble (there
    # the aggregate hit-edge signal benefits from the rescaling).
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PLAYER-PROP CALIBRATION (model prob vs realized frequency)")
    print(f"{'=' * 60}")
    print("  RAW      = ensemble output, unchanged.")
    print("  ISOTONIC = per-season isotonic recalibration (cross-fit within")
    print("             each season; tracks temporal drift).")
    print("  Platt/temp is applied moneyline-only (hurts per-player ECE).")

    def _print_calib(label: str, preds: list[tuple[float, int]], tcol: str) -> None:
        gp = [
            GamePrediction(
                date=None,
                player_id=0,
                game_pk=0,
                predicted_prob=float(p),
                actual=a,
                hits=a,
                target_col=tcol,
            )
            for p, a in preds
        ]
        ece = expected_calibration_error(gp, n_bins=10)
        obs_rate = np.mean([a for _, a in preds])
        print(f"\n  {label}: n={len(preds)}, base rate={obs_rate:.3f}, ECE={ece:.4f}")
        print("      Bucket   Predicted  Observed   Count")
        print("      " + "-" * 42)
        for b in calibration_buckets(gp, n_bins=10):
            print(
                f"      [{b['bin_lower']:.2f}-{b['bin_upper']:.2f})  "
                f"{b['mean_predicted']:.4f}     {b['observed_freq']:.4f}   {b['count']:>6d}"
            )

    for label, tcol, raw_list, iso_list in [
        ("P(hits>=1) / target_0.5", "target_0.5", prop_preds_05_raw, prop_preds_05_iso),
        ("P(hits>=2) / target_1.5", "target_1.5", prop_preds_15_raw, prop_preds_15_iso),
    ]:
        _print_calib(f"{label} [RAW]", raw_list, tcol)
        _print_calib(f"{label} [ISOTONIC]", iso_list, tcol)

    # Illustrative fair-line value at -110 (decimal 1.909, breakeven 0.524)
    breakeven = 1.0 / 1.909
    print(f"\n  Illustrative: at -110 (decimal 1.909) breakeven p = {breakeven:.3f}")
    for label, raw_list, iso_list in [
        ("0.5", prop_preds_05_raw, prop_preds_05_iso),
        ("1.5", prop_preds_15_raw, prop_preds_15_iso),
    ]:
        for tag, preds in [("RAW", raw_list), ("ISOTONIC", iso_list)]:
            val = [(p, a) for p, a in preds if p > breakeven]
            if val:
                realized = np.mean([a for _, a in val])
                print(
                    f"    target_{label} [{tag}]: model > breakeven in "
                    f"{len(val)}/{len(preds)} ({len(val) / len(preds) * 100:.1f}%); "
                    f"realized={realized * 100:.1f}% (need {breakeven * 100:.1f}%)"
                )
            else:
                print(f"    target_{label} [{tag}]: none above breakeven")

    print("\nDone.")


if __name__ == "__main__":
    main()
