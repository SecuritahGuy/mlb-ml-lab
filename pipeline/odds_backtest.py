"""Fetch historical SBR odds, match to our model predictions, find edges.

Walk-forward ensemble predictions → aggregate to team-game level →
compare to market moneyline odds from SBR → identify profitable
disagreements between model and market.

Usage:
    poetry run python pipeline/odds_backtest.py
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer

from mlb_ml_lab import MlbClient, PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.data.odds import fetch_game_odds, load_cached_odds, save_cached_odds
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


def ml_to_implied_prob(ml: int | None) -> float | None:
    """Convert American moneyline odds to implied probability."""
    if ml is None or abs(ml) >= 10000:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 100.0 / (ml + 100.0)


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


def fetch_all_historical_odds(
    dates: list[str],
    sportsbook: str = "betmgm",
) -> dict[str, list[dict[str, Any]]]:
    """Fetch SBR odds for all dates, using disk cache."""
    result: dict[str, list[dict[str, Any]]] = {}
    cached = 0
    fetched = 0

    for ds in dates:
        odds = load_cached_odds(ds)
        if odds is not None:
            result[ds] = odds
            cached += 1
        else:
            odds = fetch_game_odds(ds, sportsbook=sportsbook)
            result[ds] = odds
            save_cached_odds(ds, odds)
            fetched += 1
            time.sleep(0.5)  # be polite

        if (cached + fetched) % 100 == 0:
            print(f"    Odds: {cached} cached + {fetched} fetched ({cached + fetched}/{len(dates)})")

    print(f"  Odds complete: {cached} cached + {fetched} fetched = {len(result)} dates")
    return result


def main() -> None:
    # ── Load data ────────────────────────────────────────────────
    print("Loading data...")
    raw_logs = load_game_logs(CACHED_DATASET)
    feature_matrix, targets_list, meta = load_feature_data(CACHED_DATASET)
    print(f"  {len(raw_logs)} game logs, {len(feature_matrix)} feature rows")

    # Build team ID → abbreviation mapping
    client = MlbClient()
    teams = client.get_teams()
    id_to_abbrev: dict[int, str] = {}
    for t in teams:
        if t.get("sport", {}).get("id") == 1:
            id_to_abbrev[t["id"]] = t.get("abbreviation", "")

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

    # All unique dates in test seasons
    test_dates = sorted(set(
        d["date"][:10] for d in targets
        if d["date"][:4] in ("2022", "2023", "2024", "2025")
    ))
    print(f"  {len(test_dates)} unique dates in test seasons (2022-2025)")

    # ── Fetch historical odds ────────────────────────────────────
    print("\nFetching historical SBR odds (cached after first run)...")
    date_odds = fetch_all_historical_odds(test_dates, sportsbook="betmgm")

    # Build odds index: (date_str, away_abbrev, home_abbrev) → odds dict
    odds_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ds, odds_list in date_odds.items():
        for g in odds_list:
            odds_index[(ds, g["away_team"], g["home_team"])] = g

    print(f"  {len(odds_index)} game-odds entries indexed")

    # ── Walk-forward ensemble + compare to market ────────────────
    print(f"\n{'=' * 60}")
    print("Walk-forward: model predictions vs market odds")
    print(f"{'=' * 60}")

    target_col = "target_0.5"
    team_game_results: list[dict] = []

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

        # Build ensemble predictions for each test game
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

        # Aggregate to team-game level
        team_games: dict[tuple[int, int], dict] = {}
        for i, (pid, gpk) in enumerate(_keys_te):
            hp = float(hybrid_te_proba[i])
            xp = xgb_proba_map.get((pid, gpk), hp)
            ensemble_prob = (hp + xp) / 2.0

            for lg in test_logs:
                if lg.player_id == pid and lg.game_pk == gpk:
                    key = (lg.team_id, gpk)
                    if key not in team_games:
                        team_games[key] = {
                            "team_id": lg.team_id,
                            "game_pk": gpk,
                            "date": lg.date[:10],
                            "opponent_id": lg.opponent_id,
                            "is_home": lg.is_home,
                            "sum_probs": 0.0,
                            "n_players": 0,
                            "actual_team_hits": 0,
                        }
                    team_games[key]["sum_probs"] += ensemble_prob
                    team_games[key]["n_players"] += 1
                    team_games[key]["actual_team_hits"] += lg.hits
                    break

        # Match to market odds
        fold_matched = 0
        for key, tg in team_games.items():
            team_abbrev = id_to_abbrev.get(tg["team_id"], "")
            opp_abbrev = id_to_abbrev.get(tg["opponent_id"], "")
            ds = tg["date"]

            # Our team could be home or away in SBR's listing
            odds_row = odds_index.get((ds, opp_abbrev, team_abbrev))
            if odds_row is None:
                odds_row = odds_index.get((ds, team_abbrev, opp_abbrev))

            if odds_row is None:
                continue

            fold_matched += 1
            tg["team_abbrev"] = team_abbrev
            tg["opp_abbrev"] = opp_abbrev

            # Determine if our team is the home or away team in SBR odds
            if odds_row["home_team"] == team_abbrev:
                tg["team_ml"] = odds_row["home_ml"]
                tg["opp_ml"] = odds_row["away_ml"]
            else:
                tg["team_ml"] = odds_row["away_ml"]
                tg["opp_ml"] = odds_row["home_ml"]

            tg["team_implied_prob"] = ml_to_implied_prob(tg["team_ml"])
            tg["avg_hit_prob"] = tg["sum_probs"] / max(tg["n_players"], 1)
            team_game_results.append(tg)

        n_team = len(team_games)
        print(f"    {n_team} team-games, {fold_matched} matched to market odds")

    # ── Analysis ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Analysis: model hit expectations vs market moneyline")
    print(f"{'=' * 60}")

    if not team_game_results:
        print("  No matched results!")
        return

    results = team_game_results
    print(f"\n  {len(results)} team-game rows with both model predictions and market odds")

    # Build game index to pair home/away
    game_pairs: dict[int, dict] = {}
    for r in results:
        gpk = r["game_pk"]
        if gpk not in game_pairs:
            game_pairs[gpk] = {}
        game_pairs[gpk][r["team_id"]] = r

    # Augment results with opponent data
    for r in results:
        gpk = r["game_pk"]
        pair = game_pairs.get(gpk, {})
        opp_id = r["opponent_id"]
        opp = pair.get(opp_id, {})
        r["opp_sum_probs"] = opp.get("sum_probs", 0)
        r["opp_actual_hits"] = opp.get("actual_team_hits", 0)
        r["opp_avg_hit_prob"] = opp.get("avg_hit_prob", 0)
        r["opp_n_players"] = opp.get("n_players", 0)

    # Does our model agree with the market?
    favorites_match = 0
    for r in results:
        model_fav = 1 if r["sum_probs"] > r["opp_sum_probs"] else 0
        market_fav = 1 if (r["team_ml"] is not None and r["opp_ml"] is not None
                          and r["team_ml"] < r["opp_ml"]) else 0
        if model_fav == market_fav:
            favorites_match += 1

    n_with_both = sum(1 for r in results if r["team_ml"] is not None and r.get("opp_ml") is not None)
    if n_with_both > 0:
        pct = favorites_match / n_with_both * 100
        print(f"  Model & market agree on favorite: {favorites_match}/{n_with_both} ({pct:.1f}%)")

    # Correlation: avg_hit_prob vs market implied prob
    valid = [(r["avg_hit_prob"], r["team_implied_prob"])
             for r in results if r["team_implied_prob"] is not None]
    if len(valid) > 10:
        hit_probs = np.array([v[0] for v in valid])
        market_probs = np.array([v[1] for v in valid])
        corr = np.corrcoef(hit_probs, market_probs)[0, 1]
        print(f"  Correlation (avg hit prob × market win prob): {corr:.4f}")

    # Edge analysis: find games where our model expects more hits than market expects wins
    print("\n  --- Games where avg_hit_prob >> market_implied_prob ---")
    edges = []
    for r in results:
        if r["team_implied_prob"] is None or r["n_players"] < 5:
            continue
        edge = r["avg_hit_prob"] - r["team_implied_prob"]
        r["edge"] = edge
        edges.append(r)

    edges.sort(key=lambda x: -x["edge"])
    print(f"  {len(edges)} games with both probabilities")
    print(f"  {'Team':>6}  {'Opp':>6}  {'Date':>12}  {'AvgHit%':>8}  {'MktWin%':>8}  {'Edge':>8}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in edges[:20]:
        print(f"  {r['team_abbrev']:>6}  {r['opp_abbrev']:>6}  {r['date']:>12}  "
              f"{r['avg_hit_prob']:>8.3f}  {r['team_implied_prob']:>8.3f}  "
              f"{r['edge']:>+8.3f}")

    # ── Betting simulation: bet on model-favored team when it disagrees with market ──
    print("\n  --- Betting when model expects hits > market expects wins ---")
    print(f"  {'MinEdge':>8}  {'Bets':>6}  {'WinRate':>8}  {'P&L':>8}  {'ROI':>8}")

    for min_edge in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        bets = [r for r in edges if r["edge"] >= min_edge]
        if not bets:
            continue
        wins = sum(1 for r in bets if r["actual_team_hits"] > r["opp_actual_hits"])
        n = len(bets)
        win_rate = wins / n
        net_pnl = wins * (DECIMAL_ODDS - 1) - (n - wins)
        roi = net_pnl / n * 100
        print(f"  {min_edge:>8.2f}  {n:>6}  {win_rate:>8.4f}  {net_pnl:>8.2f}  {roi:>8.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
