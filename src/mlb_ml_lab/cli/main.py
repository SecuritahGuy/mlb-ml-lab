"""CLI entry point for mlb-ml-lab.

Usage:
    mlb --help
    mlb fetch --help
    mlb train --help
    mlb predict --help
    mlb backtest --help
    mlb tune --help
    mlb e2e --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from mlb_ml_lab import (
    MlbClient,
    PlayerGameLog,
    build_feature_matrix,
    describe_features,
    load_feature_data,
    make_targets,
    save_feature_data,
)
from mlb_ml_lab.evaluation.backtest import (
    calibrate_predictions_crossfit,
    expected_calibration_error,
    simulate_bets,
    walk_forward_predict,
)
from mlb_ml_lab.models.train import (
    DEFAULT_PARAM_GRIDS,
    MODEL_HELP,
    load_model,
    save_model,
    train_baselines,
    train_final,
    tune_hyperparameters,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

CACHED_DATASET_DEFAULT = "data/datasets/full_2021_2026_30teams"
MODEL_DIR_DEFAULT = "data/models/final"
ENSEMBLE_DIR_DEFAULT = "data/models/ensemble_0_5"
PREDICT_SEASON_DEFAULT = 2026
TRAIN_SEASONS_DEFAULT = [2021, 2022, 2023, 2024, 2025]


def cmd_bet(args: argparse.Namespace) -> None:
    """Delegate to the betting strategy script."""
    cmd = [sys.executable, "experiments/betting_strategy.py"]
    if args.date:
        cmd.extend(["--date", args.date])
    if args.settle:
        cmd.append("--settle")
    if args.pnl:
        cmd.append("--pnl")
    cmd.extend(["--threshold", str(args.threshold)])
    cmd.extend(["--stake", str(args.stake)])
    cmd.extend(["--model-dir", args.model_dir])
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch data and build feature matrix for one or more seasons."""
    client = MlbClient()
    all_teams = client.get_teams()
    team_ids = [t["id"] for t in all_teams if t.get("sport", {}).get("id") == 1]
    rosters: dict[tuple[int, int], list[dict[str, Any]]] = {}

    print(f"Fetching rosters for {len(team_ids)} teams, seasons {args.seasons}...")
    for tid in team_ids:
        for s in args.seasons:
            roster = client.get_roster(tid, season=s, roster_type="40Man")
            players = [
                p
                for p in roster
                if (p.get("position") or {}).get("abbreviation", "") not in ("P",)
            ][:args.max_players]
            rosters[(tid, s)] = players

    all_player_ids: set[int] = set()
    for players in rosters.values():
        all_player_ids.update(p["person"]["id"] for p in players)

    print(f"Fetching game logs for {len(all_player_ids)} players...")
    all_game_logs: list[PlayerGameLog] = []
    for s in args.seasons:
        for pid in all_player_ids:
            try:
                raw = client.get_player_game_log(pid, season=s)
                for split in raw:
                    all_game_logs.append(PlayerGameLog.from_split_dict(split))
            except Exception:
                pass

    print(f"  {len(all_game_logs)} game log rows")

    print("Fetching game contexts...")
    schedule_lookups: dict[int, dict[str, Any]] = {}
    for s in args.seasons:
        sched = client.get_enriched_schedule(s)
        schedule_lookups.update(sched)

    print("Fetching team stats...")
    opp_ids = list({log.opponent_id for log in all_game_logs})
    opponent_pitching = client.get_team_pitching_stats(opp_ids, args.seasons[-1]) or {}
    monthly_pitching = (
        client.get_team_pitching_monthly_stats(opp_ids, args.seasons[-1]) or {}
    )
    team_fielding = client.get_team_fielding_stats(opp_ids, args.seasons[-1]) or {}

    print("Building feature matrix...")
    season_schedule = list(schedule_lookups.values())
    feature_matrix = build_feature_matrix(
        all_game_logs,
        teams=all_teams,
        extra_kwargs={
            "game_contexts": schedule_lookups,
            "opponent_pitching": opponent_pitching,
            "monthly_pitching": monthly_pitching,
            "team_fielding": team_fielding,
            "season_schedule": season_schedule,
        },
    )
    print(f"  {len(feature_matrix)} feature rows")

    metas = describe_features()
    print(f"  {len(metas)} feature columns")

    print("Building targets...")
    targets = make_targets(all_game_logs)
    print(f"  {len(targets)} target rows")

    output_dir = args.output or f"data/datasets/fetch_{args.seasons[0]}_{args.seasons[-1]}"
    save_feature_data(
        feature_matrix,
        targets,
        output_dir,
        {"seasons": args.seasons, "team_count": len(team_ids)},
    )
    print(f"Saved to {output_dir}/")


def cmd_train(args: argparse.Namespace) -> None:
    """Train models on cached or live data via walk-forward validation."""
    if args.use_cached:
        print(f"Loading cached dataset from {args.use_cached}...")
        feature_matrix, targets, meta = load_feature_data(args.use_cached)
        print(f"  {len(feature_matrix)} rows, {len(targets)} targets")
    else:
        print("Fetching live data...")
        cmd_fetch(args)

        print("Loading freshly saved data...")
        dataset_dir = (
            args.output
            or f"data/datasets/fetch_{args.seasons[0]}_{args.seasons[-1]}"
        )
        feature_matrix, targets, meta = load_feature_data(dataset_dir)

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n--- Target: {target_col} ---")
        result = train_baselines(
            feature_matrix,
            targets,
            target_col=target_col,
            n_splits=args.folds,
        )
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        for model_type, mdata in result["models"].items():
            label = MODEL_HELP.get(model_type, model_type.upper())
            print(f"\n  {label}")
            print(f"    Avg accuracy: {mdata['avg_accuracy']:.4f}")
            print(f"    Avg AUC:      {mdata['avg_auc']:.4f}")
            print(f"    Folds:        {mdata['n_folds']}")

    print("\nTraining final model (target_1.5, XGBoost)...")
    final = train_final(
        feature_matrix,
        targets,
        target_col="target_1.5",
        model_type="xgb",
        params={
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
        },
    )
    if final["model"] is None:
        print("  ERROR: final training failed")
        return

    model_dir = args.model_dir
    save_model(
        final["model"],
        final["feature_cols"],
        final["imputer"],
        model_dir,
        {
            "target_col": "target_1.5",
            "model_type": "xgb",
            **final["metadata"],
        },
    )
    print(f"  Model saved to {model_dir}")

    if args.save_05:
        final_05 = train_final(
            feature_matrix,
            targets,
            target_col="target_0.5",
            model_type="xgb",
            params={
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 1.0,
                "min_child_weight": 1,
            },
        )
        if final_05["model"] is not None:
            save_model(
                final_05["model"],
                final_05["feature_cols"],
                final_05["imputer"],
                f"{model_dir}_0_5",
                {
                    "target_col": "target_0.5",
                    "model_type": "xgb",
                    **final_05["metadata"],
                },
            )
            print(f"  target_0.5 model saved to {model_dir}_0_5")


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict on a season using a saved model."""
    print(f"Loading model from {args.model_dir}...")
    model, feature_cols, imputer, metadata = load_model(args.model_dir)
    target_col = metadata.get("target_col", "target_1.5")
    print(f"  Model type: {metadata.get('model_type', '?')}")
    print(f"  Features: {len(feature_cols)}")

    client = MlbClient()
    all_teams = client.get_teams()
    all_team_ids = [t["id"] for t in all_teams if t.get("sport", {}).get("id") == 1]

    print(f"Fetching {args.season} data...")
    all_player_ids: set[int] = set()
    for tid in all_team_ids:
        roster = client.get_roster(tid, args.season, roster_type="40Man")
        for p in roster:
            pos = (p.get("position") or {}).get("abbreviation", "")
            if pos not in ("P", "", "Two-Way Player"):
                all_player_ids.add(p["person"]["id"])

    print(f"  {len(all_player_ids)} position players")

    all_game_logs: list[PlayerGameLog] = []
    for pid in sorted(all_player_ids):
        try:
            raw = client.get_player_game_log(pid, season=args.season)
            for split in raw:
                all_game_logs.append(PlayerGameLog.from_split_dict(split))
        except Exception:
            pass
    print(f"  {len(all_game_logs)} game log rows")

    schedule_lookup = client.get_enriched_schedule(args.season)
    opp_ids = list({log.opponent_id for log in all_game_logs})
    opponent_pitching = client.get_team_pitching_stats(opp_ids, args.season) or {}
    monthly_pitching = (
        client.get_team_pitching_monthly_stats(opp_ids, args.season) or {}
    )
    team_fielding = client.get_team_fielding_stats(opp_ids, args.season) or {}

    print("Building feature matrix...")
    feature_matrix = build_feature_matrix(
        all_game_logs,
        teams=all_teams,
        extra_kwargs={
            "game_contexts": schedule_lookup,
            "opponent_pitching": opponent_pitching,
            "monthly_pitching": monthly_pitching,
            "team_fielding": team_fielding,
            "season_schedule": list(schedule_lookup.values()),
        },
    )
    print(f"  {len(feature_matrix)} rows")

    if not feature_matrix:
        print("Empty feature matrix.")
        return

    x = np.array(
        [[row.get(c, 0.0) or 0.0 for c in feature_cols] for row in feature_matrix],
        dtype=np.float64,
    )
    x = imputer.transform(x)
    x = np.nan_to_num(x, nan=0.0)

    y_proba = model.predict_proba(x)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    import json
    import os

    output_dir = args.output or f"data/predictions/{args.season}"
    os.makedirs(output_dir, exist_ok=True)
    proba_key = f"prob_{target_col}"
    pred_key = f"pred_{target_col}"
    predictions = []
    for row, proba, pred in zip(feature_matrix, y_proba.tolist(), y_pred.tolist()):
        predictions.append(
            {
                "player_id": row["player_id"],
                "game_pk": row["game_pk"],
                "date": row["date"],
                proba_key: round(proba, 4),
                pred_key: int(pred),
            }
        )

    out_path = os.path.join(output_dir, "predictions.jsonl")
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    pos_rate = sum(p[pred_key] for p in predictions)
    print(f"  {len(predictions)} predictions -> {out_path}")
    print(f"  Positive rate: {pos_rate}/{len(predictions)} ({pos_rate / len(predictions) * 100:.1f}%)")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Walk-forward backtest with flat-stake betting simulation."""
    print(f"Loading data from {args.dataset}...")
    feature_matrix, targets_list, _meta = load_feature_data(args.dataset)

    # Support ensemble: comma-separated list of model types
    model_types = [m.strip() for m in args.model.split(",")]

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n{'=' * 60}")
        print(f"Walk-forward — {target_col}")
        print(f"{'=' * 60}")

        predictions = walk_forward_predict(
            feature_matrix,
            targets_list,
            target_col=target_col,
            model_type=model_types if len(model_types) > 1 else model_types[0],
            n_splits=args.folds,
        )
        if not predictions:
            print("  No predictions generated.")
            continue

        try:
            from sklearn.metrics import roc_auc_score
            y_true = [p.actual for p in predictions]
            y_prob = [p.predicted_prob for p in predictions]
            auc = roc_auc_score(y_true, y_prob)
            ece = expected_calibration_error(predictions, n_bins=10)
            print(f"  Overall AUC: {auc:.4f}  ECE: {ece:.4f}  ({len(predictions)} predictions)")
        except Exception:
            pass

        if args.calibrate:
            print("  Applying per-season isotonic calibration...")
            cal_preds = calibrate_predictions_crossfit(predictions, n_splits=5, seed=42)
            try:
                y_prob_cal = [p.predicted_prob for p in cal_preds]
                auc_cal = roc_auc_score([p.actual for p in cal_preds], y_prob_cal)
                ece_cal = expected_calibration_error(cal_preds, n_bins=10)
                print(f"  Calibrated AUC: {auc_cal:.4f}  ECE: {ece_cal:.4f}")
            except Exception:
                pass
            result_preds = cal_preds
        else:
            result_preds = predictions

        bet_results = simulate_bets(
            result_preds,
            odds=args.odds,
            thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        )
        label = "Calibrated" if args.calibrate else "Raw"
        print(f"\n  {label} — {'Thresh':>6}  {'Bets':>6}  {'WinRate':>8}  {'ROI':>8}  {'MaxDD':>8}")
        print(f"  {'-' * 6}  {'-' * 10}  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        for br in bet_results:
            if br.total_bets == 0:
                continue
            print(
                f"  {label:>10}  {br.threshold:>6.2f}  {br.total_bets:>6}  "
                f"{br.win_rate:>8.4f}  {br.roi:>8.2f}%  "
                f"{br.max_drawdown:>8.2f}%"
            )


def cmd_tune(args: argparse.Namespace) -> None:
    """Hyperparameter tuning via random search inside walk-forward."""
    print(f"Loading cached dataset from {args.dataset}...")
    feature_matrix, targets, meta = load_feature_data(args.dataset)
    print(f"  {len(feature_matrix)} rows, {len(targets)} targets")

    for model_type in args.models:
        for target_col in ("target_0.5", "target_1.5"):
            print(f"\n{'=' * 60}")
            print(f"Tuning {model_type.upper()} — {target_col}")
            print(f"{'=' * 60}")

            result = tune_hyperparameters(
                feature_matrix,
                targets,
                target_col=target_col,
                model_type=model_type,
                param_grid=DEFAULT_PARAM_GRIDS.get(model_type),
                n_trials=args.trials,
                n_splits=args.folds,
                metric=args.metric,
                seed=args.seed,
            )
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue
            print(f"  Best params: {result['best_params']}")
            print(f"  Best {args.metric.upper()}: {result['best_score']:.4f}")


def cmd_e2e(args: argparse.Namespace) -> None:
    """End-to-end: fetch single team/season, featurize, train, print metrics."""
    client = MlbClient()
    roster = client.get_roster(args.team_id, season=args.season)
    players = [
        p
        for p in roster
        if (p.get("position") or {}).get("abbreviation", "") not in ("P",)
    ][:args.max_players]

    print(f"Found {len(players)} position players for team {args.team_id} ({args.season})")
    player_ids = [p["person"]["id"] for p in players]

    all_game_logs: list[PlayerGameLog] = []
    for pid in player_ids:
        raw = client.get_player_game_log(pid, season=args.season)
        for split in raw[:args.games]:
            all_game_logs.append(PlayerGameLog.from_split_dict(split))
    print(f"  {len(all_game_logs)} game log rows")

    seen_pks = set()
    schedule_lookup: dict[int, dict[str, Any]] = {}
    for log in all_game_logs:
        if log.game_pk not in seen_pks:
            seen_pks.add(log.game_pk)
            try:
                ctx = client.get_game_context(log.game_pk)
                schedule_lookup[log.game_pk] = ctx
            except Exception:
                schedule_lookup[log.game_pk] = {}

    opp_ids = list({log.opponent_id for log in all_game_logs})
    opponent_pitching = client.get_team_pitching_stats(opp_ids, args.season) or {}
    monthly_pitching = (
        client.get_team_pitching_monthly_stats(opp_ids, args.season) or {}
    )
    team_fielding = client.get_team_fielding_stats(opp_ids, args.season) or {}

    feature_matrix = build_feature_matrix(
        all_game_logs,
        teams=client.get_teams(),
        extra_kwargs={
            "game_contexts": schedule_lookup,
            "opponent_pitching": opponent_pitching,
            "monthly_pitching": monthly_pitching,
            "team_fielding": team_fielding,
            "season_schedule": list(schedule_lookup.values()),
        },
    )
    print(f"  {len(feature_matrix)} feature rows")

    targets = make_targets(all_game_logs)
    print(f"  {len(targets)} target rows")

    cache_dir = f"data/features/{args.team_id}_{args.season}"
    save_feature_data(feature_matrix, targets, cache_dir, {"season": args.season})
    print(f"  Cached to {cache_dir}")

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n  --- Target: {target_col} ---")
        result = train_baselines(
            feature_matrix, targets, target_col=target_col, n_splits=args.folds
        )
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        for model_type, mdata in result["models"].items():
            label = MODEL_HELP.get(model_type, model_type.upper())
            print(f"    {label}")
            print(f"      Avg accuracy: {mdata['avg_accuracy']:.4f}")
            print(f"      Avg AUC:      {mdata['avg_auc']:.4f}")

    print("\nDone.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlb",
        description="MLB prediction models — fetch, train, predict, backtest, tune.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch data and build feature matrix")
    p_fetch.add_argument("--seasons", nargs="+", type=int, default=TRAIN_SEASONS_DEFAULT)
    p_fetch.add_argument("--max-players", type=int, default=20)
    p_fetch.add_argument("-o", "--output", type=str, default=None)
    p_fetch.set_defaults(func=cmd_fetch)

    # train
    p_train = sub.add_parser("train", help="Train models via walk-forward validation")
    p_train.add_argument("--use-cached", type=str, nargs="?", const=CACHED_DATASET_DEFAULT, default=None)
    p_train.add_argument("--seasons", nargs="+", type=int, default=TRAIN_SEASONS_DEFAULT)
    p_train.add_argument("--max-players", type=int, default=20)
    p_train.add_argument("--folds", type=int, default=4)
    p_train.add_argument("-o", "--output", type=str, default=None)
    p_train.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    p_train.add_argument("--save-05", action="store_true", help="Also train and save target_0.5 model")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = sub.add_parser("predict", help="Predict on a season using a saved model")
    p_pred.add_argument("--season", type=int, default=PREDICT_SEASON_DEFAULT)
    p_pred.add_argument("--model-dir", type=str, default=MODEL_DIR_DEFAULT)
    p_pred.add_argument("-o", "--output", type=str, default=None)
    p_pred.set_defaults(func=cmd_predict)

    # backtest
    p_bt = sub.add_parser("backtest", help="Walk-forward backtest with betting simulation")
    p_bt.add_argument("--dataset", type=str, default=CACHED_DATASET_DEFAULT)
    p_bt.add_argument("--model", type=str, default="lgb",
                       help="Model type(s). Ensemble: comma-sep (e.g. 'lr,xgb,rf,lgb')")
    p_bt.add_argument("--folds", type=int, default=5)
    p_bt.add_argument("--odds", type=int, default=-110)
    p_bt.add_argument("--calibrate", action="store_true",
                       help="Apply per-season isotonic calibration")
    p_bt.set_defaults(func=cmd_backtest)

    # tune
    p_tune = sub.add_parser("tune", help="Hyperparameter tuning via random search")
    p_tune.add_argument("--dataset", type=str, default=CACHED_DATASET_DEFAULT)
    p_tune.add_argument("--models", nargs="+", default=["lgb", "xgb"], choices=["lr", "xgb", "rf", "lgb", "mlx"])
    p_tune.add_argument("--trials", type=int, default=12)
    p_tune.add_argument("--folds", type=int, default=4)
    p_tune.add_argument("--metric", type=str, default="auc", choices=["auc", "log_loss"])
    p_tune.add_argument("--seed", type=int, default=42)
    p_tune.set_defaults(func=cmd_tune)

    # bet
    p_bet = sub.add_parser("bet", help="Generate daily betting recommendations and track P&L")
    p_bet.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    p_bet.add_argument("--threshold", type=float, default=0.55)
    p_bet.add_argument("--stake", type=float, default=1.0)
    p_bet.add_argument("--model-dir", type=str, default=ENSEMBLE_DIR_DEFAULT)
    p_bet.add_argument("--settle", action="store_true", help="Settle unsettled bets")
    p_bet.add_argument("--pnl", action="store_true", help="Show P&L summary")
    p_bet.set_defaults(func=cmd_bet)

    # e2e
    p_e2e = sub.add_parser("e2e", help="End-to-end: fetch, featurize, train for one team/season")
    p_e2e.add_argument("--team-id", type=int, default=108, help="Los Angeles Angels")
    p_e2e.add_argument("--season", type=int, default=2024)
    p_e2e.add_argument("--max-players", type=int, default=5)
    p_e2e.add_argument("--games", type=int, default=60)
    p_e2e.add_argument("--folds", type=int, default=3)
    p_e2e.set_defaults(func=cmd_e2e)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
