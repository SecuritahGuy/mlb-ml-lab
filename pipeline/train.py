"""Multi-season, all-teams training pipeline.

Fetches game logs for every position player across 30 teams for
2021–2025, builds the full feature matrix, runs walk-forward
validation across season boundaries, and saves the final model.

Usage:
    poetry run python pipeline/train.py

First run takes ~30-45 minutes (all API calls cached to disk).
Subsequent runs are seconds.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

from sklearn.exceptions import ConvergenceWarning

from mlb_ml_lab import (
    MlbClient,
    PlayerGameLog,
    build_feature_matrix,
    describe_features,
    load_feature_data,
    make_targets,
)
from mlb_ml_lab.models.train import (
    MODEL_HELP,
    save_model,
    train_baselines,
    train_final,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
MODEL_DIR = "data/models/final"
PREDICT_SEASON = 2026
CACHED_DATASET = "data/datasets/full_2021_2026_30teams"


def main() -> None:
    use_cached = None
    for arg in sys.argv[1:]:
        if arg == "--use-cached" or arg.startswith("--use-cached="):
            parts = arg.split("=", 1)
            use_cached = parts[1] if len(parts) > 1 else CACHED_DATASET

    if use_cached:
        print(f"Loading cached dataset from {use_cached}...")
        feature_matrix, targets, meta = load_feature_data(use_cached)
        print(f"  {len(feature_matrix)} feature rows")
        print(f"  {len(targets)} target rows")
        print(f"  {meta.get('feature_count', '?')} feature columns")
        all_player_ids: set[int] = set()
        for row in feature_matrix:
            all_player_ids.add(row["player_id"])
        _run_training(feature_matrix, targets, all_player_ids)
        return

    client = MlbClient()

    all_team_ids = [t["id"] for t in client.get_teams()]
    print(f"Found {len(all_team_ids)} teams")

    all_game_logs: list[PlayerGameLog] = []
    all_player_ids: set[int] = set()

    for season in TRAIN_SEASONS:
        print(f"\n{'=' * 60}")
        print(f"Season {season}")
        print(f"{'=' * 60}")

        roster_players: list[dict[str, Any]] = []
        for tid in all_team_ids:
            roster = client.get_roster(tid, season, roster_type="40Man")
            for p in roster:
                pos = (p.get("position") or {}).get("abbreviation", "")
                if pos not in ("P", "", "Two-Way Player"):
                    roster_players.append(p)
                    all_player_ids.add(p["person"]["id"])

        n_players = len(roster_players)
        print(f"  {n_players} position players across {len(all_team_ids)} teams")

        logs_this_season = 0
        for i, p in enumerate(roster_players):
            pid = p["person"]["id"]
            try:
                raw = client.get_player_game_log(pid, season=season)
                for split in raw:
                    all_game_logs.append(PlayerGameLog.from_split_dict(split))
                    logs_this_season += 1
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            if (i + 1) % 100 == 0:
                print(
                    f"    ... {i + 1}/{n_players} players ({logs_this_season} game logs)"
                )

        print(f"  {logs_this_season} game log rows for season {season}")

    print(
        f"\nTotal: {len(all_game_logs)} game log rows, "
        f"{len(all_player_ids)} unique players"
    )

    print("\nFetching enriched schedules...")
    schedule_lookups: dict[int, dict[str, Any]] = {}
    for season in TRAIN_SEASONS:
        sched = client.get_enriched_schedule(season)
        schedule_lookups.update(sched)
        print(f"  {season}: {len(sched)} games hydrated")

    print("\nFetching opponent pitching stats (season-averaged)...")
    opp_ids = list({log.opponent_id for log in all_game_logs})
    opponent_pitching: dict[int, dict[str, Any]] = {}
    for season in TRAIN_SEASONS:
        try:
            stats = client.get_team_pitching_stats(opp_ids, season)
            opponent_pitching.update(stats)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(opponent_pitching)} team stat sets ({TRAIN_SEASONS[-1]} season)")

    print("Fetching monthly pitching splits...")
    monthly_pitching: dict[int, Any] = {}
    for season in TRAIN_SEASONS:
        try:
            mp = client.get_team_pitching_monthly_stats(opp_ids, season)
            monthly_pitching.update(mp)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(monthly_pitching)} monthly stat sets")

    print("Fetching team fielding...")
    team_fielding: dict[int, dict[str, Any]] = {}
    for season in TRAIN_SEASONS:
        try:
            tf = client.get_team_fielding_stats(opp_ids, season)
            team_fielding.update(tf)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(team_fielding)} fielding stat sets")

    print("Fetching league stats (last season)...")
    league_stats = {}
    try:
        all_hitting = client.get_team_hitting_stats(all_team_ids, TRAIN_SEASONS[-1])
        league_stats = _compute_league_stats(all_hitting)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    print(f"  {'populated' if league_stats else 'empty'}")

    print("Fetching player details...")
    player_details: dict[int, dict[str, Any]] = {}
    for i, pid in enumerate(all_player_ids):
        try:
            player_details[pid] = client.get_player(pid, season=TRAIN_SEASONS[-1])
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if (i + 1) % 200 == 0:
            print(f"    ... {i + 1}/{len(all_player_ids)} players")
    print(f"  {len(player_details)} player details")

    print("Fetching bullpen stats...")
    bullpen_stats: dict[int, dict[str, float]] = {}
    for tid in opp_ids:
        try:
            bp = client.get_team_bullpen_stats(tid, TRAIN_SEASONS[-1])
            if bp:
                bullpen_stats[tid] = bp
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(bullpen_stats)} bullpen stat sets")

    print("Fetching game pace...")
    game_pace_stats: dict[int, dict[str, float]] = {}
    for tid in opp_ids:
        try:
            pace_rows = client.get_game_pace(TRAIN_SEASONS[-1], team_id=tid)
            if pace_rows:
                p = pace_rows[0]
                game_pace_stats[tid] = {
                    "time_per_game": p.get("timePerGame"),
                    "pitches_per_game": p.get("pitchesPerGame"),
                }
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(game_pace_stats)} team pace records")

    print("Fetching team leaders...")
    team_leaders: dict[int, dict[str, float]] = {}
    for tid in opp_ids:
        try:
            leaders = client.get_team_leaders(
                tid,
                TRAIN_SEASONS[-1],
                leader_categories=["battingAverage", "homeRuns", "runsBattedIn"],
                limit=1,
            )
            ld: dict[str, float] = {}
            for entry in leaders:
                cat = entry.get("leaderCategory", "")
                leaders_list = entry.get("leaders", [])
                val = leaders_list[0].get("value") if leaders_list else None
                if val is not None:
                    if "battingAverage" in cat:
                        ld["top_avg"] = _parse_avg(val)
                    elif "homeRuns" in cat:
                        ld["top_hr"] = float(val)
                    elif "runsBattedIn" in cat:
                        ld["top_rbi"] = float(val)
            if ld:
                team_leaders[tid] = ld
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(team_leaders)} team leader sets")

    season_schedule = list(schedule_lookups.values())

    print("\nBuilding feature matrix...")
    feature_matrix = build_feature_matrix(
        all_game_logs,
        teams=client.get_teams(),
        extra_kwargs={
            "game_contexts": schedule_lookups,
            "opponent_pitching": opponent_pitching,
            "monthly_pitching": monthly_pitching,
            "team_fielding": team_fielding,
            "player_details": player_details,
            "league_stats": league_stats,
            "season_schedule": season_schedule,
            "bullpen_stats": bullpen_stats,
            "game_pace_stats": game_pace_stats,
            "team_leaders": team_leaders,
        },
    )
    print(f"  {len(feature_matrix)} feature rows")
    metas = describe_features()
    print(f"  {len(metas)} feature columns registered")

    feature_matrix.sort(key=lambda r: r["date"])

    print("Building targets...")
    targets = make_targets(all_game_logs)
    print(f"  {len(targets)} target rows")

    _run_training(feature_matrix, targets, all_player_ids, len(all_game_logs))


def _run_training(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    all_player_ids: set[int],
    n_game_logs: int = 0,
) -> None:
    """Walk-forward validation + final model training.

    Shared by both the live-fetch path and the ``--use-cached`` path.
    """
    print(f"\n{'=' * 60}")
    print("Walk-forward validation (season-boundary splits)")
    print(f"{'=' * 60}")

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n--- Target: {target_col} ---")
        result = train_baselines(
            feature_matrix,
            targets,
            target_col=target_col,
            n_splits=4,
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

    print(f"\n{'=' * 60}")
    print("Training final model on all available seasons")
    print(f"{'=' * 60}")

    # Best params from hyperparameter tuning (XGB, target_1.5)
    _TUNED_XGB = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
    }

    final_result = train_final(
        feature_matrix,
        targets,
        target_col="target_1.5",
        model_type="xgb",
        params=_TUNED_XGB,
    )
    if final_result["model"] is None:
        print("  ERROR: final training failed")
        return

    save_model(
        final_result["model"],
        final_result["feature_cols"],
        final_result["imputer"],
        MODEL_DIR,
        {
            "target_col": "target_1.5",
            "model_type": "lgb",
            "train_seasons": TRAIN_SEASONS,
            "n_players": len(all_player_ids),
            "n_game_logs": n_game_logs,
            **final_result["metadata"],
        },
    )
    print(f"  Model saved to {MODEL_DIR}")
    print(f"  Features: {final_result['metadata']['n_features']}")
    print(f"  Training rows: {final_result['metadata']['n_rows']}")

    # Also train final model for target_0.5
    final_05 = train_final(
        feature_matrix,
        targets,
        target_col="target_0.5",
        model_type="xgb",
        params=_TUNED_XGB,
    )
    if final_05["model"] is not None:
        save_model(
            final_05["model"],
            final_05["feature_cols"],
            final_05["imputer"],
            f"{MODEL_DIR}_0_5",
            {
                "target_col": "target_0.5",
                "model_type": "xgb",
                "train_seasons": TRAIN_SEASONS,
                "n_players": len(all_player_ids),
                **final_05["metadata"],
            },
        )
        print(f"  target_0.5 model saved to {MODEL_DIR}_0_5")

    print(f"\nDone. Models saved to {MODEL_DIR}*")
    print(f"Run `poetry run python pipeline/predict.py` to predict on {PREDICT_SEASON}")


def _parse_avg(val: str | float | None) -> float | None:
    if val is None or val == "" or val == ".---":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _compute_league_stats(
    hitting: dict[int, dict[str, Any]],
) -> dict[str, float]:
    total_ab = 0
    total_h = 0
    total_bb = 0
    total_1b = 0
    total_2b = 0
    total_3b = 0
    total_hr = 0
    total_r = 0
    total_g = 0
    for stat in hitting.values():
        ab = int(stat.get("atBats", 0))
        h = int(stat.get("hits", 0))
        bb = int(stat.get("baseOnBalls", 0))
        _2b = int(stat.get("doubles", 0))
        _3b = int(stat.get("triples", 0))
        hr = int(stat.get("homeRuns", 0))
        r = int(stat.get("runs", 0))
        g = int(stat.get("gamesPlayed", 0))
        total_ab += ab
        total_h += h
        total_bb += bb
        total_2b += _2b
        total_3b += _3b
        total_hr += hr
        total_r += r
        total_g += g
        total_1b += h - _2b - _3b - hr

    if total_ab == 0:
        return {}

    avg = round(total_h / total_ab, 3)
    obp = round((total_h + total_bb) / (total_ab + total_bb), 3)
    slg = round((total_1b + 2 * total_2b + 3 * total_3b + 4 * total_hr) / total_ab, 3)
    return {
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": round(obp + slg, 3),
        "runs_per_game": round(total_r / max(total_g, 1), 2),
    }


if __name__ == "__main__":
    main()
