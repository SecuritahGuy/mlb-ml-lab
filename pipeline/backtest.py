"""Walk-forward backtest across multiple seasons.

Fetches game logs and features for 2021–2025, runs walk-forward
prediction for every out-of-sample game, then simulates flat-stake
betting and reports ROI, drawdown, and calibration.

Usage:
    poetry run python pipeline/backtest.py

First run takes ~30-45 minutes (API calls cached to disk).
Subsequent runs are seconds.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

from sklearn.exceptions import ConvergenceWarning

from mlb_ml_lab import (
    MlbClient,
    PlayerGameLog,
    build_feature_matrix,
    describe_features,
    make_targets,
)
from mlb_ml_lab.evaluation.backtest import (
    BetResult,
    calibration_buckets,
    print_backtest_report,
    simulate_bets,
    walk_forward_predict,
)
from mlb_ml_lab.models.train import MODEL_HELP

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_ODDS = 1.909  # -110 US
OUTPUT_DIR = "data/backtest"


def main() -> None:
    client = MlbClient()

    all_team_ids = [t["id"] for t in client.get_teams()]
    print(f"Found {len(all_team_ids)} teams")

    all_game_logs: list[PlayerGameLog] = []
    all_player_ids: set[int] = set()

    for season in TRAIN_SEASONS:
        print(f"\n{'='*60}")
        print(f"Season {season}")
        print(f"{'='*60}")

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
                print(f"    ... {i+1}/{n_players} players ({logs_this_season} game logs)")

        print(f"  {logs_this_season} game log rows for season {season}")

    print(f"\nTotal: {len(all_game_logs)} game log rows, "
          f"{len(all_player_ids)} unique players")

    print("\nFetching enriched schedules...")
    schedule_lookups: dict[int, dict[str, Any]] = {}
    for season in TRAIN_SEASONS:
        sched = client.get_enriched_schedule(season)
        schedule_lookups.update(sched)
        print(f"  {season}: {len(sched)} games hydrated")

    print("\nFetching opponent pitching stats...")
    opp_ids = list({log.opponent_id for log in all_game_logs})
    season_opp_pitching: dict[int, dict[str, Any]] = {}
    for season in TRAIN_SEASONS:
        try:
            stats = client.get_team_pitching_stats(opp_ids, season)
            season_opp_pitching.update(stats)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(season_opp_pitching)} team stat sets")

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

    print("Fetching league stats...")
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
            print(f"    ... {i+1}/{len(all_player_ids)} players")
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
                tid, TRAIN_SEASONS[-1],
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
            "opponent_pitching": season_opp_pitching,
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

    models_to_test = ["lgb", "xgb", "rf", "lr", "ensemble"]
    targets_to_test = ["target_0.5", "target_1.5"]
    prob_thresholds = [None, 0.6, 0.7, 0.8]

    print(f"\n{'='*60}")
    print("WALK-FORWARD BACKTEST")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for target_col in targets_to_test:
        print(f"\n{'='*60}")
        print(f"  Target: {target_col}")
        print(f"{'='*60}")

        for model_type in models_to_test:
            print(f"\n  --- Model: {MODEL_HELP.get(model_type, model_type.upper())} ---")
            label = MODEL_HELP.get(model_type, model_type.upper())

            predictions = walk_forward_predict(
                feature_matrix,
                targets,
                target_col=target_col,
                model_type=model_type if model_type != "ensemble" else "lgb",
                n_splits=4,
            )
            if not predictions:
                print("    No predictions generated.")
                continue

            n_outcomes = len(predictions)
            base_rate = sum(p.actual for p in predictions) / n_outcomes
            print(f"    Out-of-sample predictions: {n_outcomes}")
            print(f"    Base rate:                 {base_rate:.4f}")

            thresholds_eval = prob_thresholds if model_type != "ensemble" else [None]
            for thresh in thresholds_eval:
                result = simulate_bets(
                    predictions,
                    decimal_odds=DEFAULT_ODDS,
                    min_prob=thresh,
                )
                result.target_col = target_col
                result.model_type = label
                result.n_seasons = len(TRAIN_SEASONS)
                cal = calibration_buckets(predictions)

                print_backtest_report(result, calibration=cal)

                safe_label = label.lower().replace(" ", "_")
                safe_thresh = f"p{int(thresh*100)}" if thresh else "breakeven"
                report_path = os.path.join(
                    OUTPUT_DIR,
                    f"{target_col}_{safe_label}_{safe_thresh}.json",
                )
                _save_report(result, cal, report_path)

    print(f"\nReports saved to {OUTPUT_DIR}/")
    print("Done.")


def _save_report(
    result: BetResult,
    calibration: list[dict[str, float]] | None,
    path: str,
) -> None:
    data = {
        "total_bets": result.total_bets,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "total_stake": result.total_stake,
        "total_profit": result.total_profit,
        "roi": result.roi,
        "max_drawdown": result.max_drawdown,
        "avg_prob": result.predicted_prob_mean,
        "odds": result.avg_odds,
        "threshold": result.threshold,
        "stake_per_bet": result.stake_per_bet,
        "target_col": result.target_col,
        "model_type": result.model_type,
        "n_seasons": result.n_seasons,
    }
    if calibration:
        data["calibration"] = calibration
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
