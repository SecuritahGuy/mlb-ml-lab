"""Predict on a new season using a previously saved model.

Usage:
    poetry run python pipeline/predict.py [--model-dir data/models/final] [--season 2026]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

from mlb_ml_lab import (
    MlbClient,
    PlayerGameLog,
    build_feature_matrix,
    save_feature_data,
)
from mlb_ml_lab.models.train import load_model

MODEL_DIR = "data/models/final"
PREDICT_SEASON = 2026


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else MODEL_DIR
    predict_season = PREDICT_SEASON

    for arg in sys.argv[1:]:
        if arg.startswith("--season="):
            predict_season = int(arg.split("=")[1])
        elif arg.startswith("--model-dir="):
            model_dir = arg.split("=")[1]

    print(f"Loading model from {model_dir}...")
    model, feature_cols, imputer, metadata = load_model(model_dir)
    target_col = metadata.get("target_col", "target_1.5")
    print(f"  Model type: {metadata.get('model_type', '?')}")
    print(f"  Target: {target_col}")
    print(f"  Features: {len(feature_cols)}")

    client = MlbClient()

    all_team_ids = [t["id"] for t in client.get_teams()]
    print(f"Found {len(all_team_ids)} teams")

    print(f"\nFetching {predict_season} rosters and game logs...")
    all_game_logs: list[PlayerGameLog] = []
    all_player_ids: set[int] = set()

    for tid in all_team_ids:
        roster = client.get_roster(tid, predict_season, roster_type="40Man")
        for p in roster:
            pos = (p.get("position") or {}).get("abbreviation", "")
            if pos not in ("P", "", "Two-Way Player"):
                all_player_ids.add(p["person"]["id"])

    all_player_ids_list = sorted(all_player_ids)
    print(f"  {len(all_player_ids_list)} position players")

    for i, pid in enumerate(all_player_ids_list):
        try:
            raw = client.get_player_game_log(pid, season=predict_season)
            for split in raw:
                all_game_logs.append(PlayerGameLog.from_split_dict(split))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{len(all_player_ids_list)} players "
                  f"({len(all_game_logs)} game logs)")

    print(f"  {len(all_game_logs)} game log rows")

    if not all_game_logs:
        print("No game logs found for {predict_season}.")
        return

    print("Fetching enriched schedule...")
    schedule_lookup = client.get_enriched_schedule(predict_season)
    print(f"  {len(schedule_lookup)} games hydrated")

    opp_ids = list({log.opponent_id for log in all_game_logs})
    print("Fetching opponent pitching stats...")
    opponent_pitching = {}
    try:
        stats = client.get_team_pitching_stats(opp_ids, predict_season)
        opponent_pitching.update(stats)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    print(f"  {len(opponent_pitching)} team stat sets")

    print("Fetching monthly pitching splits...")
    monthly_pitching = {}
    try:
        mp = client.get_team_pitching_monthly_stats(opp_ids, predict_season)
        monthly_pitching.update(mp)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    print("Fetching team fielding...")
    team_fielding = {}
    try:
        tf = client.get_team_fielding_stats(opp_ids, predict_season)
        team_fielding.update(tf)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    print("Fetching player details...")
    player_details: dict[int, dict[str, Any]] = {}
    for pid in all_player_ids_list:
        try:
            player_details[pid] = client.get_player(pid, season=predict_season)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    print(f"  {len(player_details)} player details")

    print("Fetching bullpen stats...")
    bullpen_stats = {}
    for tid in opp_ids:
        try:
            bp = client.get_team_bullpen_stats(tid, predict_season)
            if bp:
                bullpen_stats[tid] = bp
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    print("Fetching game pace...")
    game_pace_stats: dict[int, dict[str, float]] = {}
    for tid in opp_ids:
        try:
            pace_rows = client.get_game_pace(predict_season, team_id=tid)
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
                tid, predict_season,
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

    print("Fetching league stats...")
    league_stats = {}
    try:
        all_hitting = client.get_team_hitting_stats(all_team_ids, predict_season)
        league_stats = _compute_league_stats(all_hitting)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    season_schedule = list(schedule_lookup.values())

    print("\nBuilding feature matrix...")
    feature_matrix = build_feature_matrix(
        all_game_logs,
        teams=client.get_teams(),
        extra_kwargs={
            "game_contexts": schedule_lookup,
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

    if not feature_matrix:
        print("Empty feature matrix — nothing to predict.")
        return

    print("\nPredicting...")
    missing_cols = [c for c in feature_cols if c not in feature_matrix[0]]
    if missing_cols:
        print(f"  Warning: {len(missing_cols)} feature columns missing in "
              f"prediction data (will be imputed as 0)")

    x = np.array(
        [
            [row.get(c, 0.0) or 0.0 for c in feature_cols]
            for row in feature_matrix
        ],
        dtype=np.float64,
    )
    x = imputer.transform(x)
    x = np.nan_to_num(x, nan=0.0)

    y_proba = model.predict_proba(x)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    output_dir = f"data/predictions/{predict_season}"
    os.makedirs(output_dir, exist_ok=True)

    proba_key = f"prob_{target_col}"
    pred_key = f"pred_{target_col}"

    predictions: list[dict[str, Any]] = []
    for row, proba, pred in zip(feature_matrix, y_proba.tolist(), y_pred.tolist()):
        predictions.append({
            "player_id": row["player_id"],
            "game_pk": row["game_pk"],
            "date": row["date"],
            proba_key: round(proba, 4),
            pred_key: pred,
        })

    out_path = os.path.join(output_dir, "predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for pred_row in predictions:
            f.write(json.dumps(pred_row) + "\n")

    pos_rate = sum(p[pred_key] for p in predictions)
    print(f"\n  {len(predictions)} predictions written to {out_path}")
    print(f"  Positive prediction rate: {pos_rate}/{len(predictions)} "
          f"({pos_rate/len(predictions)*100:.1f}%)")

    # Also save as feature data for reuse
    save_feature_data(
        feature_matrix,
        predictions,
        f"data/predictions/{predict_season}_features",
        {"season": predict_season, "model_dir": model_dir},
    )
    print(f"  Feature data also cached to data/predictions/{predict_season}_features")
    print("\nDone.")


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
