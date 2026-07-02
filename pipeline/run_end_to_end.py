"""End-to-end pipeline: fetch MLB data → featurize → train → evaluate.

Usage:
    poetry run python pipeline/run_end_to_end.py

Configurable constants at the top of the file (``TEAM_ID``, ``SEASON``,
``MAX_PLAYERS``, etc.).
"""

from __future__ import annotations

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
from mlb_ml_lab.models.train import MODEL_HELP, train_baselines

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Configuration ─────────────────────────────────────────────────────────────
TEAM_ID = 108  # Los Angeles Angels
SEASON = 2024
MAX_PLAYERS = 5
SAMPLE_GAMES = 60


def fetch_roster_players(client: MlbClient) -> list[dict[str, Any]]:
    roster = client.get_roster(TEAM_ID, season=SEASON)
    players: list[dict[str, Any]] = []
    for p in roster:
        pos = (p.get("position") or {}).get("abbreviation", "")
        if pos not in ("P",):
            players.append(p)
        if len(players) >= MAX_PLAYERS:
            break
    return players


def fetch_game_logs(
    client: MlbClient, player_ids: list[int]
) -> list[PlayerGameLog]:
    all_logs: list[PlayerGameLog] = []
    for pid in player_ids:
        raw = client.get_player_game_log(pid, season=SEASON)
        for split in raw[:SAMPLE_GAMES]:
            all_logs.append(PlayerGameLog.from_split_dict(split))
    return all_logs


def build_game_contexts(
    client: MlbClient, game_logs: list[Any]
) -> dict[int, dict[str, Any]]:
    seen = set()
    game_pks: list[int] = []
    for log in game_logs:
        pk = log.game_pk
        if pk not in seen:
            seen.add(pk)
            game_pks.append(pk)

    contexts: dict[int, dict[str, Any]] = {}
    for pk in game_pks:
        try:
            ctx = client.get_game_context(pk)
            contexts[pk] = ctx
        except Exception:  # pylint: disable=broad-exception-caught
            contexts[pk] = {}
    return contexts


def fetch_opponent_pitching(
    client: MlbClient, game_logs: list[Any]
) -> dict[int, dict[str, float]]:
    opp_ids = list({log.opponent_id for log in game_logs})
    try:
        pitching = client.get_team_pitching_stats(opp_ids, SEASON)
    except Exception:  # pylint: disable=broad-exception-caught
        pitching = {}
    return pitching


def fetch_player_details(
    client: MlbClient, player_ids: list[int]
) -> dict[int, dict[str, Any]]:
    details: dict[int, dict[str, Any]] = {}
    for pid in player_ids:
        try:
            details[pid] = client.get_player(pid, season=SEASON)
        except Exception:  # pylint: disable=broad-exception-caught
            details[pid] = {}
    return details


def fetch_prev_season_stats(
    client: MlbClient, player_ids: list[int]
) -> dict[int, dict[str, Any]]:
    prev = SEASON - 1
    stats: dict[int, dict[str, Any]] = {}
    for pid in player_ids:
        try:
            stats[pid] = client.get_player_season_stats(pid, season=prev)
        except Exception:  # pylint: disable=broad-exception-caught
            stats[pid] = {}
    return stats


def main() -> None:
    client = MlbClient()

    print(f"Fetching roster for team {TEAM_ID} ({SEASON})...")
    players = fetch_roster_players(client)
    print(f"  Found {len(players)} position players")
    player_ids = [p["person"]["id"] for p in players]
    print(f"  Player IDs: {player_ids}")

    print("Fetching game logs...")
    game_logs = fetch_game_logs(client, player_ids)
    print(f"  {len(game_logs)} game-log rows")

    print("Fetching game contexts...")
    game_contexts = build_game_contexts(client, game_logs)
    print(f"  {len(game_contexts)} unique game contexts")

    print("Fetching opponent pitching stats...")
    opponent_pitching = fetch_opponent_pitching(client, game_logs)
    print(f"  {len(opponent_pitching)} teams")

    print("Fetching player details...")
    player_details = fetch_player_details(client, player_ids)
    print(f"  {len(player_details)} players")

    print("Fetching previous-season stats...")
    prev_season_stats = fetch_prev_season_stats(client, player_ids)
    print(f"  {len(prev_season_stats)} players")

    print("Fetching teams list...")
    teams = client.get_teams()
    print(f"  {len(teams)} teams")

    print("Building feature matrix...")
    feature_matrix = build_feature_matrix(
        game_logs,
        season=SEASON,
        teams=teams,
        extra_kwargs={
            "game_contexts": game_contexts,
            "opponent_pitching": opponent_pitching,
            "player_details": player_details,
            "prev_season_stats": prev_season_stats,
        },
    )
    print(f"  {len(feature_matrix)} feature rows")

    metas = describe_features()
    print(f"  {len(metas)} feature columns registered")

    print("Building targets...")
    targets = make_targets(game_logs)
    print(f"  {len(targets)} target rows")

    print("=" * 60)
    print("Training models (walk-forward, 3 folds)...")
    print("=" * 60)

    for target_col in ("target_0.5", "target_1.5"):
        print(f"\n--- Target: {target_col} ---")
        result = train_baselines(
            feature_matrix,
            targets,
            target_col=target_col,
            n_splits=3,
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
            for fm in mdata["fold_metrics"]:
                auc_s = f"{fm['auc']:.4f}" if not (
                    fm.get("auc") is None or fm["auc"] != fm["auc"]
                ) else "N/A"
                print(f"      Fold {fm['fold']}: "
                      f"acc={fm['accuracy']:.4f}  auc={auc_s}  "
                      f"n_train={fm['n_train']}  n_test={fm['n_test']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
