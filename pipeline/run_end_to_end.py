"""End-to-end pipeline: fetch MLB data → featurize → train → evaluate.

Usage:
    poetry run python pipeline/run_end_to_end.py

Configurable constants at the top of the file (``TEAM_ID``, ``SEASON``,
``MAX_PLAYERS``, etc.).
"""

from __future__ import annotations

from typing import Any

from mlb_ml_lab import (
    MlbClient,
    PlayerGameLog,
    build_feature_matrix,
    describe_features,
    make_targets,
)
from mlb_ml_lab.models import train_baselines

# ── Configuration ─────────────────────────────────────────────────────────────
TEAM_ID = 108  # Los Angeles Angels
SEASON = 2024
MAX_PLAYERS = 5
SAMPLE_GAMES = 60  # games per player (null values allowed)


def fetch_roster_players(client: MlbClient) -> list[dict[str, Any]]:
    """Fetch roster and return position players only."""
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
    """Fetch game logs for a list of player IDs, limited to SAMPLE_GAMES."""
    all_logs: list[PlayerGameLog] = []
    for pid in player_ids:
        raw = client.get_player_game_log(pid, season=SEASON)
        for split in raw[:SAMPLE_GAMES]:
            all_logs.append(PlayerGameLog.from_split_dict(split))
    return all_logs


def build_game_contexts(
    client: MlbClient, game_logs: list[Any]
) -> dict[int, dict[str, Any]]:
    """Fetch game context (venue, weather) for unique game_pks."""
    seen = set()
    game_pks: list[int] = []
    for log in game_logs:
        pk = log.game_pk if hasattr(log, "game_pk") else log.get("game_pk")
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
    """Fetch pitching stats for all unique opponent teams."""
    opp_ids = set()
    for log in game_logs:
        opp = log.opponent_id if hasattr(log, "opponent_id") else log.get("opponent_id")
        opp_ids.add(opp)

    pitching: dict[int, dict[str, float]] = {}
    for tid in opp_ids:
        try:
            stats = client.get_team_pitching_stats(SEASON, tid)
            pitching[tid] = stats
        except Exception:  # pylint: disable=broad-exception-caught
            pitching[tid] = {}
    return pitching


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
        },
    )
    print(f"  {len(feature_matrix)} feature rows")

    metas = describe_features()
    print(f"  {len(metas)} feature columns registered")
    for m in metas:
        print(f"    {m.name:30s} {m.source:15s} {m.description}")

    print("Building targets...")
    targets = make_targets(game_logs)
    print(f"  {len(targets)} target rows")

    print("=" * 60)
    print("Training baseline models (walk-forward, 3 folds)...")
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
            print(f"\n  Model: {model_type.upper()}")
            print(f"    Avg accuracy: {mdata['avg_accuracy']:.4f}")
            print(f"    Avg AUC:      {mdata['avg_auc']:.4f}")
            print(f"    Folds:        {mdata['n_folds']}")
            for fm in mdata["fold_metrics"]:
                auc_str = f"{fm['auc']:.4f}" if not fm.get("auc") != fm.get("auc") else "N/A"
                print(f"      Fold {fm['fold']}: "
                      f"acc={fm['accuracy']:.4f}  auc={auc_str}  "
                      f"n_train={fm['n_train']}  n_test={fm['n_test']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
