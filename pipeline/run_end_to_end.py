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


def fetch_monthly_pitching(
    client: MlbClient, game_logs: list[Any]
) -> dict[int, list[dict[str, Any]]]:
    opp_ids = list({log.opponent_id for log in game_logs})
    try:
        return client.get_team_pitching_monthly_stats(opp_ids, SEASON)
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def fetch_team_fielding(
    client: MlbClient, game_logs: list[Any]
) -> dict[int, dict[str, Any]]:
    opp_ids = list({log.opponent_id for log in game_logs})
    try:
        return client.get_team_fielding_stats(opp_ids, SEASON)
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def fetch_league_stats(client: MlbClient) -> dict[str, float]:
    """Compute league-wide avg/obp/slg/ops/runs for the season."""
    try:
        all_teams = client.get_teams()
        all_ids = [t["id"] for t in all_teams]
        hitting = client.get_team_hitting_stats(all_ids, SEASON)
    except Exception:  # pylint: disable=broad-exception-caught
        return {}

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
    ops = round(obp + slg, 3)
    rpg = round(total_r / max(total_g, 1), 2)

    return {
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": ops,
        "runs_per_game": rpg,
    }


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


def fetch_career_hitting_stats(
    client: MlbClient, player_ids: list[int]
) -> dict[int, dict[str, Any]]:
    """Weighted career average from last 3 seasons (50/30/20 weights)."""
    weights = [0.5, 0.3, 0.2]
    result: dict[int, dict[str, Any]] = {}
    for pid in player_ids:
        weighted: dict[str, float] = {}
        total_w = 0.0
        for i, w in enumerate(weights):
            s = SEASON - 1 - i
            try:
                stats = client.get_player_season_stats(pid, season=s)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if not stats:
                continue
            for key in ("avg", "obp", "slg", "ops", "homeRuns"):
                raw = stats.get(key)
                if raw is not None:
                    try:
                        weighted[key] = weighted.get(key, 0.0) + float(raw) * w
                    except (ValueError, TypeError):
                        pass
            total_w += w
        if weighted and total_w > 0:
            result[pid] = {k: round(v / total_w, 3) for k, v in weighted.items()}
        else:
            result[pid] = {}
    return result


def fetch_pitcher_career_stats(
    client: MlbClient, pitcher_ids: list[int]
) -> dict[int, dict[str, Any]]:
    """Weighted career pitching average from last 3 seasons."""
    weights = [0.5, 0.3, 0.2]
    result: dict[int, dict[str, Any]] = {}
    for pid in pitcher_ids:
        weighted: dict[str, float] = {}
        total_w = 0.0
        for i, w in enumerate(weights):
            s = SEASON - 1 - i
            try:
                stats = client.get_player_season_stats(pid, season=s, group="pitching")
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if not stats:
                continue
            for key in ("era", "strikeoutsPer9Inn", "whip", "avg", "homeRunsPer9",
                        "battersFaced", "inningsPitched"):
                raw = stats.get(key)
                if raw is not None:
                    try:
                        weighted[key] = weighted.get(key, 0.0) + float(raw) * w
                    except (ValueError, TypeError):
                        pass
            total_w += w
        if weighted and total_w > 0:
            result[pid] = {k: round(v / total_w, 3) for k, v in weighted.items()}
        else:
            result[pid] = {}
    return result


def collect_pitcher_ids(game_contexts: dict[int, dict[str, Any]]) -> list[int]:
    seen: set[int] = set()
    for ctx in game_contexts.values():
        hid = ctx.get("home_probable_pitcher_id")
        aid = ctx.get("away_probable_pitcher_id")
        if hid:
            seen.add(hid)
        if aid:
            seen.add(aid)
    return list(seen)


def fetch_pitcher_data(
    client: MlbClient, game_contexts: dict[int, dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    pitcher_ids = collect_pitcher_ids(game_contexts)
    print(f"  {len(pitcher_ids)} unique pitchers")
    details = fetch_player_details(client, pitcher_ids)
    print(f"  {len(details)} pitcher details")
    stats = fetch_pitcher_career_stats(client, pitcher_ids)
    print(f"  {len(stats)} pitcher stat sets")
    return details, stats


def main() -> None:
    client = MlbClient()

    players = fetch_roster_players(client)
    print(f"  Found {len(players)} position players for team {TEAM_ID}")
    player_ids = [p["person"]["id"] for p in players]

    print("Fetching game logs...")
    game_logs = fetch_game_logs(client, player_ids)
    print(f"  {len(game_logs)} game-log rows")

    print("Fetching game contexts...")
    game_contexts = build_game_contexts(client, game_logs)
    print(f"  {len(game_contexts)} unique game contexts")

    opponent_pitching = fetch_opponent_pitching(client, game_logs)
    monthly_pitching = fetch_monthly_pitching(client, game_logs)
    team_fielding = fetch_team_fielding(client, game_logs)

    player_details = fetch_player_details(client, player_ids)
    career_stats = fetch_career_hitting_stats(client, player_ids)

    pitcher_details, pitcher_stats = fetch_pitcher_data(client, game_contexts)
    player_details.update(pitcher_details)

    league_stats = fetch_league_stats(client)
    teams = client.get_teams()

    print("Building feature matrix...")
    feature_matrix = build_feature_matrix(
        game_logs,
        season=SEASON,
        teams=teams,
        extra_kwargs={
            "game_contexts": game_contexts,
            "opponent_pitching": opponent_pitching,
            "monthly_pitching": monthly_pitching,
            "team_fielding": team_fielding,
            "player_details": player_details,
            "career_stats": career_stats,
            "pitcher_stats": pitcher_stats,
            "league_stats": league_stats,
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
