"""Harvest a large dataset across multiple teams and seasons.

Usage:
    .venv/bin/python pipeline/harvest_dataset.py

Fetches rosters, game logs, game contexts, team stats, and player data
for all 30 MLB teams across the specified seasons, then builds and
caches the complete feature matrix + targets.

Data flows through the disk cache in ``data/cache/`` — re-running is
near-instant after the first fetch.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from mlb_ml_lab.data.meteostat_weather import MeteostatWeather
from mlb_ml_lab.data.weather import INDOOR_VENUES

from mlb_ml_lab import (
    MlbClient,
    PlayerGameLog,
    build_feature_matrix,
    make_targets,
    save_feature_data,
)

SEASONS = list(range(2021, 2027))  # 2021–2026
POSITIONS_TO_EXCLUDE = {"P"}
MAX_PLAYERS_PER_TEAM = 20  # safety cap


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def get_all_teams(client: MlbClient) -> list[dict[str, Any]]:
    print("  Fetching all teams…")
    teams = client.get_teams()
    print(f"  → {len(teams)} teams")
    return teams


def get_team_rosters(
    client: MlbClient, seasons: list[int],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Fetch rosters for all teams and seasons.
    Returns dict keyed by ``(team_id, season)``.
    """
    all_teams = client.get_teams()
    team_ids = [t["id"] for t in all_teams if t.get("sport", {}).get("id") == 1]
    print(f"  {len(team_ids)} MLB clubs across {len(seasons)} seasons")

    rosters: dict[tuple[int, int], list[dict[str, Any]]] = {}
    len(team_ids) * len(seasons)
    for idx, tid in enumerate(team_ids):
        for s in seasons:
            roster = client.get_roster(tid, season=s)
            # Filter to position players only
            players = [
                p for p in roster
                if (p.get("position") or {}).get("abbreviation", "") not in POSITIONS_TO_EXCLUDE
            ][:MAX_PLAYERS_PER_TEAM]
            rosters[(tid, s)] = players
        if (idx + 1) % 5 == 0:
            print(f"    {idx + 1}/{len(team_ids)} teams done")
    print(f"  → {len(rosters)} team×season rosters cached")
    return rosters


def fetch_game_logs(
    client: MlbClient,
    rosters: dict[tuple[int, int], list[dict[str, Any]]],
) -> list[PlayerGameLog]:
    """Fetch game logs for all position players across all teams/seasons."""
    all_logs: list[PlayerGameLog] = []
    total_players = sum(len(players) for players in rosters.values())
    done = 0
    t0 = time.time()

    for (tid, s), players in rosters.items():
        for p in players:
            pid = p["person"]["id"]
            raw = client.get_player_game_log(pid, season=s)
            for split in raw:
                try:
                    all_logs.append(PlayerGameLog.from_split_dict(split))
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"    {done}/{total_players} players "
                      f"({len(all_logs)} game entries, {rate:.0f}/s)")

    print(f"  → {len(all_logs)} total game-log entries")
    return all_logs


def fetch_game_contexts(
    client: MlbClient, seasons: list[int],
) -> dict[int, dict[str, Any]]:
    """Fetch enriched schedule (venue, probable pitchers, datetime) per season.
    This replaces thousands of per-game API calls with one per season.
    """
    print("  Fetching enriched schedules (1 call per season)…")
    contexts: dict[int, dict[str, Any]] = {}
    for s in seasons:
        ctx = client.get_enriched_schedule(season=s)
        contexts.update(ctx)
        print(f"    {s}: {len(ctx)} games")
    print(f"  → {len(contexts)} total game contexts")
    return contexts


def fetch_season_schedules(
    client: MlbClient, seasons: list[int],
) -> dict[int, list[dict[str, Any]]]:
    """Fetch full season schedules."""
    print("  Fetching season schedules…")
    schedules: dict[int, list[dict[str, Any]]] = {}
    for s in seasons:
        schedules[s] = client.get_season_schedule(season=s)
    print(f"  → {len(schedules)} seasons")
    return schedules


def fetch_team_season_stats(
    client: MlbClient, seasons: list[int],
) -> dict[str, dict[int, Any]]:
    """Fetch team-level hitting & pitching stats per season."""
    all_teams = client.get_teams()
    team_ids = [t["id"] for t in all_teams if t.get("sport", {}).get("id") == 1]

    results: dict[str, dict[int, Any]] = {
        "pitching": {},
        "hitting": {},
        "monthly_pitching": {},
        "fielding": {},
    }

    def safe(fn, *args, **kw):
        try:
            return fn(*args, **kw) or {}
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"    Warning: {fn.__name__} failed: {e}")
            return {}

    for s in seasons:
        print(f"  {s}: team stats…")
        results["pitching"][s] = safe(client.get_team_pitching_stats, team_ids, s)
        results["hitting"][s] = safe(client.get_team_hitting_stats, team_ids, s)
        results["monthly_pitching"][s] = safe(
            client.get_team_pitching_monthly_stats, team_ids, s
        )
        results["fielding"][s] = safe(client.get_team_fielding_stats, team_ids, s)

    return results


def fetch_bullpen_stats(
    client: MlbClient, team_ids: list[int], seasons: list[int],
) -> dict[int, dict[int, dict[str, float]]]:
    """Fetch bullpen stats per team per season.
    Returns dict: season → team_id → stats.
    """
    print("  Fetching bullpen stats…")
    by_season: dict[int, dict[int, dict[str, float]]] = {}
    for s in seasons:
        by_season[s] = {}
        for tid in team_ids:
            try:
                bp = client.get_team_bullpen_stats(tid, s)
                if bp:
                    by_season[s][tid] = bp
            except Exception:  # pylint: disable=broad-exception-caught
                pass
    total = sum(len(v) for v in by_season.values())
    print(f"  → {total} team×season bullpen stat sets")
    return by_season


def fetch_player_details(
    client: MlbClient,
    rosters: dict[tuple[int, int], list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Fetch player biographical details."""
    print("  Fetching player details…")
    seen_ids: set[int] = set()
    details: dict[int, dict[str, Any]] = {}
    sum(len(players) for players in rosters.values())
    done = 0
    for players in rosters.values():
        for p in players:
            pid = p["person"]["id"]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            try:
                details[pid] = client.get_player(pid)
            except Exception:  # pylint: disable=broad-exception-caught
                details[pid] = {}
            done += 1
    print(f"  → {len(details)} unique players")
    return details


def fetch_career_stats(
    client: MlbClient,
    player_ids: set[int],
    seasons: list[int],
    group: str = "hitting",
) -> dict[int, dict[str, Any]]:
    """Weighted career average from last 3 seasons (50/30/20 weights)."""
    print(f"  Fetching career {group} stats…")
    weights = [0.5, 0.3, 0.2]
    result: dict[int, dict[str, Any]] = {}
    len(player_ids)
    done = 0
    for pid in player_ids:
        weighted: dict[str, float] = {}
        total_w = 0.0
        for i, w in enumerate(weights):
            s = max(seasons) - 1 - i
            try:
                stats = client.get_player_season_stats(pid, season=s, group=group)
            except Exception:
                continue
            if not stats:
                continue
            for key in stats:
                raw = stats[key]
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
        done += 1
    print(f"  → {len(result)} players")
    return result


def fetch_statcast_data(
    client: MlbClient,
    seasons: list[int],
) -> tuple[list[dict[str, str]] | None, list[dict[str, str]] | None]:
    """Fetch Savant leaderboard data (xBA, barrel%, etc.) for recent years."""
    print("Fetching Statcast leaderboard data…")
    statcast_batters: list[dict[str, str]] = []
    expected_stats: list[dict[str, str]] = []
    for s in [2024]:  # Statcast leaderboard available for recent seasons
        try:
            sb = client.get_statcast_batters(season=s)
            if sb:
                statcast_batters.extend(sb)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            es = client.get_expected_stats(season=s)
            if es:
                expected_stats.extend(es)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    print(f"  Statcast batters: {len(statcast_batters)}, "
          f"Expected stats: {len(expected_stats)}")
    return statcast_batters or None, expected_stats or None


# ---------------------------------------------------------------------------
# Additional data fetchers (game pace, team leaders, streaks)
# ---------------------------------------------------------------------------


def fetch_game_pace_stats(
    client: MlbClient, team_ids: list[int], seasons: list[int],
) -> dict[int, dict[str, float]]:
    """Fetch game pace (time per game, pitches per game) per team per season."""
    result: dict[int, dict[str, float]] = {}
    for s in seasons:
        for tid in team_ids:
            try:
                pace_rows = client.get_game_pace(s, team_id=tid)
                if pace_rows:
                    p = pace_rows[0]
                    result[(tid, s)] = {
                        "time_per_game": p.get("timePerGame"),
                        "pitches_per_game": p.get("pitchesPerGame"),
                    }
            except Exception:  # pylint: disable=broad-exception-caught
                pass
    return result


def fetch_team_leaders(
    client: MlbClient, team_ids: list[int], seasons: list[int],
) -> dict[int, dict[str, float]]:
    """Fetch top hitter stats per team (avg, HR, RBI)."""
    result: dict[int, dict[str, float]] = {}
    for s in seasons:
        for tid in team_ids:
            try:
                leaders = client.get_team_leaders(
                    tid, s,
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
                    result[(tid, s)] = ld
            except Exception:  # pylint: disable=broad-exception-caught
                pass
    return result


def _parse_avg(val: str | float | None) -> float | None:
    if val is None or val == "" or val == ".---":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Weather enrichment via Meteostat
# ---------------------------------------------------------------------------


def enrich_weather_meteostat(
    game_contexts: dict[int, dict[str, Any]],
    teams: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Fill missing weather_* fields in game contexts from Meteostat."""
    # Build venue map: team_id → venue_id
    venue_map: dict[int, int] = {}
    for t in teams:
        venue = t.get("venue") or {}
        vid = venue.get("id")
        if vid is not None:
            venue_map[t["id"]] = vid

    mw = MeteostatWeather()
    enriched: dict[int, dict[str, Any]] = {}
    n_filled = 0
    n_skipped = 0

    for gid, ctx in game_contexts.items():
        row = dict(ctx)
        # Skip if already has weather_temp
        if row.get("weather_temp") is not None:
            n_skipped += 1
            enriched[gid] = row
            continue

        # Find home team → venue
        home_id = ctx.get("home_team_id")
        vid = venue_map.get(home_id) if home_id else None
        if vid is None or vid in INDOOR_VENUES:
            enriched[gid] = row
            if vid in INDOOR_VENUES:
                row["weather_temp"] = 72
                row["weather_wind"] = "0 mph"
                row["weather_condition"] = "Indoor"
            continue

        game_dt_str: str | None = ctx.get("game_datetime")
        if not game_dt_str:
            enriched[gid] = row
            continue

        try:
            dt = datetime.fromisoformat(game_dt_str)
        except (ValueError, TypeError):
            enriched[gid] = row
            continue

        wx = mw.weather(vid, target_time=dt)
        if wx and wx.get("source") != "indoor":
            row["weather_temp"] = wx.get("temp")
            row["weather_wind"] = wx.get("wind_speed", "")
            row["weather_condition"] = wx.get("conditions")
            n_filled += 1

        enriched[gid] = row

    print(f"  Meteostat weather: {n_filled} filled, {n_skipped} already had data")
    return enriched


# ---------------------------------------------------------------------------
# League-level helpers
# ---------------------------------------------------------------------------


def compute_league_stats(
    client: MlbClient, seasons: list[int],
) -> dict[int, dict[str, float]]:
    """Compute league-wide avg/obp/slg/ops/runs per season."""
    all_teams = client.get_teams()
    team_ids = [t["id"] for t in all_teams if t.get("sport", {}).get("id") == 1]
    results: dict[int, dict[str, float]] = {}

    for s in seasons:
        hitting = client.get_team_hitting_stats(team_ids, s) or {}
        total_ab = sum(int(st.get("atBats", 0)) for st in hitting.values())
        total_h = sum(int(st.get("hits", 0)) for st in hitting.values())
        total_bb = sum(int(st.get("baseOnBalls", 0)) for st in hitting.values())
        total_r = sum(int(st.get("runs", 0)) for st in hitting.values())
        total_g = sum(int(st.get("gamesPlayed", 0)) for st in hitting.values())

        if total_ab == 0:
            results[s] = {}
            continue

        _1b = total_h - sum(
            int(st.get("doubles", 0)) + int(st.get("triples", 0))
            + int(st.get("homeRuns", 0))
            for st in hitting.values()
        )
        avg = round(total_h / total_ab, 3)
        obp = round((total_h + total_bb) / (total_ab + total_bb), 3)
        slg = round((_1b + 2 * sum(int(st.get("doubles", 0)) for st in hitting.values())
                      + 3 * sum(int(st.get("triples", 0)) for st in hitting.values())
                      + 4 * sum(int(st.get("homeRuns", 0)) for st in hitting.values()))
                     / total_ab, 3)
        results[s] = {
            "avg": avg, "obp": obp, "slg": slg, "ops": round(obp + slg, 3),
            "runs_per_game": round(total_r / max(total_g, 1), 2),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    client = MlbClient()

    # ── Stage 1: Teams & Rosters ─────────────────────────────────────────
    print("\n=== Stage 1: Teams & Rosters ===")
    teams = get_all_teams(client)
    all_team_ids = [t["id"] for t in teams if t.get("sport", {}).get("id") == 1]
    rosters = get_team_rosters(client, SEASONS)

    # Collect unique player IDs
    all_player_ids: set[int] = set()
    for players in rosters.values():
        for p in players:
            all_player_ids.add(p["person"]["id"])
    print(f"  {len(all_player_ids)} unique position players across all seasons")

    # ── Stage 2: Game Logs ───────────────────────────────────────────────
    print("\n=== Stage 2: Game Logs ===")
    game_logs = fetch_game_logs(client, rosters)
    print(f"  Date range: {game_logs[0].date if game_logs else 'N/A'} – "
          f"{game_logs[-1].date if game_logs else 'N/A'}")

    # ── Stage 3: Game Contexts (enriched schedule) ────────────────────────
    print("\n=== Stage 3: Game Contexts ===")
    game_contexts = fetch_game_contexts(client, SEASONS)

    # ── Stage 3b: Weather enrichment via Meteostat ────────────────────────
    print("\n=== Stage 3b: Weather Enrichment (Meteostat) ===")
    game_contexts = enrich_weather_meteostat(game_contexts, teams)

    # ── Stage 4: Season schedules ────────────────────────────────────────
    print("\n=== Stage 4: Season Schedules ===")
    season_schedules = fetch_season_schedules(client, SEASONS)

    # ── Stage 5: Team stats ──────────────────────────────────────────────
    print("\n=== Stage 5: Team Stats ===")
    team_stats = fetch_team_season_stats(client, SEASONS)

    # ── Stage 6: Bullpen stats ───────────────────────────────────────────
    print("\n=== Stage 6: Bullpen Stats ===")
    bullpen_stats = fetch_bullpen_stats(client, all_team_ids, SEASONS)

    # ── Stage 7: Player details ──────────────────────────────────────────
    print("\n=== Stage 7: Player Details ===")
    player_details = fetch_player_details(client, rosters)
    # Also fetch details for pitchers (needed for platoon advantage)
    pitcher_ids = set()
    for ctx in game_contexts.values():
        h = ctx.get("home_probable_pitcher_id")
        a = ctx.get("away_probable_pitcher_id")
        if h:
            pitcher_ids.add(h)
        if a:
            pitcher_ids.add(a)
    existing = set(player_details)
    new_pitchers = pitcher_ids - existing
    if new_pitchers:
        print(f"  Fetching details for {len(new_pitchers)} pitchers...")
        for pid in new_pitchers:
            try:
                player_details[pid] = client.get_player(pid)
            except Exception:
                pass
    print(f"  → {len(player_details)} total players (incl. {len(new_pitchers)} pitchers)")

    # ── Stage 8: Career hitting stats ────────────────────────────────────
    print("\n=== Stage 8: Career Hitting Stats ===")
    career_hitting = fetch_career_stats(client, all_player_ids, SEASONS, "hitting")

    # ── Stage 9: Career pitching stats (for opponent pitcher features) ───
    print("\n=== Stage 9: Career Pitching Stats ===")
    # Get all pitcher IDs from enriched schedule for opp_pitcher_* features
    pitcher_ids: set[int] = set()
    for ctx in game_contexts.values():
        h = ctx.get("home_probable_pitcher_id")
        a = ctx.get("away_probable_pitcher_id")
        if h:
            pitcher_ids.add(h)
        if a:
            pitcher_ids.add(a)
    print(f"  {len(pitcher_ids)} unique pitcher IDs from schedule")
    career_pitching = fetch_career_stats(client, pitcher_ids, SEASONS, "pitching")

    # ── Stage 10: League stats ───────────────────────────────────────────
    print("\n=== Stage 10: League Stats ===")
    league_stats = compute_league_stats(client, SEASONS)
    print(f"  {len(league_stats)} seasons")

    # ── Stage 11: Statcast leaderboard data ───────────────────────────────
    print("\n=== Stage 11: Statcast Data ===")
    statcast_batters, expected_stats = fetch_statcast_data(client, SEASONS)

    # ── Stage 11a: Game pace stats ───────────────────────────────────────
    print("\n=== Stage 11a: Game Pace Stats ===")
    game_pace_stats = fetch_game_pace_stats(client, all_team_ids, SEASONS)
    print(f"  {len(game_pace_stats)} team×season pace records")

    # ── Stage 11b: Team leaders ──────────────────────────────────────────
    print("\n=== Stage 11b: Team Leaders ===")
    team_leaders = fetch_team_leaders(client, all_team_ids, SEASONS)
    print(f"  {len(team_leaders)} team×season leader sets")

    # ── Stage 12: Build Feature Matrix ───────────────────────────────────
    print("\n=== Stage 12: Building Feature Matrix ===")
    # We need to build per-season so game_contexts match
    all_feature_rows: list[dict[str, Any]] = []
    all_targets: list[dict[str, Any]] = []

    for s in SEASONS:
        season_logs = [
            log for log in game_logs
            if log.date[:4] == str(s)
        ]
        if not season_logs:
            print(f"  {s}: no game logs, skipping")
            continue

        # Build opponent pitching map for this season
        opp_pitching = team_stats["pitching"].get(s, {})
        monthly_pitching = team_stats["monthly_pitching"].get(s, {})
        team_fielding = team_stats["fielding"].get(s, {})

        season_league_stats = league_stats.get(s, {})

        print(f"  Building feature matrix for {s} ({len(season_logs)} logs)…")

        # Build per-season lookups for additional data
        s_game_pace = {
            k[0]: v for k, v in game_pace_stats.items() if k[1] == s
        }
        s_team_leaders = {
            k[0]: v for k, v in team_leaders.items() if k[1] == s
        }

        fm = build_feature_matrix(
            season_logs,
            season=s,
            teams=teams,
            statcast_batters=statcast_batters,
            expected_stats=expected_stats,
            extra_kwargs={
                "game_contexts": game_contexts,
                "opponent_pitching": opp_pitching,
                "monthly_pitching": monthly_pitching,
                "team_fielding": team_fielding,
                "player_details": player_details,
                "career_stats": career_hitting,
                "pitcher_stats": career_pitching,
                "league_stats": season_league_stats,
                "season_schedule": season_schedules.get(s, []),
                "bullpen_stats": bullpen_stats.get(s, {}),
                "game_pace_stats": s_game_pace,
                "team_leaders": s_team_leaders,
            },
        )
        tgt = make_targets(season_logs)
        all_feature_rows.extend(fm)
        all_targets.extend(tgt)
        print(f"    → {len(fm)} feature rows, {len(tgt)} target rows")

    print(f"\n  Total: {len(all_feature_rows)} feature rows, "
          f"{len(all_targets)} target rows")

    # ── Stage 13: Save ───────────────────────────────────────────────────
    print("\n=== Stage 13: Saving Dataset ===")
    season_range = f"{min(SEASONS)}_{max(SEASONS)}"
    n_teams = len(all_team_ids)
    output_dir = f"data/datasets/full_{season_range}_{n_teams}teams"
    save_feature_data(
        all_feature_rows,
        all_targets,
        output_dir,
        metadata={
            "seasons": SEASONS,
            "team_count": n_teams,
            "player_count": len(all_player_ids),
            "game_log_count": len(game_logs),
        },
    )
    print(f"  Saved to {output_dir}/")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed / 60:.1f} minutes")
    print(f"Dataset: {output_dir}")


if __name__ == "__main__":
    main()
