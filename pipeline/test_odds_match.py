"""Test SBR odds scraper: fetch historical odds and match to our game data.

Usage:
    poetry run python pipeline/test_odds_match.py
"""

from __future__ import annotations

from collections import defaultdict

from mlb_ml_lab import load_game_logs
from mlb_ml_lab.data.odds import fetch_game_odds

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"


def main() -> None:
    print("Loading game logs...")
    raw_logs = load_game_logs(CACHED_DATASET)
    print(f"  {len(raw_logs)} game logs")

    # Build index of (date, away_team_abbrev, home_team_abbrev) → games
    # We need to map team_id → abbreviation. Let's fetch teams from MLB API.
    from mlb_ml_lab import MlbClient

    client = MlbClient()
    teams = client.get_teams()
    id_to_abbrev: dict[int, str] = {}
    for t in teams:
        if t.get("sport", {}).get("id") == 1:
            id_to_abbrev[t["id"]] = t.get("abbreviation", "")

    # Build game index: (date, away_abbrev, home_abbrev) → list of game logs
    game_index: dict[tuple[str, str, str], list] = defaultdict(list)
    for d in raw_logs:
        away_abbrev = id_to_abbrev.get(d.get("opponent_id", 0), "")
        home_abbrev = id_to_abbrev.get(d.get("team_id", 0), "")
        date_str = d["date"][:10]  # YYYY-MM-DD
        game_index[(date_str, away_abbrev, home_abbrev)].append(d)

    print(f"  {len(game_index)} unique (date, away, home) game slots")

    # Test a few dates across different seasons
    test_dates = [
        "2022-05-15",
        "2022-07-04",
        "2022-09-01",
        "2023-05-15",
        "2023-07-04",
        "2023-09-01",
        "2024-05-15",
        "2024-07-04",
        "2024-09-01",
        "2025-05-15",
        "2025-07-04",
        "2025-09-01",
    ]

    total_matched = 0
    total_sbr_games = 0

    for date_str in test_dates:
        odds = fetch_game_odds(date_str, sportsbook="betmgm")

        # SBR shows away_ml/home_ml ordered as away vs home
        # Our game logs have team_id (home) and opponent_id (away)
        # So: SBR away_team ≈ our opponent_id, SBR home_team ≈ our team_id
        sbr_games = 0
        matched = 0
        our_games = 0

        for g in odds:
            sbr_games += 1
            key = (date_str, g["away_team"], g["home_team"])
            our = game_index.get(key)

            # Try reverse (in case of home/away flip)
            if our is None:
                key = (date_str, g["home_team"], g["away_team"])
                our = game_index.get(key)

            if our:
                matched += 1
                our_games += len(our)

        total_matched += matched
        total_sbr_games += sbr_games

        ml_list = [g.get("away_ml") for g in odds if g.get("away_ml") is not None]
        if ml_list:
            print(
                f"  {date_str}: SBR={sbr_games} games, matched={matched}/{sbr_games}, "
                f"odds_range={min(ml_list)}~{max(ml_list)}"
            )
        else:
            print(f"  {date_str}: {sbr_games} games, matched={matched}")

    pct = total_matched / total_sbr_games * 100 if total_sbr_games else 0
    print(f"\n  Total: {total_matched}/{total_sbr_games} games matched ({pct:.0f}%)")
    print()

    # Also test a dense week
    print("Testing 7-day window: 2024-05-13 → 2024-05-19...")
    from datetime import date, timedelta

    start = date(2024, 5, 13)
    end = date(2024, 5, 19)
    week_matched = 0
    week_sbr = 0
    current = start
    while current <= end:
        ds = current.isoformat()
        odds = fetch_game_odds(ds, sportsbook="betmgm")
        for g in odds:
            week_sbr += 1
            key = (ds, g["away_team"], g["home_team"])
            our = game_index.get(key)
            if our is None:
                key = (ds, g["home_team"], g["away_team"])
                our = game_index.get(key)
            if our:
                week_matched += 1
        current += timedelta(days=1)
    print(
        f"  Week matched: {week_matched}/{week_sbr} ({week_matched / week_sbr * 100:.0f}%)"
    )


if __name__ == "__main__":
    main()
