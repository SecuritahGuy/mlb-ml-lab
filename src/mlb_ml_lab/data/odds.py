"""Fetch MLB betting odds from Sportsbook Review (SBR).

Pulls moneyline, run line, and total odds from the ``__NEXT_DATA__``
JSON embedded in SBR's game listing pages. Supports both current and
historical dates.

Usage::

    from mlb_ml_lab.data.odds import fetch_game_odds, SBR_TEAM_MAP

    odds = fetch_game_odds("2026-05-15")
    for g in odds:
        print(g["away_team"], g["home_team"], g["away_ml"], g["home_ml"])
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import date, timedelta
from typing import Any

SBR_BASE = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/"
ODDS_CACHE_DIR = "data/cache/sbr_odds"

# SBR uses different abbreviations than the MLB API for some teams
SBR_TEAM_MAP: dict[str, str] = {
    "CHW": "CWS",
    "WAS": "WSH",
}

# Reverse map: MLB API abbrev → SBR abbrev
_MLB_TO_SBR: dict[str, str] = {v: k for k, v in SBR_TEAM_MAP.items()}


def _parse_sbr_abbrev(sbr_abbrev: str) -> str:
    """Convert SBR team abbreviation to MLB API abbreviation."""
    return SBR_TEAM_MAP.get(sbr_abbrev, sbr_abbrev)


def _mlb_to_sbr(mlb_abbrev: str) -> str:
    """Convert MLB API abbreviation to SBR abbreviation."""
    return _MLB_TO_SBR.get(mlb_abbrev, mlb_abbrev)


def _fetch_page(date_str: str) -> dict[str, Any] | None:
    """Fetch SBR page and extract ``__NEXT_DATA__`` JSON."""
    url = f"{SBR_BASE}?date={date_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        html = urllib.request.urlopen(req, timeout=15).read().decode()
    except Exception:
        return None

    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    )
    if not match:
        return None
    return json.loads(match.group(1))


def fetch_game_odds(
    date_str: str,
    sportsbook: str = "betmgm",
) -> list[dict[str, Any]]:
    """Fetch MLB game odds from SBR for a single date.

    Args:
        date_str: ISO date string e.g. ``"2026-05-15"``.
        sportsbook: Sportsbook slug (default ``"betmgm"``).

    Returns:
        List of dicts with keys: ``game_id``, ``away_team``, ``home_team``,
        ``away_pitcher``, ``home_pitcher``, ``away_ml``, ``home_ml``,
        ``away_spread``, ``home_spread``, ``over``, ``under``,
        ``open_away_ml``, ``open_home_ml``, ``start_time``.
    """
    parsed = _fetch_page(date_str)
    if parsed is None:
        return []

    otm = (
        parsed.get("props", {})
        .get("pageProps", {})
        .get("oddsTables", [{}])[0]
        .get("oddsTableModel", {})
    ) if parsed.get("props", {}).get("pageProps", {}).get("oddsTables") else {}
    game_rows = otm.get("gameRows", [])
    sportsbooks = otm.get("sportsbooks", [])

    # Find the target sportsbook index
    sb_idx = 0
    for i, sb in enumerate(sportsbooks):
        if sb.get("machineName", "") == sportsbook:
            sb_idx = i
            break

    results: list[dict[str, Any]] = []
    for game in game_rows:
        gv = game.get("gameView")
        odds_views = game.get("oddsViews", [])
        if gv is None or sb_idx >= len(odds_views) or odds_views[sb_idx] is None:
            continue
        odds_view = odds_views[sb_idx]
        cur = odds_view.get("currentLine", {})
        open_line = odds_view.get("openingLine", {})

        away_abbrev = gv["awayTeam"]["shortName"]
        home_abbrev = gv["homeTeam"]["shortName"]

        away_starter = gv.get("awayStarter") or {}
        home_starter = gv.get("homeStarter") or {}
        results.append({
            "game_id": gv["gameId"],
            "date": date_str,
            "away_team": _parse_sbr_abbrev(away_abbrev),
            "home_team": _parse_sbr_abbrev(home_abbrev),
            "away_pitcher": away_starter.get("lastName", ""),
            "home_pitcher": home_starter.get("lastName", ""),
            "away_ml": cur.get("awayOdds"),
            "home_ml": cur.get("homeOdds"),
            "away_spread": cur.get("awaySpread"),
            "home_spread": cur.get("homeSpread"),
            "over": cur.get("overOdds"),
            "under": cur.get("underOdds"),
            "open_away_ml": open_line.get("awayOdds"),
            "open_home_ml": open_line.get("homeOdds"),
            "start_time": gv.get("startDate", ""),
        })

    return results


def fetch_odds_range(
    start_date: str,
    end_date: str,
    sportsbook: str = "betmgm",
) -> list[dict[str, Any]]:
    """Fetch odds for all dates in ``[start_date, end_date]``.

    Dates with no MLB games (off-days, off-season) return empty lists.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    all_odds: list[dict[str, Any]] = []
    current = start
    while current <= end:
        ds = current.isoformat()
        odds = fetch_game_odds(ds, sportsbook=sportsbook)
        all_odds.extend(odds)
        current += timedelta(days=1)
    return all_odds


def fetch_league_avg_odds(
    games: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute league-average implied win probability from a set of games.

    Useful for detecting which games have significant market bias.
    """
    total_prob = 0.0
    count = 0
    for g in games:
        for key in ("away_ml", "home_ml"):
            ml = g.get(key)
            if ml is not None and abs(ml) < 10000:
                prob = 100.0 / (abs(ml) + 100.0)
                if ml < 0:
                    prob = 1.0 - prob if ml < 0 else prob
                # Actually: negative ML = favorite, positive = dog
                # Implied prob for -150: 150/(150+100) = 0.6
                # Implied prob for +150: 100/(150+100) = 0.4
                if ml < 0:
                    prob = abs(ml) / (abs(ml) + 100.0)
                else:
                    prob = 100.0 / (ml + 100.0)
                total_prob += prob
                count += 1
    return {"avg_implied_prob": total_prob / count if count > 0 else 0.0}


def load_cached_odds(date_str: str) -> list[dict[str, Any]] | None:
    """Load SBR odds from disk cache."""
    path = os.path.join(ODDS_CACHE_DIR, f"{date_str}.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_cached_odds(date_str: str, odds: list[dict[str, Any]]) -> None:
    """Save SBR odds to disk cache."""
    os.makedirs(ODDS_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(ODDS_CACHE_DIR, f"{date_str}.json")
    with open(cache_path, "w") as f:
        json.dump(odds, f, default=str)


def load_all_cached_odds(dates: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Load cached odds for a list of dates. Missing dates return empty lists."""
    result: dict[str, list[dict[str, Any]]] = {}
    for ds in dates:
        odds = load_cached_odds(ds)
        result[ds] = odds if odds is not None else []
    return result
