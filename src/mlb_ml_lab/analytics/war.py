from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

WOBA_PARAMS: dict[int, dict[str, float]] = {
    2016: {
        "wBB": 0.692,
        "wHBP": 0.723,
        "w1B": 0.877,
        "w2B": 1.232,
        "w3B": 1.579,
        "wHR": 2.003,
        "scale": 1.257,
    },
    2017: {
        "wBB": 0.693,
        "wHBP": 0.726,
        "w1B": 0.872,
        "w2B": 1.220,
        "w3B": 1.552,
        "wHR": 1.966,
        "scale": 1.260,
    },
    2018: {
        "wBB": 0.690,
        "wHBP": 0.722,
        "w1B": 0.872,
        "w2B": 1.224,
        "w3B": 1.563,
        "wHR": 1.982,
        "scale": 1.253,
    },
    2019: {
        "wBB": 0.689,
        "wHBP": 0.720,
        "w1B": 0.878,
        "w2B": 1.238,
        "w3B": 1.587,
        "wHR": 2.024,
        "scale": 1.255,
    },
    2020: {
        "wBB": 0.693,
        "wHBP": 0.725,
        "w1B": 0.872,
        "w2B": 1.213,
        "w3B": 1.530,
        "wHR": 1.934,
        "scale": 1.252,
    },
    2021: {
        "wBB": 0.691,
        "wHBP": 0.723,
        "w1B": 0.872,
        "w2B": 1.218,
        "w3B": 1.551,
        "wHR": 1.965,
        "scale": 1.257,
    },
    2022: {
        "wBB": 0.689,
        "wHBP": 0.721,
        "w1B": 0.874,
        "w2B": 1.232,
        "w3B": 1.581,
        "wHR": 2.012,
        "scale": 1.254,
    },
    2023: {
        "wBB": 0.689,
        "wHBP": 0.722,
        "w1B": 0.876,
        "w2B": 1.230,
        "w3B": 1.573,
        "wHR": 1.996,
        "scale": 1.255,
    },
    2024: {
        "wBB": 0.689,
        "wHBP": 0.721,
        "w1B": 0.874,
        "w2B": 1.231,
        "w3B": 1.578,
        "wHR": 2.004,
        "scale": 1.254,
    },
}
for _yr in (2025, 2026):
    WOBA_PARAMS[_yr] = dict(WOBA_PARAMS[2024])

TEAM_VENUE: list[tuple[int, int, int, int]] = [
    (108, 2016, 2026, 1),
    (109, 2016, 2026, 15),
    (110, 2016, 2026, 2),
    (111, 2016, 2026, 3),
    (112, 2016, 2026, 17),
    (113, 2016, 2026, 2602),
    (114, 2016, 2026, 5),
    (115, 2016, 2026, 19),
    (116, 2016, 2026, 2394),
    (117, 2016, 2026, 2392),
    (118, 2016, 2026, 7),
    (119, 2016, 2026, 22),
    (120, 2016, 2026, 3309),
    (121, 2016, 2026, 3289),
    (133, 2016, 2024, 32),
    (133, 2025, 2026, 2529),
    (134, 2016, 2026, 31),
    (135, 2016, 2026, 2680),
    (136, 2016, 2026, 680),
    (137, 2016, 2026, 2395),
    (138, 2016, 2026, 2889),
    (139, 2016, 2026, 12),
    (140, 2016, 2019, 13),
    (140, 2020, 2026, 5325),
    (141, 2016, 2026, 14),
    (142, 2016, 2026, 3312),
    (143, 2016, 2026, 2681),
    (144, 2016, 2016, 4705),
    (144, 2017, 2026, 4705),
    (145, 2016, 2026, 4),
    (146, 2016, 2026, 4169),
    (147, 2016, 2026, 3313),
    (158, 2016, 2026, 32),
]

_LATEST_VENUE: dict[int, int] = {}
for _tid, _lo, _hi, _vid in TEAM_VENUE:
    _LATEST_VENUE[_tid] = _vid

POSITIONAL_RUNS: dict[str, float] = {
    "C": 12.5,
    "1B": -12.5,
    "2B": 2.5,
    "3B": 2.5,
    "SS": 7.5,
    "LF": -7.5,
    "CF": 2.5,
    "RF": -7.5,
    "OF": -7.5,
    "DH": -17.5,
    "PH": 0.0,
    "PR": 0.0,
}

DEFAULT_RUNS_PER_WIN = 10.0
RUNS_PER_WIN_BY_SEASON: dict[int, float] = {
    2016: 9.7,
    2017: 9.6,
    2018: 9.8,
    2019: 9.5,
    2020: 9.7,
    2021: 9.6,
    2022: 9.7,
    2023: 9.8,
    2024: 9.7,
}
for _yr in (2025, 2026):
    RUNS_PER_WIN_BY_SEASON[_yr] = RUNS_PER_WIN_BY_SEASON.get(_yr, DEFAULT_RUNS_PER_WIN)


def _woba_params(season: int) -> dict[str, float]:
    if season in WOBA_PARAMS:
        return WOBA_PARAMS[season]
    return WOBA_PARAMS[min(WOBA_PARAMS, key=lambda y: abs(y - season))]


def _runs_per_win(season: int) -> float:
    if season in RUNS_PER_WIN_BY_SEASON:
        return RUNS_PER_WIN_BY_SEASON[season]
    return RUNS_PER_WIN_BY_SEASON[
        min(RUNS_PER_WIN_BY_SEASON, key=lambda y: abs(y - season))
    ]


def venue_for_team(team_id: int, season: int) -> int | None:
    for tid, lo, hi, vid in TEAM_VENUE:
        if tid == team_id and lo <= season <= hi:
            return vid
    return None


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------


def get_park_factor(team_id: int, season: int, cache_dir: str | None = None) -> float:
    from mlb_ml_lab.data.parks import ParkFactors

    vid = venue_for_team(team_id, season)
    if vid is None:
        vid = _LATEST_VENUE.get(team_id)
    if vid is None:
        return 1.0
    pf = ParkFactors(cache_dir=cache_dir)
    try:
        factor = pf.factor(vid, metric="runs", season=season)
        return factor if factor and factor > 0 else 1.0
    except Exception:
        return 1.0
    finally:
        pf.close()


# ---------------------------------------------------------------------------
# Player season aggregation
# ---------------------------------------------------------------------------


def aggregate_player_seasons(
    game_logs: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    agg: dict[tuple[int, int], dict[str, Any]] = {}
    for log in game_logs:
        pid = log["player_id"]
        season = int(log["season"])
        key = (pid, season)
        if key not in agg:
            agg[key] = {
                "player_id": pid,
                "player_name": log.get("player_name", ""),
                "season": season,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "d": 0,
                "t": 0,
                "hr": 0,
                "rbi": 0,
                "bb": 0,
                "so": 0,
                "position_freq": {},
                "team_games": {},
            }
        s = agg[key]
        game_pk = log["game_pk"]
        if game_pk in s.setdefault("_seen_games", set()):
            continue
        s["_seen_games"].add(game_pk)
        s["pa"] += int(log.get("plate_appearances", 0))
        s["ab"] += int(log.get("at_bats", 0))
        s["h"] += int(log.get("hits", 0))
        s["r"] += int(log.get("runs", 0))
        s["d"] += int(log.get("doubles", 0))
        s["t"] += int(log.get("triples", 0))
        s["hr"] += int(log.get("home_runs", 0))
        s["rbi"] += int(log.get("rbi", 0))
        s["bb"] += int(log.get("walks", 0))
        s["so"] += int(log.get("strikeouts", 0))
        tid = log["team_id"]
        s["team_games"][tid] = s["team_games"].get(tid, 0) + 1
        pos = log.get("position_abbr", "")
        if pos:
            s["position_freq"][pos] = s["position_freq"].get(pos, 0) + 1
    for s in agg.values():
        s["1b"] = s["h"] - s["d"] - s["t"] - s["hr"]
        s["avg"] = s["h"] / s["ab"] if s["ab"] > 0 else 0.0
        s["obp_raw"] = (
            (s["h"] + s["bb"]) / (s["ab"] + s["bb"]) if (s["ab"] + s["bb"]) > 0 else 0.0
        )
        s["games_played"] = sum(s["team_games"].values())
        s["primary_team"] = (
            max(s["team_games"], key=s["team_games"].get) if s["team_games"] else 0
        )
        s["primary_pos"] = _primary_position(s["position_freq"])
        del s["position_freq"]
        del s["team_games"]
        del s["_seen_games"]
    return agg


def _primary_position(position_freq: dict[str, int]) -> str:
    if not position_freq:
        return "DH"
    for pos in ("SS", "CF", "2B", "3B", "C", "RF", "LF", "OF", "1B"):
        if pos in position_freq:
            return pos
    if "DH" in position_freq:
        return "DH"
    return max(position_freq, key=position_freq.get)


# ---------------------------------------------------------------------------
# Supplemental stats (HBP, SF, IBB, SB, CS)
# ---------------------------------------------------------------------------


def fetch_supplemental_stats(
    player_ids: list[int],
    season: int,
    client=None,
    cache_dir: str | None = "data/cache/war_supplement",
    use_cache: bool = True,
) -> dict[int, dict[str, Any]]:
    from mlb_ml_lab import MlbClient

    cache_path = None
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"season_stats_{season}.json")
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    close_client = client is None
    if client is None:
        client = MlbClient()
    result: dict[int, dict[str, Any]] = {}
    try:
        for i, pid in enumerate(player_ids):
            if (i + 1) % 50 == 0:
                logger.info(
                    "Fetched %d/%d player stats for %d", i + 1, len(player_ids), season
                )
            try:
                stats = client.get_player_season_stats(pid, season, group="hitting")
                if not stats:
                    continue
                result[pid] = {
                    "hbp": int(stats.get("hitByPitch", 0)),
                    "sf": int(stats.get("sacFlies", 0)),
                    "ibb": int(stats.get("intentionalWalks", 0)),
                    "sb": int(stats.get("stolenBases", 0)),
                    "cs": int(stats.get("caughtStealing", 0)),
                }
            except Exception:
                continue
    finally:
        if close_client:
            client.close()
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in result.items()}, f)
    return result


# ---------------------------------------------------------------------------
# wOBA computation
# ---------------------------------------------------------------------------


def compute_woba(
    h: int,
    d: int,
    t: int,
    hr: int,
    bb: int,
    hbp: int,
    ab: int,
    sf: int,
    season: int,
) -> float | None:
    denom = ab + bb + sf + hbp
    if denom <= 0:
        return None
    p = _woba_params(season)
    num = (
        p["wBB"] * bb
        + p["wHBP"] * hbp
        + p["w1B"] * (h - d - t - hr)
        + p["w2B"] * d
        + p["w3B"] * t
        + p["wHR"] * hr
    )
    return num / denom


def compute_league_woba(totals: dict[str, int], season: int) -> float | None:
    return compute_woba(
        h=totals["h"],
        d=totals["d"],
        t=totals["t"],
        hr=totals["hr"],
        bb=totals["bb"],
        hbp=totals.get("hbp", 0),
        ab=totals["ab"],
        sf=totals.get("sf", 0),
        season=season,
    )


def compute_wraa(player_woba: float, lg_woba: float, scale: float, pa: int) -> float:
    if pa <= 0 or scale <= 0:
        return 0.0
    return ((player_woba - lg_woba) / scale) * pa


def park_adjust_wraa(raw_wraa: float, park_factor: float) -> float:
    if park_factor <= 0:
        return raw_wraa
    return raw_wraa / park_factor


# ---------------------------------------------------------------------------
# Baserunning / positional / replacement
# ---------------------------------------------------------------------------


def compute_baserunning_runs(sb: int, cs: int) -> float:
    return round(sb * 0.20 - cs * 0.40, 1)


def compute_positional_runs(
    primary_pos: str, pa: int, league_pa_per_game: float = 4.4
) -> float:
    adj_per_162 = POSITIONAL_RUNS.get(primary_pos, 0.0)
    if adj_per_162 == 0.0:
        return 0.0
    games_equiv = pa / league_pa_per_game if league_pa_per_game > 0 else 0
    return adj_per_162 * (games_equiv / 162.0)


def compute_replacement_runs(
    pa: int, league_pa_per_game: float = 4.4, replacement_per_162: float = 20.0
) -> float:
    if league_pa_per_game <= 0:
        return 0.0
    games_equiv = pa / league_pa_per_game
    return replacement_per_162 * (games_equiv / 162.0)


# ---------------------------------------------------------------------------
# Full player WAR
# ---------------------------------------------------------------------------


def compute_player_war(
    player_stats: dict[str, Any],
    lg_woba: float,
    lg_woba_scale: float,
    season: int,
    park_factor: float = 1.0,
    league_pa_per_game: float = 4.4,
    supplemental: dict[str, int] | None = None,
    fielding_runs: float = 0.0,
) -> dict[str, Any]:
    sup = supplemental or {}
    hbp = sup.get("hbp", 0)
    sf = sup.get("sf", 0)
    sb = sup.get("sb", 0)
    cs = sup.get("cs", 0)
    raw_woba = compute_woba(
        h=player_stats["h"],
        d=player_stats["d"],
        t=player_stats["t"],
        hr=player_stats["hr"],
        bb=player_stats["bb"],
        hbp=hbp,
        ab=player_stats["ab"],
        sf=sf,
        season=season,
    )
    if raw_woba is None:
        return {"war": 0.0, "error": "zero PA denominator"}
    adj_woba = raw_woba / park_factor if park_factor > 0 else raw_woba
    raw_wraa = compute_wraa(raw_woba, lg_woba, lg_woba_scale, player_stats["pa"])
    batting_runs = park_adjust_wraa(raw_wraa, park_factor)
    br_runs = compute_baserunning_runs(sb, cs)
    pos_runs = compute_positional_runs(
        player_stats["primary_pos"], player_stats["pa"], league_pa_per_game
    )
    repl_runs = compute_replacement_runs(player_stats["pa"], league_pa_per_game)
    total_runs = batting_runs + br_runs + pos_runs + repl_runs + fielding_runs
    rpw = _runs_per_win(season)
    war = total_runs / rpw
    return {
        "player_id": player_stats["player_id"],
        "player_name": player_stats["player_name"],
        "season": season,
        "team_id": player_stats.get("primary_team", player_stats.get("team_id", 0)),
        "primary_pos": player_stats["primary_pos"],
        "g": player_stats["games_played"],
        "pa": player_stats["pa"],
        "ab": player_stats["ab"],
        "h": player_stats["h"],
        "d": player_stats["d"],
        "t": player_stats["t"],
        "hr": player_stats["hr"],
        "bb": player_stats["bb"],
        "so": player_stats["so"],
        "hbp": hbp,
        "sf": sf,
        "sb": sb,
        "cs": cs,
        "avg": round(player_stats["avg"], 3),
        "raw_woba": round(raw_woba, 4),
        "adj_woba": round(adj_woba, 4),
        "lg_woba": round(lg_woba, 4),
        "park_factor": round(park_factor, 4),
        "raw_wraa": round(raw_wraa, 1),
        "wraa": round(batting_runs, 1),
        "br_runs": round(br_runs, 1),
        "pos_adj": round(pos_runs, 1),
        "repl_runs": round(repl_runs, 1),
        "fielding_runs": round(fielding_runs, 1),
        "total_runs": round(total_runs, 1),
        "rpw": rpw,
        "war": round(war, 2),
    }


# ---------------------------------------------------------------------------
# Batch WAR computation
# ---------------------------------------------------------------------------


def _league_totals_from_agg(
    raw_agg: dict[tuple[int, int], dict[str, Any]],
    supplemental: dict[tuple[int, int], dict[str, int]],
) -> dict[int, dict[str, int]]:
    by_season: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for v in raw_agg.values():
        by_season[v["season"]].append(v)
    league_totals: dict[int, dict[str, int]] = {}
    for season, players in by_season.items():
        lt: dict[str, int] = defaultdict(int)
        for p in players:
            sup = supplemental.get((p["player_id"], season), {})
            lt["ab"] += p["ab"]
            lt["h"] += p["h"]
            lt["d"] += p["d"]
            lt["t"] += p["t"]
            lt["hr"] += p["hr"]
            lt["bb"] += p["bb"]
            lt["hbp"] += sup.get("hbp", 0)
            lt["sf"] += sup.get("sf", 0)
        league_totals[season] = dict(lt)
    return league_totals


def compute_all_war(
    game_logs: list[dict[str, Any]],
    fielding_runs_by_season: dict[int, dict[int, float]] | None = None,
    seasons: list[int] | None = None,
    min_pa: int = 200,
    fetch_supplemental: bool = True,
    supplemental_cache_dir: str | None = "data/cache/war_supplement",
) -> list[dict[str, Any]]:
    agg = aggregate_player_seasons(game_logs)
    if seasons:
        agg = {k: v for k, v in agg.items() if v["season"] in seasons}
    agg = {k: v for k, v in agg.items() if v["pa"] >= min_pa}
    by_season: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for v in agg.values():
        by_season[v["season"]].append(v)

    supplemental: dict[tuple[int, int], dict[str, int]] = {}
    if fetch_supplemental:
        from mlb_ml_lab import MlbClient

        client = MlbClient()
        try:
            for season in sorted(by_season):
                pids = [s["player_id"] for s in by_season[season]]
                season_sup = fetch_supplemental_stats(
                    pids, season, client, cache_dir=supplemental_cache_dir
                )
                for pid, stats in season_sup.items():
                    supplemental[(pid, season)] = stats
        finally:
            client.close()

    raw_agg = aggregate_player_seasons(game_logs)
    if seasons:
        raw_agg = {k: v for k, v in raw_agg.items() if v["season"] in seasons}
    league_totals = _league_totals_from_agg(raw_agg, supplemental)

    fg_fielding = fielding_runs_by_season or {}
    all_war: list[dict[str, Any]] = []
    for season in sorted(by_season):
        p = _woba_params(season)
        lg_woba = compute_league_woba(league_totals[season], season) or 0.320
        fg = fg_fielding.get(season, {})
        for player_stats in by_season[season]:
            pid = player_stats["player_id"]
            sup = supplemental.get((pid, season), {})
            tid = player_stats.get("primary_team", player_stats.get("team_id", 0))
            pf_val = get_park_factor(tid, season)
            fielding = fg.get(pid, 0.0)
            result = compute_player_war(
                player_stats,
                lg_woba,
                p["scale"],
                season,
                pf_val,
                4.4,
                sup,
                fielding,
            )
            all_war.append(result)
    all_war.sort(key=lambda x: x["war"], reverse=True)
    return all_war


# ---------------------------------------------------------------------------
# FanGraphs comparison
# ---------------------------------------------------------------------------

FANGRAPHS_API_URL = (
    "https://www.fangraphs.com/api/leaders/major-league/data?"
    "pos=all&stats=bat&lg=all&qual=y&type=8&season={season}"
    "&month=0&season1={season}&ind=0&page=1_1000"
)


def fetch_fangraphs_war(
    season: int, cache_dir: str = "data/cache/war_comparison", use_cache: bool = True
) -> list[dict[str, Any]]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"fangraphs_war_{season}.json")
    if use_cache and os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    try:
        import httpx
        import re

        resp = httpx.get(
            FANGRAPHS_API_URL.format(season=season), timeout=30.0, follow_redirects=True
        )
        resp.raise_for_status()
        data = resp.json()
        raw_rows = data.get("data", [])
        rows: list[dict[str, Any]] = []
        for r in raw_rows:
            name_html = r.get("Name", "")
            name_match = re.search(r">([^<]+)<", name_html)
            name = name_match.group(1).strip() if name_match else name_html.strip()
            team_html = r.get("Team", "")
            team_match = re.search(r">([^<]+)<", team_html)
            team = team_match.group(1).strip() if team_match else team_html.strip()
            rows.append(
                {
                    "player_id": _int_or(r.get("xMLBAMID"), 0),
                    "name": name,
                    "team": team,
                    "g": _int_or(r.get("G"), 0),
                    "pa": _int_or(r.get("PA"), 0),
                    "woba": _float_or(r.get("wOBA"), 0.0),
                    "wraa": _float_or(r.get("wRAA"), 0.0),
                    "war": _float_or(r.get("WAR"), 0.0),
                    "off": _float_or(r.get("Offense"), 0.0),
                    "def": _float_or(r.get("Defense"), 0.0),
                    "batting": _float_or(r.get("Batting"), 0.0),
                    "fielding": _float_or(r.get("Fielding"), 0.0),
                    "baserunning": _float_or(r.get("BaseRunning"), 0.0),
                    "positional": _float_or(r.get("Positional"), 0.0),
                    "replacement": _float_or(r.get("Replacement"), 0.0),
                    "wRAA": _float_or(r.get("wRAA"), 0.0),
                }
            )
        if use_cache:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
        return rows
    except Exception as e:
        logger.warning("Failed to fetch FG WAR for %d: %s", season, e)
        return []


def fetch_all_fg_fielding(
    season: int, cache_dir: str = "data/cache/war_comparison", use_cache: bool = True
) -> dict[int, float]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"fangraphs_all_{season}.json")
    if use_cache and os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    try:
        import httpx

        url = (
            "https://www.fangraphs.com/api/leaders/major-league/data?"
            "pos=all&stats=bat&lg=all&qual=n&type=8&season={season}"
            "&month=0&season1={season}&ind=0&pageitems=5000&pagenum=1"
        ).format(season=season)
        resp = httpx.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
        raw_rows = data.get("data", [])
        result: dict[int, float] = {}
        for r in raw_rows:
            pid = _int_or(r.get("xMLBAMID"), 0)
            if not pid:
                continue
            fielding = _float_or(r.get("Fielding"), None)
            if fielding is None:
                continue
            result[pid] = fielding
        if use_cache:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in result.items()}, f, indent=2)
        return result
    except Exception as e:
        logger.warning("Failed to fetch FG fielding for %d: %s", season, e)
        return {}


def _int_or(val, default: int = 0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _float_or(val, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def match_players(
    our_war: list[dict[str, Any]], fg_war: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fg_by_id: dict[int, dict[str, Any]] = {}
    fg_by_name: dict[str, dict[str, Any]] = {}
    for r in fg_war:
        pid = r.get("player_id", 0)
        if pid:
            fg_by_id[pid] = r
        fg_by_name[r["name"].lower()] = r
    matches: list[dict[str, Any]] = []
    for r in our_war:
        pid = r.get("player_id", 0)
        fg = fg_by_id.get(pid)
        if fg is None:
            name = r["player_name"].strip().lower()
            fg = fg_by_name.get(name)
            if fg is None:
                for fg_name, fg_row in fg_by_name.items():
                    parts = name.split()
                    if (
                        len(parts) >= 2
                        and parts[-1] in fg_name
                        and parts[0][0] in fg_name
                    ):
                        fg = fg_row
                        break
        if fg:
            rpw = _runs_per_win(r["season"])
            our_fielding = r.get("fielding_runs", 0.0)
            our_no_def = r["war"] - (our_fielding / rpw)
            fg_no_def = (
                fg["batting"] + fg["baserunning"] + fg["positional"] + fg["replacement"]
            ) / rpw
            matches.append(
                {
                    "player_id": r["player_id"],
                    "player_name": r["player_name"],
                    "season": r["season"],
                    "pa": r["pa"],
                    "g": r["g"],
                    "our_war": r["war"],
                    "fg_war": fg["war"],
                    "our_war_no_def": round(our_no_def, 2),
                    "fg_war_no_def": round(fg_no_def, 2),
                    "our_wraa": r.get("wraa", 0.0),
                    "fg_batting": fg["batting"],
                    "our_br": r.get("br_runs", 0.0),
                    "fg_br": fg["baserunning"],
                    "our_fielding": our_fielding,
                    "fg_fielding": fg["fielding"],
                    "our_pos_adj": r.get("pos_adj", 0.0),
                    "fg_positional": fg["positional"],
                    "our_repl": r.get("repl_runs", 0.0),
                    "fg_replacement": fg["replacement"],
                    "our_woba": r.get("raw_woba", 0.0),
                    "fg_woba": fg.get("woba", 0.0),
                }
            )
    return matches


def comparison_metrics(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if len(matches) < 2:
        return {"error": "not enough matches", "n": len(matches)}
    our = np.array([m["our_war"] for m in matches])
    fg = np.array([m["fg_war"] for m in matches])
    our_nd = np.array([m["our_war_no_def"] for m in matches])
    fg_nd = np.array([m["fg_war_no_def"] for m in matches])
    comps = _component_stats(matches)
    return {
        "n": len(matches),
        "correlation": float(np.corrcoef(our, fg)[0, 1]),
        "corr_no_fielding": float(np.corrcoef(our_nd, fg_nd)[0, 1])
        if len(matches) > 1
        else 0.0,
        "mae": float(np.mean(np.abs(our - fg))),
        "mae_no_fielding": float(np.mean(np.abs(our_nd - fg_nd))),
        "bias_our_minus_fg": float(np.mean(our - fg)),
        "bias_no_fielding": float(np.mean(our_nd - fg_nd)),
        "our_mean_war": float(np.mean(our)),
        "fg_mean_war": float(np.mean(fg)),
        "components": comps,
    }


def _component_stats(matches: list[dict[str, Any]]) -> dict[str, Any]:
    components = {
        "wRAA/Batting": ("our_wraa", "fg_batting"),
        "Baserunning": ("our_br", "fg_br"),
        "Fielding": ("our_fielding", "fg_fielding"),
        "Positional Adj": ("our_pos_adj", "fg_positional"),
        "Replacement": ("our_repl", "fg_replacement"),
        "WAR (total)": ("our_war", "fg_war"),
        "WAR (no fielding)": ("our_war_no_def", "fg_war_no_def"),
    }
    stats = {}
    for label, (our_key, fg_key) in components.items():
        our = np.array([m[our_key] for m in matches])
        fg = np.array([m[fg_key] for m in matches])
        if len(matches) > 1:
            corr = float(np.corrcoef(our, fg)[0, 1])
        else:
            corr = 0.0
        stats[label] = {
            "correlation": round(corr, 4),
            "bias_our_minus_fg": round(float(np.mean(our - fg)), 3),
            "mae": round(float(np.mean(np.abs(our - fg))), 3),
            "our_mean": round(float(np.mean(our)), 2),
            "fg_mean": round(float(np.mean(fg)), 2),
        }
    return stats


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_war_table(results: list[dict[str, Any]], top_n: int = 20) -> None:
    print(
        f"\n{'Player':<22} {'Pos':>4} {'G':>4} {'PA':>5} "
        f"{'wOBA':>6} {'wRAA':>6} {'BR':>5} {'PosAdj':>7} "
        f"{'Repl':>5} {'WAR':>5}"
    )
    print("-" * 75)
    for r in results[:top_n]:
        print(
            f"{r['player_name']:<22} {r['primary_pos']:>4} "
            f"{r['g']:>4} {r['pa']:>5} "
            f"{r['raw_woba'] or 0:>6.3f} "
            f"{r['wraa']:>6.1f} {r['br_runs']:>5.1f} "
            f"{r['pos_adj']:>7.1f} {r['repl_runs']:>5.1f} "
            f"{r['war']:>5.2f}"
        )


def print_comparison_table(matches: list[dict[str, Any]], top_n: int = 10) -> None:
    print(
        f"\n{'Player':<22} {'PA':>5} {'Our WAR':>8} "
        f"{'FG WAR':>8} {'Diff':>6} {'NoDef D':>7}"
    )
    print("-" * 62)
    matches_sorted = sorted(
        matches, key=lambda m: abs(m["our_war"] - m["fg_war"]), reverse=True
    )
    for m in matches_sorted[:top_n]:
        diff = m["our_war"] - m["fg_war"]
        nd_diff = m["our_war_no_def"] - m["fg_war_no_def"]
        print(
            f"{m['player_name']:<22} {m['pa']:>5} "
            f"{m['our_war']:>8.2f} {m['fg_war']:>8.2f} "
            f"{diff:>+6.2f} {nd_diff:>+7.2f}"
        )


def print_component_comparison(matches: list[dict[str, Any]]) -> None:
    comps = {
        "wRAA/Batting": ("our_wraa", "fg_batting"),
        "Baserunning": ("our_br", "fg_br"),
        "Fielding": ("our_fielding", "fg_fielding"),
        "Pos. Adj.": ("our_pos_adj", "fg_positional"),
        "Replacement": ("our_repl", "fg_replacement"),
    }
    print(f"\n{'Component':<18} {'Our Mean':>9} {'FG Mean':>9} {'Bias':>7} {'Corr':>7}")
    print("-" * 55)
    for label, (our_k, fg_k) in comps.items():
        our_vals = np.array([m[our_k] for m in matches])
        fg_vals = np.array([m[fg_k] for m in matches])
        bias = np.mean(our_vals - fg_vals)
        corr = np.corrcoef(our_vals, fg_vals)[0, 1] if len(matches) > 1 else 0
        print(
            f"{label:<18} {np.mean(our_vals):>9.2f} {np.mean(fg_vals):>9.2f} "
            f"{bias:>+7.2f} {corr:>7.3f}"
        )
    our_war = np.array([m["our_war"] for m in matches])
    fg_war = np.array([m["fg_war"] for m in matches])
    print(
        f"{'WAR (total)':<18} {np.mean(our_war):>9.2f} {np.mean(fg_war):>9.2f} "
        f"{np.mean(our_war - fg_war):>+7.2f} {np.corrcoef(our_war, fg_war)[0, 1]:>7.3f}"
    )
    our_nd = np.array([m["our_war_no_def"] for m in matches])
    fg_nd = np.array([m["fg_war_no_def"] for m in matches])
    print(
        f"{'WAR (no field)':<18} {np.mean(our_nd):>9.2f} {np.mean(fg_nd):>9.2f} "
        f"{np.mean(our_nd - fg_nd):>+7.2f} {np.corrcoef(our_nd, fg_nd)[0, 1]:>7.3f}"
    )


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n{'=' * 55}")
    print(f"  WAR COMPARISON ({metrics['n']} matched players)")
    print(f"{'=' * 55}")
    print("  Total WAR:")
    print(f"    Correlation:          {metrics.get('correlation', 0):.4f}")
    print(f"    MAE:                  {metrics.get('mae', 0):.3f}")
    print(f"    Bias (our - FG):      {metrics.get('bias_our_minus_fg', 0):+.3f}")
    print(f"    Our mean WAR:         {metrics.get('our_mean_war', 0):.2f}")
    print(f"    FG mean WAR:          {metrics.get('fg_mean_war', 0):.2f}")
    print("  Excluding fielding:")
    print(f"    Corr:                 {metrics.get('corr_no_fielding', 0):.4f}")
    print(f"    MAE:                  {metrics.get('mae_no_fielding', 0):.3f}")
    print(f"    Bias:                 {metrics.get('bias_no_fielding', 0):+.3f}")
