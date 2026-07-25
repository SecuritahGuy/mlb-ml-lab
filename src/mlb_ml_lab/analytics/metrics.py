from __future__ import annotations

# pylint: disable=too-many-return-statements  # classify_player is a 16-archetype decision tree

from collections import defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Per-player rate stat helpers
# ---------------------------------------------------------------------------


def compute_obp(h: int, bb: int, hbp: int, ab: int, sf: int) -> float:
    denom = ab + bb + sf + hbp
    return (h + bb + hbp) / denom if denom > 0 else 0.0


def compute_slg(h: int, d: int, t: int, hr: int, ab: int) -> float:
    return (h + d + 2 * t + 3 * hr) / ab if ab > 0 else 0.0


def compute_iso(slg: float, avg: float) -> float:
    return slg - avg


def compute_babip(h: int, hr: int, ab: int, k: int, sf: int) -> float:
    denom = ab - k - hr + sf
    return (h - hr) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# League-level rates
# ---------------------------------------------------------------------------


def compute_league_rate_stats(
    game_logs: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    totals: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for log in game_logs:
        season = int(log["season"])
        t = totals[season]
        t["ab"] += int(log.get("at_bats", 0))
        t["h"] += int(log.get("hits", 0))
        t["d"] += int(log.get("doubles", 0))
        t["t"] += int(log.get("triples", 0))
        t["hr"] += int(log.get("home_runs", 0))
        t["bb"] += int(log.get("walks", 0))
        t["so"] += int(log.get("strikeouts", 0))
        t["r"] += int(log.get("runs", 0))
        t["pa"] += int(log.get("plate_appearances", 0))
    results: dict[int, dict[str, float]] = {}
    for season, t in totals.items():
        ab = t["ab"]
        h = t["h"]
        tb = h + t["d"] + 2 * t["t"] + 3 * t["hr"]
        pa = t["pa"]
        hbp_est = int(pa * 0.01)
        sf_est = int(pa * 0.005)
        obp_denom = ab + t["bb"] + hbp_est + sf_est
        obp_num = h + t["bb"] + hbp_est
        results[season] = {
            "lg_obp": obp_num / obp_denom if obp_denom > 0 else 0.0,
            "lg_slg": tb / ab if ab > 0 else 0.0,
            "lg_r_per_pa": t["r"] / pa if pa > 0 else 0.12,
            "lg_pa": pa,
            "lg_ab": ab,
            "lg_h": h,
        }
    return results


# ---------------------------------------------------------------------------
# Adjusted metrics
# ---------------------------------------------------------------------------


def compute_ops_plus(
    obp: float,
    slg: float,
    lg_obp: float,
    lg_slg: float,
    park_factor: float = 1.0,
) -> float:
    if lg_obp <= 0 or lg_slg <= 0:
        return 100.0
    pf = max(park_factor, 0.1)
    adj_obp = obp / pf
    adj_slg = slg / pf
    return round((adj_obp / lg_obp + adj_slg / lg_slg - 1) * 100, 1)


def compute_wrc_plus(wraa: float, pa: int, lg_r_per_pa: float) -> float:
    if pa <= 0 or lg_r_per_pa <= 0:
        return 100.0
    return round((wraa / (lg_r_per_pa * pa) + 1) * 100, 1)


def compute_woba_plus(
    raw_woba: float, lg_woba: float, park_factor: float = 1.0
) -> float:
    if lg_woba <= 0:
        return 100.0
    pf = max(park_factor, 0.1)
    return round((raw_woba / pf / lg_woba) * 100, 1)


# ---------------------------------------------------------------------------
# WAR rate stats
# ---------------------------------------------------------------------------


def compute_war_per_162(war: float, g: int) -> float:
    return round(war / g * 162, 1) if g > 0 else 0.0


def compute_war_per_600(war: float, pa: int) -> float:
    return round(war / pa * 600, 1) if pa > 0 else 0.0


# ---------------------------------------------------------------------------
# Player classification (archetype)
# ---------------------------------------------------------------------------

ISO_LOW = 0.100
ISO_MED = 0.180
ISO_HIGH = 0.300
BB_LOW = 7.0
BB_MED = 11.0
BB_HIGH = 15.0
K_LOW = 18.0
K_MED = 24.0
K_HIGH = 30.0
SB_FEW = 3
SB_SOME = 12
SB_MANY = 25


def classify_player(
    iso: float,
    bb_pct: float,
    k_pct: float,
    sb: int,
    hr_per_pa: float,
) -> str:
    elite_power = iso > ISO_HIGH
    power = iso > ISO_MED
    some_power = iso > ISO_LOW
    patient = bb_pct > BB_MED
    some_patience = bb_pct > BB_LOW
    high_contact = k_pct < K_LOW
    contact = k_pct < K_MED
    low_contact = k_pct > K_HIGH
    elite_speed = sb > SB_MANY
    speed = sb > SB_SOME
    some_speed = sb > SB_FEW
    elite_hr_rate = hr_per_pa > 0.065

    if elite_power and elite_speed:
        return "Superstar"
    if elite_power and speed:
        return "Power/Speed"
    if elite_power and patient:
        return "Elite Bat" if high_contact else "Power/Patience"
    if elite_power:
        return "Elite Power" if elite_hr_rate else "Power"
    if power and speed and high_contact:
        return "Five-Tool"
    if power and speed:
        return "Power/Speed"
    if power and high_contact and patient:
        return "Elite Bat"
    if power and high_contact:
        return "Power/Contact"
    if power and patient:
        return "Power/Patience"
    if power:
        return "Power"
    if speed and high_contact and patient:
        return "Dynamic"
    if speed and high_contact:
        return "Contact/Speed"
    if speed and some_patience:
        return "All-Around"
    if speed:
        return "Speed"
    if high_contact and patient:
        return "Professional"
    if high_contact:
        return "Contact"
    if contact and patient:
        return "Patient/Contact"
    if patient:
        return "Patient"
    if low_contact and not some_patience:
        return "Free Swinger"
    if low_contact:
        return "Three True Outcomes"
    if not some_power and not some_patience and not some_speed and contact:
        return "Light Hitting"
    if not some_power and not some_patience and not some_speed:
        return "Below Average"
    return "Balanced"


# ---------------------------------------------------------------------------
# Build complete advanced-metrics row
# ---------------------------------------------------------------------------


def compute_player_advanced_metrics(
    row: dict[str, Any],
    lg_obp: float,
    lg_slg: float,
    lg_r_per_pa: float,
    lg_woba: float,
) -> dict[str, Any]:
    h = row["h"]
    d = row["d"]
    t = row["t"]
    hr = row["hr"]
    bb = row["bb"]
    so = row["so"]
    ab = row["ab"]
    pa = row["pa"]
    hbp = row.get("hbp", 0)
    sf = row.get("sf", 0)
    sb = row.get("sb", 0)

    obp = compute_obp(h, bb, hbp, ab, sf)
    slg = compute_slg(h, d, t, hr, ab)
    avg = h / ab if ab > 0 else 0.0
    iso = compute_iso(slg, avg)
    babip = compute_babip(h, hr, ab, so, sf)
    bb_pct = bb / pa * 100 if pa > 0 else 0.0
    k_pct = so / pa * 100 if pa > 0 else 0.0
    bb_per_k = bb / so if so > 0 else (bb if bb > 0 else 0.0)
    pf = row.get("park_factor", 1.0)
    wraa = row.get("wraa", 0.0)
    war = row.get("war", 0.0)
    g = row.get("g", 0)
    ops_plus = (
        compute_ops_plus(obp, slg, lg_obp, lg_slg, pf) if lg_obp and lg_slg else 100.0
    )
    wrc_plus = compute_wrc_plus(wraa, pa, lg_r_per_pa)
    woba_plus = compute_woba_plus(row.get("raw_woba", 0.0), lg_woba, pf)
    war162 = compute_war_per_162(war, g)
    war600 = compute_war_per_600(war, pa)
    hr_per_pa = hr / pa if pa > 0 else 0.0
    archetype = classify_player(iso, bb_pct, k_pct, sb, hr_per_pa)

    return {
        "obp": round(obp, 3),
        "slg": round(slg, 3),
        "iso": round(iso, 3),
        "babip": round(babip, 3),
        "bb_pct": round(bb_pct, 1),
        "k_pct": round(k_pct, 1),
        "bb_per_k": round(bb_per_k, 2),
        "ops_plus": ops_plus,
        "wrc_plus": wrc_plus,
        "woba_plus": woba_plus,
        "war_per_162": war162,
        "war_per_600": war600,
        "archetype": archetype,
        "hr_per_pa": round(hr_per_pa, 4),
        "avg": round(avg, 3),
    }


def add_advanced_metrics(
    war_results: list[dict[str, Any]],
    lg_stats: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    for row in war_results:
        season = row["season"]
        lg = lg_stats.get(season, {})
        metrics = compute_player_advanced_metrics(
            row,
            lg.get("lg_obp", 0.0),
            lg.get("lg_slg", 0.0),
            lg.get("lg_r_per_pa", 0.12),
            row.get("lg_woba", 0.310),
        )
        row.update(metrics)
    return war_results


# ---------------------------------------------------------------------------
# Display / tables
# ---------------------------------------------------------------------------


def print_advanced_metrics_table(
    results: list[dict[str, Any]],
    top_n: int = 30,
    sort_by: str = "war",
    metrics: list[str] | None = None,
) -> None:
    if metrics is None:
        metrics = [
            "wrc_plus",
            "ops_plus",
            "iso",
            "babip",
            "bb_pct",
            "k_pct",
            "war_per_162",
            "archetype",
        ]
    sorted_results = sorted(results, key=lambda r: r.get(sort_by, 0), reverse=True)
    display = sorted_results[:top_n]
    header = f"{'Player':<22} {'Pos':>4} {'PA':>5} {'WAR':>6}"
    fmt = "{name:<22} {pos:>4} {pa:>5} {war:>6.2f}"
    col_widths = {
        "wrc_plus": 7,
        "ops_plus": 7,
        "iso": 7,
        "babip": 7,
        "bb_pct": 6,
        "k_pct": 6,
        "war_per_162": 8,
        "war_per_600": 8,
        "archetype": 16,
    }
    for m in metrics:
        w = col_widths.get(m, 8)
        label = m.upper() if len(m) <= 8 else m.upper()[:w]
        header += f" {label:>{w}}"
        fmt += f" {{{m}:>{w}}}"
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")
    for r in display:
        name = r["player_name"]
        pos = r["primary_pos"]
        pa = r["pa"]
        war = r["war"]
        vals = {m: r.get(m, "?") for m in metrics}
        vals["name"] = name
        vals["pos"] = pos
        vals["pa"] = pa
        vals["war"] = war
        print(fmt.format(**vals))
    print(f"{'─' * len(header)}")
    print(f"  Sorted by {sort_by}, top {len(display)} of {len(results)}")


def print_archetype_summary(results: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r.get("archetype", "Unknown")] += 1
    print(
        f"\n{'Archetype':<25} {'Count':>6} {'Avg WAR':>8} {'Avg wRC+':>9} {'Avg OPS+':>9}"
    )
    print("-" * 57)
    for arch in sorted(counts, key=lambda a: -counts[a]):
        group = [r for r in results if r.get("archetype") == arch]
        avg_war = np.mean([r["war"] for r in group])
        avg_wrc = np.mean([r.get("wrc_plus", 100) for r in group])
        avg_ops = np.mean([r.get("ops_plus", 100) for r in group])
        print(
            f"{arch:<25} {counts[arch]:>6} {avg_war:>8.2f} {avg_wrc:>9.1f} {avg_ops:>9.1f}"
        )
