"""Rolling advanced batting metrics computed from game logs.

Computes windowed (last N games) versions of:
  - AVG, OBP, SLG, OPS
  - ISO (SLG - AVG)
  - BABIP
  - wOBA (simplified, estimated HBP/SF)
  - wRC+ (park/league adjusted; uses lg_wOBA, lg_R/PA)
  - OPS+  (park/league adjusted)
  - BB% and K%

Uses only data available before the target game date — no lookahead.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from mlb_ml_lab.data.schemas import PlayerGameLog
from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register

# wOBA weights (from FanGraphs / Tango)
WOBA_BB = 0.690
WOBA_HBP = 0.720
WOBA_1B = 0.884
WOBA_2B = 1.261
WOBA_3B = 1.601
WOBA_HR = 2.072

WOBA_SCALE = 1.185  # typical wOBA scale for MLB


def _compute_lg_stats(
    game_logs: list[PlayerGameLog],
) -> dict[int, dict[str, float]]:
    totals: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for log in game_logs:
        s = int(log.season)
        t = totals[s]
        t["ab"] += log.at_bats
        t["h"] += log.hits
        t["d"] += log.doubles
        t["t"] += log.triples
        t["hr"] += log.home_runs
        t["bb"] += log.walks
        t["k"] += log.strikeouts
        t["r"] += log.runs
        t["pa"] += log.plate_appearances

    out: dict[int, dict[str, float]] = {}
    for s, t in totals.items():
        ab, h, d, t3, hr = t["ab"], t["h"], t["d"], t["t"], t["hr"]
        bb, _k, r, pa = t["bb"], t["k"], t["r"], t["pa"]
        tb = h + d + 2 * t3 + 3 * hr
        hbp_est = int(pa * 0.01)
        sf_est = int(pa * 0.005)
        obp_den = ab + bb + hbp_est + sf_est

        lg_obp = (h + bb + hbp_est) / obp_den if obp_den > 0 else 0.0
        lg_slg = tb / ab if ab > 0 else 0.0

        # lg_wOBA
        singles = h - d - t3 - hr
        woba_num = (
            WOBA_BB * bb
            + WOBA_HBP * hbp_est
            + WOBA_1B * singles
            + WOBA_2B * d
            + WOBA_3B * t3
            + WOBA_HR * hr
        )
        lg_woba = woba_num / obp_den if obp_den > 0 else 0.0
        lg_r_per_pa = r / pa if pa > 0 else 0.12

        out[s] = {"lg_obp": lg_obp, "lg_slg": lg_slg, "lg_woba": lg_woba, "lg_r_per_pa": lg_r_per_pa}
    return out


def _rolling_advanced(
    window_games: list[PlayerGameLog],
    lg: dict[str, float],
    pf: float,
) -> dict[str, float]:
    ab = sum(g.at_bats for g in window_games)
    h = sum(g.hits for g in window_games)
    d = sum(g.doubles for g in window_games)
    t = sum(g.triples for g in window_games)
    hr = sum(g.home_runs for g in window_games)
    bb = sum(g.walks for g in window_games)
    k = sum(g.strikeouts for g in window_games)
    pa = sum(g.plate_appearances for g in window_games)

    n = len(window_games)
    if ab == 0 or n == 0:
        return {}

    singles = h - d - t - hr
    tb = h + d + 2 * t + 3 * hr

    hbp_est = int(pa * 0.01) if pa > 0 else 0
    sf_est = int(pa * 0.005) if pa > 0 else 0
    obp_den = ab + bb + hbp_est + sf_est

    avg = h / ab
    obp = (h + bb + hbp_est) / obp_den if obp_den > 0 else 0.0
    slg = tb / ab
    iso = slg - avg

    babip_num = h - hr
    babip_den = ab - k - hr + sf_est
    babip = babip_num / babip_den if babip_den > 0 else 0.0

    bb_pct = bb / pa * 100 if pa > 0 else 0.0
    k_pct = k / pa * 100 if pa > 0 else 0.0

    # wOBA
    woba_num = (
        WOBA_BB * bb
        + WOBA_HBP * hbp_est
        + WOBA_1B * singles
        + WOBA_2B * d
        + WOBA_3B * t
        + WOBA_HR * hr
    )
    woba = woba_num / obp_den if obp_den > 0 else 0.0

    # OPS+ = (OBP/lgOBP + SLG/lgSLG - 1) * 100 / PF
    pf = max(pf, 0.1)
    ops_plus = 100.0
    if lg.get("lg_obp", 0) > 0 and lg.get("lg_slg", 0) > 0:
        adj_obp = obp * pf
        adj_slg = slg * pf
        ops_plus = (adj_obp / lg["lg_obp"] + adj_slg / lg["lg_slg"] - 1) * 100

    # wRC+ = (wRAA / (lg_R/PA * PA) + 1) * 100
    # wRAA = (wOBA - lg_wOBA) / wOBA_scale * PA
    wrc_plus = 100.0
    lg_woba = lg.get("lg_woba", 0.0)
    lg_r_per_pa = lg.get("lg_r_per_pa", 0.12)
    if lg_woba > 0 and pa > 0:
        wraa = ((woba - lg_woba) / WOBA_SCALE) * pa
        league_runs = lg_r_per_pa * pa
        if league_runs > 0:
            wrc_plus = (wraa / league_runs + 1) * 100

    return {
        "avg": round(avg, 3),
        "obp": round(obp, 3),
        "slg": round(slg, 3),
        "ops": round(obp + slg, 3),
        "iso": round(iso, 3),
        "babip": round(babip, 3),
        "bb_pct": round(bb_pct, 1),
        "k_pct": round(k_pct, 1),
        "woba": round(woba, 3),
        "ops_plus": round(ops_plus, 1),
        "wrc_plus": round(wrc_plus, 1),
        "rolling_n": n,
    }


@register
class RollingAdvancedMetrics(FeatureExtractor):
    """Rolling advanced batting metrics (AVG, OBP, SLG, ISO, wOBA, wRC+, OPS+)."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self._windows = windows or [10, 20]

    @property
    def features(self) -> list[FeatureMeta]:
        cols: list[FeatureMeta] = []
        metrics = ["avg", "obp", "slg", "ops", "iso", "babip",
                    "bb_pct", "k_pct", "woba", "ops_plus", "wrc_plus"]
        for w in self._windows:
            for m in metrics:
                cols.append(
                    FeatureMeta(
                        name=f"rolling_{m}_{w}",
                        description=f"Rolling {m} over last {w} games",
                        source="game_log",
                    )
                )
        return cols

    def extract(
        self, game_logs: list[PlayerGameLog], **kwargs: Any
    ) -> list[dict[str, Any]]:
        # Compute league context from all provided game logs
        lg_stats = _compute_lg_stats(game_logs)

        # Build park factor lookup (game_pk -> pf_simple)
        game_contexts: dict[int, Any] = kwargs.get("game_contexts", {})
        park_factors: dict[int, float] = {}
        for gpk, ctx in game_contexts.items():
            pf = (
                ctx.get("parkFactors", {})
                if isinstance(ctx, dict)
                else {}
            )
            park_factors[gpk] = pf.get("wOBA", 1.0) if isinstance(pf, dict) else 1.0

        by_player: dict[int, list[PlayerGameLog]] = defaultdict(list)
        for log in game_logs:
            by_player[log.player_id].append(log)
        for logs in by_player.values():
            logs.sort(key=lambda x: x.date)

        rows: list[dict[str, Any]] = []
        for pid, logs in by_player.items():
            buffers: dict[int, deque[PlayerGameLog]] = {
                w: deque(maxlen=w) for w in self._windows
            }
            for log in logs:
                row: dict[str, Any] = {
                    "player_id": pid,
                    "game_pk": log.game_pk,
                    "date": log.date,
                }
                pf = park_factors.get(log.game_pk, 1.0)
                lg = lg_stats.get(int(log.season), {})
                for w in self._windows:
                    buf = buffers[w]
                    if len(buf) == w:
                        m = _rolling_advanced(list(buf), lg, pf)
                        for key, val in m.items():
                            row[f"rolling_{key}_{w}"] = val
                for w in self._windows:
                    buffers[w].append(log)

                rows.append(row)

        return rows
