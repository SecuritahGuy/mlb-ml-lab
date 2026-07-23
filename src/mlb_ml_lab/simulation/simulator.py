from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from mlb_ml_lab.simulation.outcomes import (
    OUTCOME_CLASSES,
    blend_outcomes,
    load_outcome_distributions,
    load_pbp_dataset,
)

# Average runs scored directly on each play type (from PBP data)
DEFAULT_RUNS_PER_OUTCOME = {
    "single": 0.38,
    "double": 0.65,
    "triple": 0.96,
    "home_run": 1.44,
    "walk": 0.17,
    "strikeout": 0.0,
    "other": 0.02,
}

# Average total runs per PA (runs scored on play + subsequent runs)
DEFAULT_TOTAL_RUNS_PER_OUTCOME = {
    "single": 0.48,
    "double": 0.78,
    "triple": 1.08,
    "home_run": 1.44,
    "walk": 0.22,
    "strikeout": 0.04,
    "other": 0.06,
}

AVG_PAS_PER_GAME = 78  # ~39 per team per 9-inning game


def compute_runs_per_outcome(
    pas: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute average runs per PA by outcome type from PBP data.

    Returns (direct_runs, total_runs):
        direct_runs: runs scored on the play itself.
        total_runs: runs scored from this PA until end of inning.
    """
    direct: dict[str, list[float]] = defaultdict(list)
    total: dict[str, list[float]] = defaultdict(list)

    # Sort by game, then at_bat_index
    sorted_pas = sorted(pas, key=lambda p: (p["game_pk"], p["at_bat_index"]))

    prev_home: dict[int, int] = defaultdict(int)
    prev_away: dict[int, int] = defaultdict(int)

    # Track cumulative runs per PA per game-inning
    game_inning_pas: dict[str, list[tuple[int, str, float]]] = (
        defaultdict(list)
    )

    for pa in sorted_pas:
        gpk = pa["game_pk"]
        et = pa["event_type"]
        inning_key = f"{gpk}_{pa['inning']}_{pa['half_inning']}"

        home = pa["home_score"]
        away = pa["away_score"]
        runs_this_pa = max(0, home - prev_home[gpk]) + max(0, away - prev_away[gpk])
        prev_home[gpk] = home
        prev_away[gpk] = away

        direct[et].append(float(runs_this_pa))
        game_inning_pas[inning_key].append((et, runs_this_pa))

    # Compute total runs from each PA to end of inning
    for inning_key, pa_list in game_inning_pas.items():
        remaining = sum(r for _, r in pa_list)
        for et, runs_scored in pa_list:
            total_runs = remaining
            total[et].append(float(total_runs))
            remaining -= runs_scored

    def _avg(runs_list: list[float]) -> float:
        return round(float(np.mean(runs_list)), 4) if runs_list else 0.0

    return {et: _avg(direct[et]) for et in OUTCOME_CLASSES}, {
        et: _avg(total[et]) for et in OUTCOME_CLASSES
    }


def expected_game_runs(
    batters: list[int],
    pitchers: list[int],
    league_avg: dict[str, float],
    batter_outcomes: dict[int, dict[str, float]],
    pitcher_outcomes: dict[int, dict[str, float]],
    runs_per_outcome: dict[str, float] | None = None,
) -> float:
    """Compute expected total runs scored by one team in a game.

    Args:
        batters: Ordered list of batter IDs in the lineup (9).
        pitchers: Ordered list of pitcher IDs (one per inning, typically 1).
        league_avg: MLB-wide outcome distribution.
        batter_outcomes: Per-batter outcome distributions.
        pitcher_outcomes: Per-pitcher outcome distributions.
        runs_per_outcome: Runs contributed by each outcome type.
                          Defaults to DEFAULT_TOTAL_RUNS_PER_OUTCOME.

    Returns:
        Expected total runs scored.
    """
    if runs_per_outcome is None:
        runs_per_outcome = DEFAULT_TOTAL_RUNS_PER_OUTCOME

    total_runs = 0.0
    pa_count = 0

    for inning in range(9):
        pitcher_id = pitchers[min(inning, len(pitchers) - 1)]
        pitcher_dist = pitcher_outcomes.get(pitcher_id, league_avg)

        # ~4.3 PAs per inning on average; simulate until 3 outs
        outs = 0
        batter_idx = inning  # lineup spot changes each inning

        while outs < 3:
            batter_id = batters[batter_idx % len(batters)]
            batter_idx += 1
            pa_count += 1

            batter_dist = batter_outcomes.get(batter_id, league_avg)
            blended = blend_outcomes(batter_dist, pitcher_dist, league_avg)

            for outcome, prob in blended.items():
                if prob > 0 and outcome in runs_per_outcome:
                    total_runs += prob * runs_per_outcome[outcome]

            outs += 1  # simplify: assume 1 out per PA

    return round(total_runs, 2)


def simulate_game(
    home_batters: list[int],
    away_batters: list[int],
    home_pitchers: list[int],
    away_pitchers: list[int],
    league_avg: dict[str, float],
    batter_outcomes: dict[int, dict[str, float]],
    pitcher_outcomes: dict[int, dict[str, float]],
    runs_per_outcome: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Simulate a game between two teams.

    Returns dict with ``home_runs`` and ``away_runs``.
    """
    home_runs = expected_game_runs(
        home_batters, home_pitchers,
        league_avg, batter_outcomes, pitcher_outcomes,
        runs_per_outcome,
    )
    away_runs = expected_game_runs(
        away_batters, away_pitchers,
        league_avg, batter_outcomes, pitcher_outcomes,
        runs_per_outcome,
    )
    return {
        "home_runs": home_runs,
        "away_runs": away_runs,
        "total_runs": home_runs + away_runs,
    }
