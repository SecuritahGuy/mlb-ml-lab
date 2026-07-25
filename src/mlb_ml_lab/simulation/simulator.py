from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mlb_ml_lab.simulation.outcomes import OUTCOME_CLASSES, blend_outcomes

_INDEX_TO_CLASS = dict(enumerate(OUTCOME_CLASSES))
_CLASS_TO_INDEX = {v: k for k, v in _INDEX_TO_CLASS.items()}

DEFAULT_RUNS_PER_OUTCOME = {
    "single": 0.38,
    "double": 0.65,
    "triple": 0.96,
    "home_run": 1.44,
    "walk": 0.17,
    "strikeout": 0.0,
    "other": 0.02,
}

DEFAULT_TOTAL_RUNS_PER_OUTCOME = {
    "single": 0.48,
    "double": 0.78,
    "triple": 1.08,
    "home_run": 1.44,
    "walk": 0.22,
    "strikeout": 0.04,
    "other": 0.06,
}

AVG_PAS_PER_GAME = 78


def compute_runs_per_outcome(
    pas: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    direct: dict[str, list[float]] = defaultdict(list)
    total: dict[str, list[float]] = defaultdict(list)
    sorted_pas = sorted(pas, key=lambda p: (p["game_pk"], p["at_bat_index"]))
    prev_home: dict[int, int] = defaultdict(int)
    prev_away: dict[int, int] = defaultdict(int)
    game_inning_pas: dict[str, list[tuple[int, str, float]]] = defaultdict(list)

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

    for inning_key, pa_list in game_inning_pas.items():
        remaining = sum(r for _, r in pa_list)
        for et, runs_scored in pa_list:
            total[et].append(float(remaining))
            remaining -= runs_scored

    def _avg(runs_list: list[float]) -> float:
        return round(float(np.mean(runs_list)), 4) if runs_list else 0.0

    return (
        {et: _avg(direct[et]) for et in OUTCOME_CLASSES},
        {et: _avg(total[et]) for et in OUTCOME_CLASSES},
    )


def expected_game_runs(
    batters: list[int],
    pitchers: list[int],
    league_avg: dict[str, float],
    batter_outcomes: dict[int, dict[str, float]],
    pitcher_outcomes: dict[int, dict[str, float]],
    runs_per_outcome: dict[str, float] | None = None,
) -> float:
    if runs_per_outcome is None:
        runs_per_outcome = DEFAULT_TOTAL_RUNS_PER_OUTCOME
    total_runs = 0.0
    for inning in range(9):
        pitcher_id = pitchers[min(inning, len(pitchers) - 1)]
        pitcher_dist = pitcher_outcomes.get(pitcher_id, league_avg)
        outs = 0
        batter_idx = inning
        while outs < 3:
            batter_id = batters[batter_idx % len(batters)]
            batter_idx += 1
            batter_dist = batter_outcomes.get(batter_id, league_avg)
            blended = blend_outcomes(batter_dist, pitcher_dist, league_avg)
            for outcome, prob in blended.items():
                if prob > 0 and outcome in runs_per_outcome:
                    total_runs += prob * runs_per_outcome[outcome]
            outs += 1
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
    home_runs = expected_game_runs(
        home_batters,
        home_pitchers,
        league_avg,
        batter_outcomes,
        pitcher_outcomes,
        runs_per_outcome,
    )
    away_runs = expected_game_runs(
        away_batters,
        away_pitchers,
        league_avg,
        batter_outcomes,
        pitcher_outcomes,
        runs_per_outcome,
    )
    return {
        "home_runs": home_runs,
        "away_runs": away_runs,
        "total_runs": home_runs + away_runs,
    }


class MonteCarloSimulator:
    """Game simulator using Monte Carlo sampling.

    Simulates 9 innings per team, sampling each PA outcome from
    a model's predicted probability vector, tracking outs via
    empirical out rates, and accumulating runs.
    """

    def __init__(
        self,
        model: Any,
        rolling_state: Any,
        runs_per_outcome: np.ndarray | None = None,
        out_probs: np.ndarray | None = None,
        n_simulations: int = 1000,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.rs = rolling_state
        self.n = n_simulations
        self.rng = np.random.default_rng(rng)
        self._n_classes = 7

        if runs_per_outcome is not None:
            self.runs = np.asarray(runs_per_outcome, dtype=np.float64)
        else:
            self.runs = np.array(
                [0.38, 0.65, 0.96, 1.44, 0.17, 0.0, 0.02], dtype=np.float64
            )

        if out_probs is not None:
            self.out_probs = np.asarray(out_probs, dtype=np.float64)
        else:
            self.out_probs = np.array(
                [0.0164, 0.0138, 0.0019, 0.0, 0.0004, 0.999, 0.8848],
                dtype=np.float64,
            )

    def simulate_game(
        self,
        home_order: list[int],
        away_order: list[int],
        home_pitcher: int,
        away_pitcher: int,
        game_pk: int,
    ) -> dict[str, Any]:
        home_probas = self._batch_probas(home_order, home_pitcher, game_pk)
        away_probas = self._batch_probas(away_order, away_pitcher, game_pk)
        away = self._sim_team_fast(
            away_order, away_pitcher, away_probas, "top", game_pk
        )
        home = self._sim_team_fast(
            home_order, home_pitcher, home_probas, "bottom", game_pk
        )
        total = away + home

        def _stats(arr: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        return {
            "home_runs": home,
            "away_runs": away,
            "total_runs": total,
            "home": _stats(home),
            "away": _stats(away),
            "total": _stats(total),
        }

    def _batch_probas(
        self, order: list[int], pitcher_id: int, game_pk: int
    ) -> np.ndarray:
        fvs = []
        for bid in order:
            fv = self.rs.feature_vector(
                bid,
                pitcher_id,
                include_platoon=True,
                include_game_context=True,
                include_game_log=True,
                include_game_log_rates=True,
                game_pk=game_pk,
            )
            fvs.append(fv)
        batch = np.array(fvs, dtype=np.float64)
        return self.model.predict_proba(batch)

    def _sim_team_fast(
        self,
        order: list[int],
        _pitcher_id: int,
        probas: np.ndarray,
        _half_inning: str,
        _game_pk: int,
    ) -> np.ndarray:
        n_ordered = len(order)
        results = np.empty(self.n, dtype=np.float64)
        for sim in range(self.n):
            total = 0.0
            for inning in range(9):
                outs = 0
                bi = inning
                while outs < 3:
                    proba = probas[bi % n_ordered]
                    outcome = int(self.rng.multinomial(1, proba).argmax())
                    total += self.runs[outcome]
                    if self.rng.random() < self.out_probs[outcome]:
                        outs += 1
                    bi += 1
            results[sim] = total
        return results
