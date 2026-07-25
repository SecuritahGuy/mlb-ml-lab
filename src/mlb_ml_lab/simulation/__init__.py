"""Game simulation: run expectancy, PA outcome distributions, and Monte Carlo game simulation."""

from mlb_ml_lab.simulation.outcomes import (
    OUTCOME_CLASSES,
    load_pbp_dataset,
    compute_league_averages,
    compute_player_outcomes,
    blend_outcomes,
    save_outcome_distributions,
    load_outcome_distributions,
)

from mlb_ml_lab.simulation.re24 import (
    compute_re24,
    re24_to_array,
    print_re24,
    BASE_STATES,
    OUT_STATES,
    STATES,
)

from mlb_ml_lab.simulation.simulator import (
    compute_runs_per_outcome,
    expected_game_runs,
    simulate_game,
    MonteCarloSimulator,
    DEFAULT_RUNS_PER_OUTCOME,
    DEFAULT_TOTAL_RUNS_PER_OUTCOME,
)

__all__ = [
    "OUTCOME_CLASSES",
    "load_pbp_dataset",
    "compute_league_averages",
    "compute_player_outcomes",
    "blend_outcomes",
    "save_outcome_distributions",
    "load_outcome_distributions",
    "compute_re24",
    "re24_to_array",
    "print_re24",
    "BASE_STATES",
    "OUT_STATES",
    "STATES",
    "compute_runs_per_outcome",
    "expected_game_runs",
    "simulate_game",
    "MonteCarloSimulator",
    "DEFAULT_RUNS_PER_OUTCOME",
    "DEFAULT_TOTAL_RUNS_PER_OUTCOME",
]
