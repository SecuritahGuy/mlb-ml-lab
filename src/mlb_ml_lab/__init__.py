"""mlb-ml-lab — MLB prediction models and analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlb_ml_lab.analytics import (
    add_advanced_metrics,
    aggregate_player_seasons,
    classify_player,
    comparison_metrics,
    compute_all_war,
    compute_babip,
    compute_baserunning_runs,
    compute_iso,
    compute_league_rate_stats,
    compute_league_woba,
    compute_obp,
    compute_ops_plus,
    compute_player_war,
    compute_positional_runs,
    compute_replacement_runs,
    compute_slg,
    compute_war_per_162,
    compute_war_per_600,
    compute_woba,
    compute_woba_plus,
    compute_wrc_plus,
    compute_wraa,
    fetch_all_fg_fielding,
    fetch_fangraphs_war,
    fetch_supplemental_stats,
    get_park_factor,
    load_jsonl,
    match_players,
    park_adjust_wraa,
    print_archetype_summary,
    print_metrics,
    print_war_table,
    venue_for_team,
)

from mlb_ml_lab.data import (
    BoxscorePlayer,
    DiskCache,
    INDOOR_VENUES,
    MlbClient,
    NwsWeather,
    ParkFactors,
    PlayerDetail,
    PlayerGameLog,
    PlayerIdResolver,
    RosterPlayer,
    StandingRecord,
    TeamInfo,
    TokenBucket,
    VenueInfo,
)

from mlb_ml_lab.features import (
    build_feature_matrix,
    describe_features,
    load_feature_data,
    load_game_logs,
    make_targets,
    save_feature_data,
)

from mlb_ml_lab.simulation import (
    MonteCarloSimulator,
    blend_outcomes,
    compute_league_averages,
    compute_player_outcomes,
    compute_re24,
    compute_runs_per_outcome,
    expected_game_runs,
    load_outcome_distributions,
    load_pbp_dataset,
    print_re24,
    save_outcome_distributions,
    simulate_game,
)

if TYPE_CHECKING:
    from mlb_ml_lab.models.train import (
        load_ensemble,
        load_model,
        save_model,
        train_final,
    )

__all__ = [  # pylint: disable=undefined-all-variable
    # Analytics
    "add_advanced_metrics",
    "aggregate_player_seasons",
    "classify_player",
    "comparison_metrics",
    "compute_all_war",
    "compute_babip",
    "compute_baserunning_runs",
    "compute_iso",
    "compute_league_rate_stats",
    "compute_league_woba",
    "compute_obp",
    "compute_ops_plus",
    "compute_player_war",
    "compute_positional_runs",
    "compute_replacement_runs",
    "compute_slg",
    "compute_war_per_162",
    "compute_war_per_600",
    "compute_woba",
    "compute_woba_plus",
    "compute_wrc_plus",
    "compute_wraa",
    "fetch_all_fg_fielding",
    "fetch_fangraphs_war",
    "fetch_supplemental_stats",
    "get_park_factor",
    "load_jsonl",
    "match_players",
    "park_adjust_wraa",
    "print_archetype_summary",
    "print_metrics",
    "print_war_table",
    "venue_for_team",
    # Data
    "BoxscorePlayer",
    "DiskCache",
    "INDOOR_VENUES",
    "MlbClient",
    "NwsWeather",
    "ParkFactors",
    "PlayerDetail",
    "PlayerGameLog",
    "PlayerIdResolver",
    "RosterPlayer",
    "StandingRecord",
    "TeamInfo",
    "TokenBucket",
    "VenueInfo",
    # Features
    "build_feature_matrix",
    "describe_features",
    "load_feature_data",
    "load_game_logs",
    "make_targets",
    "save_feature_data",
    # Models (lazy)
    "load_ensemble",
    "load_model",
    "save_model",
    "train_final",
    # Simulation
    "MonteCarloSimulator",
    "blend_outcomes",
    "compute_league_averages",
    "compute_player_outcomes",
    "compute_re24",
    "compute_runs_per_outcome",
    "expected_game_runs",
    "load_outcome_distributions",
    "load_pbp_dataset",
    "print_re24",
    "save_outcome_distributions",
    "simulate_game",
]


def __getattr__(name: str) -> object:
    if name in {"load_ensemble", "load_model", "save_model", "train_final"}:
        from mlb_ml_lab.models.train import (
            load_ensemble,
            load_model,
            save_model,
            train_final,
        )

        return {
            "load_ensemble": load_ensemble,
            "load_model": load_model,
            "save_model": save_model,
            "train_final": train_final,
        }[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
