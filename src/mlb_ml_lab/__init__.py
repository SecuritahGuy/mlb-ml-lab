"""mlb-ml-lab — MLB prediction models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlb_ml_lab.data import (
    BoxscorePlayer,
    DiskCache,
    INDOOR_VENUES,
    MlbClient,
    NwsWeather,
    ParkFactors,
    PlayerDetail,
    PlayerGameLog,
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

if TYPE_CHECKING:
    from mlb_ml_lab.models.train import load_ensemble, load_model, save_model, train_final

__all__ = [  # pylint: disable=undefined-all-variable
    "BoxscorePlayer",
    "DiskCache",
    "INDOOR_VENUES",
    "MlbClient",
    "NwsWeather",
    "ParkFactors",
    "PlayerDetail",
    "PlayerGameLog",
    "RosterPlayer",
    "StandingRecord",
    "TeamInfo",
    "TokenBucket",
    "VenueInfo",
    "build_feature_matrix",
    "describe_features",
    "load_feature_data",
    "load_game_logs",
    "load_ensemble",
    "load_model",
    "make_targets",
    "save_feature_data",
    "save_model",
    "train_final",
]


def __getattr__(name: str) -> object:
    if name in {"load_ensemble", "load_model", "save_model", "train_final"}:
        from mlb_ml_lab.models.train import load_ensemble, load_model, save_model, train_final

        return {
            "load_ensemble": load_ensemble,
            "load_model": load_model,
            "save_model": save_model,
            "train_final": train_final,
        }[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
