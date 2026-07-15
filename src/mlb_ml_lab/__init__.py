"""mlb-ml-lab — MLB prediction models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    from mlb_ml_lab.models.train import load_model, save_model, train_final

__all__ = [
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
    "load_model",
    "make_targets",
    "save_feature_data",
    "save_model",
    "train_final",
]


def __getattr__(name: str) -> Any:
    if name in {"load_model", "save_model", "train_final"}:
        from mlb_ml_lab.models.train import load_model, save_model, train_final

        return {
            "load_model": load_model,
            "save_model": save_model,
            "train_final": train_final,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
