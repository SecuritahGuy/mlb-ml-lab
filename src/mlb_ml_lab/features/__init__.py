# Import all feature extractors so they register themselves
import mlb_ml_lab.features.bullpen  # noqa: F401
import mlb_ml_lab.features.context  # noqa: F401
import mlb_ml_lab.features.forecast  # noqa: F401
import mlb_ml_lab.features.league  # noqa: F401
import mlb_ml_lab.features.matchup  # noqa: F401
import mlb_ml_lab.features.pitching  # noqa: F401
import mlb_ml_lab.features.player  # noqa: F401
import mlb_ml_lab.features.rolling  # noqa: F401
import mlb_ml_lab.features.rolling_advanced  # noqa: F401
import mlb_ml_lab.features.rolling_statcast  # noqa: F401
import mlb_ml_lab.features.schedule  # noqa: F401
import mlb_ml_lab.features.statcast  # noqa: F401
import mlb_ml_lab.features.streaks  # noqa: F401
import mlb_ml_lab.features.gamepace  # noqa: F401
import mlb_ml_lab.features.teamleaders  # noqa: F401
import mlb_ml_lab.features.odds_features  # noqa: F401
import mlb_ml_lab.features.team_trends  # noqa: F401

from mlb_ml_lab.features.assemble import (
    build_feature_matrix,
    describe_features,
    load_feature_data,
    load_game_logs,
    save_feature_data,
)
from mlb_ml_lab.features.targets import make_targets

__all__ = [
    "build_feature_matrix",
    "describe_features",
    "load_feature_data",
    "load_game_logs",
    "save_feature_data",
    "make_targets",
]
