"""Regime-switching classifier.

Trains separate models for different player regimes (e.g., hot vs cold
streak) and routes predictions to the appropriate model based on the
sample's regime feature value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

_BASE_FACTORIES: dict[str, Any] = {
    "lr": lambda rs, nj, **kw: LogisticRegression(
        random_state=rs, n_jobs=nj, max_iter=1000, **kw
    ),
    "xgb": lambda rs, nj, **kw: XGBClassifier(
        random_state=rs, n_jobs=nj, eval_metric="logloss", verbosity=0, **kw
    ),
    "rf": lambda rs, nj, **kw: RandomForestClassifier(
        random_state=rs, n_jobs=nj, **kw
    ),
}


class RegimeSwitchingClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper that trains separate base models per player regime.

    At fit time, splits training data by a regime feature (e.g.
    ``hit_rate_last_10``) and trains one base model per side. At
    predict time, routes each sample to the appropriate model.

    Parameters
    ----------
    base_model_type : str
        One of ``"lr"``, ``"xgb"``, ``"rf"``.
    regime_feature : str
        Name of the column to use as the regime indicator. Must be
        passed via ``feature_names`` at fit time.
    threshold : float
        Split threshold. Samples with ``regime_feature >= threshold``
        are routed to the "high" model.
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Parallelism for base models.
    base_params : dict
        Additional keyword arguments forwarded to each base model.
    """

    def __init__(
        self,
        base_model_type: str = "xgb",
        regime_feature: str = "hit_rate_last_10",
        threshold: float = 0.4,
        random_state: int = 42,
        n_jobs: int = -1,
        **base_params: Any,
    ):
        self.base_model_type = base_model_type
        self.regime_feature = regime_feature
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.base_params = base_params
        self._col_idx: int | None = None
        self.high_model_: Any = None
        self.low_model_: Any = None
        self.classes_: np.ndarray | None = None

    def fit(  # type: ignore[override]
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> RegimeSwitchingClassifier:
        factory = _BASE_FACTORIES.get(self.base_model_type)
        if factory is None:
            raise ValueError(
                f"Unknown base_model_type: {self.base_model_type}. "
                f"Choose from {list(_BASE_FACTORIES)}"
            )
        if feature_names is None:
            raise ValueError(
                "feature_names is required for RegimeSwitchingClassifier.fit(). "
                "Pass the list of column names."
            )
        try:
            col_idx = feature_names.index(self.regime_feature)
        except ValueError:
            raise ValueError(
                f"Regime feature '{self.regime_feature}' not found in "
                f"feature_names (available: {feature_names})"
            ) from None

        regime = X[:, col_idx] >= self.threshold
        self._col_idx = col_idx

        self.high_model_ = factory(
            self.random_state, self.n_jobs, **self.base_params
        )
        self.low_model_ = factory(
            self.random_state, self.n_jobs, **self.base_params
        )
        self.high_model_.fit(X[regime], y[regime])
        self.low_model_.fit(X[~regime], y[~regime])

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        regime = X[:, self._col_idx] >= self.threshold
        probas = np.zeros((X.shape[0], 2), dtype=np.float64)
        if regime.any():
            probas[regime] = self.high_model_.predict_proba(X[regime])
        if (~regime).any():
            probas[~regime] = self.low_model_.predict_proba(X[~regime])
        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
