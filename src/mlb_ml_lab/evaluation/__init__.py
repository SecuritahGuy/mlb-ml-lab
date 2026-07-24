"""Betting backtesting, calibration, and staking analysis."""

from mlb_ml_lab.evaluation.backtest import (
    GamePrediction,
    BetResult,
    walk_forward_predict,
    simulate_bets,
    simulate_kelly_bets,
    kelly_fraction,
    calibration_buckets,
    expected_calibration_error,
    isotonic_calibrate,
    fit_season_calibrators,
    apply_calibrators,
    calibrate_predictions_crossfit,
    save_calibrators,
    load_calibrators,
    max_drawdown,
    print_backtest_report,
)

__all__ = [
    "GamePrediction",
    "BetResult",
    "walk_forward_predict",
    "simulate_bets",
    "simulate_kelly_bets",
    "kelly_fraction",
    "calibration_buckets",
    "expected_calibration_error",
    "isotonic_calibrate",
    "fit_season_calibrators",
    "apply_calibrators",
    "calibrate_predictions_crossfit",
    "save_calibrators",
    "load_calibrators",
    "max_drawdown",
    "print_backtest_report",
]
