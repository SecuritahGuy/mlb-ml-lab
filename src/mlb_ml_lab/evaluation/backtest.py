"""Walk-forward backtesting with betting simulation and calibration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer

from mlb_ml_lab.models.train import (
    WalkForwardSplit,
    _build_model,
    _feature_columns,
    _merge_features_targets,
)


@dataclass
class GamePrediction:
    """A single out-of-sample prediction from walk-forward."""

    date: date
    player_id: int
    game_pk: int
    predicted_prob: float
    actual: int
    hits: int
    target_col: str


@dataclass
class BetResult:
    """Summary of a betting simulation over a backtest period."""

    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_stake: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    max_drawdown: float = 0.0
    predicted_prob_mean: float = 0.0
    avg_odds: float = 0.0
    threshold: float = 0.0
    stake_per_bet: float = 0.0
    target_col: str = ""
    model_type: str = ""
    n_seasons: int = 0
    daily_profits: list[float] = field(default_factory=list)


def walk_forward_predict(
    feature_matrix: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    target_col: str = "target_0.5",
    model_type: str = "lgb",
    n_splits: int = 5,
    seed: int = 42,
) -> list[GamePrediction]:
    """Run walk-forward validation and return every out-of-sample prediction.

    Mirrors the data-prep logic in ``train_baselines()`` but preserves
    per-game predictions instead of aggregating into fold metrics.

    Args:
        feature_matrix: Output from ``build_feature_matrix()``.
        targets: Output from ``make_targets()``.
        target_col: Which target column to predict (``target_0.5`` or
                    ``target_1.5``).
        model_type: Classifier type (``lr``, ``xgb``, ``rf``, ``lgb``).
        n_splits: Number of walk-forward folds.
        seed: Random seed for reproducibility.

    Returns:
        List of ``GamePrediction`` tuples with the predicted probability
        and actual outcome for every out-of-sample game.
    """
    merged = _merge_features_targets(feature_matrix, targets)
    if not merged:
        return []
    merged.sort(key=lambda r: r["date"])

    dates = [row["date"] for row in merged]
    feat_cols = _feature_columns(merged)

    x_all = np.array(
        [[row[c] for c in feat_cols] for row in merged], dtype=np.float64
    )
    y_all = np.array([row[target_col] for row in merged], dtype=np.int32)

    imputer = SimpleImputer(strategy="median")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        x_all = imputer.fit_transform(x_all)
    x_all = np.nan_to_num(x_all, nan=0.0)

    splitter = WalkForwardSplit(n_splits=n_splits)
    folds = splitter.split(dates)

    predictions: list[GamePrediction] = []
    for train_idx, test_idx in folds:
        model = _build_model(model_type, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(x_all[train_idx], y_all[train_idx])
        proba = model.predict_proba(x_all[test_idx])[:, 1]

        for idx, prob in zip(test_idx, proba.tolist()):
            predictions.append(
                GamePrediction(
                    date=merged[idx]["date"],
                    player_id=merged[idx]["player_id"],
                    game_pk=merged[idx]["game_pk"],
                    predicted_prob=round(float(prob), 4),
                    actual=int(y_all[idx]),
                    hits=merged[idx]["hits"],
                    target_col=target_col,
                )
            )

    return predictions


def simulate_bets(
    predictions: list[GamePrediction],
    decimal_odds: float = 1.909,
    stake_per_bet: float = 1.0,
    min_prob: float | None = None,
) -> BetResult:
    """Simulate flat-stake betting over out-of-sample predictions.

    Args:
        predictions: Out-of-sample predictions from
                     ``walk_forward_predict()``.
        decimal_odds: Assumed decimal odds for every bet
                     (1.909 ≈ -110 US).
        stake_per_bet: Unit stake per bet (default $1).
        min_prob: Minimum predicted probability to place a bet.
                  Defaults to the break-even probability
                  (``1 / decimal_odds``).

    Returns:
        ``BetResult`` with summary statistics.
    """
    if min_prob is None:
        min_prob = 1.0 / decimal_odds

    profits: list[float] = []
    wins = 0
    total_stake = 0.0

    for gp in predictions:
        if gp.predicted_prob < min_prob:
            continue
        total_stake += stake_per_bet
        if gp.actual == 1:
            payout = stake_per_bet * decimal_odds
            profit = payout - stake_per_bet
            wins += 1
        else:
            profit = -stake_per_bet
        profits.append(profit)

    total_bets = len(profits)
    if total_bets == 0:
        return BetResult(
            threshold=min_prob,
            stake_per_bet=stake_per_bet,
            avg_odds=decimal_odds,
        )

    total_profit = float(np.sum(profits))
    daily = _daily_profits(predictions, profits, min_prob)
    mdd = max_drawdown(daily)
    predicted_probs = [
        gp.predicted_prob
        for gp in predictions
        if gp.predicted_prob >= min_prob
    ]
    pred_mean = float(np.mean(predicted_probs)) if predicted_probs else 0.0

    return BetResult(
        total_bets=total_bets,
        wins=wins,
        losses=total_bets - wins,
        win_rate=wins / total_bets if total_bets > 0 else 0.0,
        total_stake=total_stake,
        total_profit=round(total_profit, 2),
        roi=round(total_profit / total_stake, 4) if total_stake > 0 else 0.0,
        max_drawdown=round(mdd, 4),
        predicted_prob_mean=round(pred_mean, 4),
        avg_odds=decimal_odds,
        threshold=round(min_prob, 4),
        stake_per_bet=stake_per_bet,
        daily_profits=daily,
    )


def calibration_buckets(
    predictions: list[GamePrediction],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Compute calibration buckets (reliability diagram).

    Groups predictions into *n_bins* equal-width probability buckets
    and returns the mean predicted probability, observed frequency,
    and count for each bucket.

    Args:
        predictions: Out-of-sample predictions.
        n_bins: Number of probability bins (default 10).

    Returns:
        List of dicts with keys ``bin_lower``, ``bin_upper``,
        ``mean_predicted``, ``observed_freq``, ``count``.
    """
    if not predictions:
        return []

    probas = np.array([gp.predicted_prob for gp in predictions])
    actuals = np.array([gp.actual for gp in predictions])

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    results: list[dict[str, float]] = []
    for i in range(n_bins):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        if i == n_bins - 1:
            mask = (probas >= lo) & (probas <= hi)
        else:
            mask = (probas >= lo) & (probas < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        results.append(
            {
                "bin_lower": round(lo, 3),
                "bin_upper": round(hi, 3),
                "mean_predicted": round(float(probas[mask].mean()), 4),
                "observed_freq": round(float(actuals[mask].mean()), 4),
                "count": count,
            }
        )
    return results


def max_drawdown(cumulative_profits: list[float]) -> float:
    """Compute maximum drawdown from a sequence of cumulative profits.

    Args:
        cumulative_profits: Ordered list of cumulative profit values
                            (e.g. daily running total).

    Returns:
        Maximum drawdown as a fraction (e.g. 0.25 = 25% peak-to-trough
        decline).  Returns 0.0 if the sequence is flat or always
        increasing.
    """
    if len(cumulative_profits) < 2:
        return 0.0
    arr = np.asarray(cumulative_profits, dtype=np.float64)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (peaks - arr) / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns))


def print_backtest_report(
    result: BetResult,
    calibration: list[dict[str, float]] | None = None,
) -> None:
    """Print a formatted backtest report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("BACKTEST REPORT")
    print(sep)
    if result.target_col:
        print(f"  Target:          {result.target_col}")
    if result.model_type:
        print(f"  Model:           {result.model_type}")
    if result.n_seasons:
        print(f"  Seasons:         {result.n_seasons}")
    print(f"  Odds:            {result.avg_odds:.3f} "
          f"(breakeven: {result.threshold:.3f})")
    print(f"  Stake/bet:       ${result.stake_per_bet:.2f}")
    print(f"  Min probability: {result.threshold:.3f}")
    print(sep)
    print(f"  Total bets:      {result.total_bets}")
    print(f"  Wins:            {result.wins}")
    print(f"  Losses:          {result.losses}")
    print(f"  Win rate:        {result.win_rate:.3f}")
    print(f"  Total stake:     ${result.total_stake:.2f}")
    print(f"  Total profit:    ${result.total_profit:.2f}")
    print(f"  ROI:             {result.roi:+.4f} ({result.roi*100:+.2f}%)")
    print(f"  Max drawdown:    {result.max_drawdown:.4f} "
          f"({result.max_drawdown*100:.2f}%)")
    print(f"  Avg prob:        {result.predicted_prob_mean:.4f}")
    print(sep)

    if calibration:
        print("\nCalibration:")
        print("  Bucket       Predicted  Observed    Count  AbsErr")
        print("  " + "-" * 55)
        total_abs_err = 0.0
        cal_entries = 0
        for b in calibration:
            err = abs(b["observed_freq"] - b["mean_predicted"])
            total_abs_err += err * b["count"]
            cal_entries += b["count"]
            print(
                f"  [{b['bin_lower']:.2f}-{b['bin_upper']:.2f})  "
                f"{b['mean_predicted']:.4f}     "
                f"{b['observed_freq']:.4f}    "
                f"{b['count']:>5d}  {err:.4f}"
            )
        if cal_entries > 0:
            mce = total_abs_err / cal_entries
            print(f"\n  Mean calibration error: {mce:.4f}")
        print(sep)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _daily_profits(
    predictions: list[GamePrediction],
    bet_profits: list[float],
    threshold: float,
) -> list[float]:
    """Aggregate bet-level profits into daily cumulative profits."""
    bet_dates = [
        gp.date for gp in predictions if gp.predicted_prob >= threshold
    ]
    daily_pnl: dict[date, float] = {}
    for d, p in zip(bet_dates, bet_profits):
        daily_pnl[d] = daily_pnl.get(d, 0.0) + p

    cumulative = []
    running = 0.0
    for d in sorted(daily_pnl):
        running += daily_pnl[d]
        cumulative.append(running)
    return cumulative
