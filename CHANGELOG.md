# Changelog

All notable changes to this project are documented here. The format is based on
keeping entries concise and chronological.

## [0.2.0] - 2026-07-15

### Added
- **Real SBR moneyline +EV backtest** (`pipeline/odds_backtest.py`): the ensemble's
  per-player hit probabilities are summed to team expected hits, bridged to a win
  probability via a walk-forward logistic fit on real team hit-differences, and
  compared against SBR vig-free moneylines. Bets settle on the actual game result.
  Flat-stake and Kelly simulations report ROI, win rate, and max drawdown.
- **Out-of-fold Platt/temperature calibration** for the moneyline hit-edge signal
  (fit on held-out-season predictions to avoid in-sample optimism).
- **Per-season isotonic recalibration** for player-prop probabilities (5-fold
  cross-fit within each season). Drops ECE from ~0.070 -> ~0.002 (target 0.5) and
  ~0.037 -> ~0.001 (target 1.5).

### Results
- Calibrated moneyline edge is modestly positive: **+2.62% flat / +3.39% Kelly**
  at edge >= 0.10 (raw uncalibrated: +1.67% / +1.68%).
- Player hit-prop market lines are **not** available from SBR's free page (only
  game-level moneyline / run-line / totals), so prop +EV is assessed by
  calibration alone, not against market prices.

### Docs
- Marked Phase 3/4 roadmap items complete; added a "Backtest results" subsection
  to `ROADMAP.md`.

## [0.1.0] - initial

- Foundation: data layer (MLB Stats API wrappers), feature engineering, baseline
  models (LogisticRegression, XGBoost), hybrid sequence ensemble (GRU + XGBoost),
  walk-forward validation, and the core hit-over-0.5 / 1.5 prediction pipeline.
