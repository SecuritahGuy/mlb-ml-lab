# mlb-ml-lab — Roadmap

Experimental models predicting MLB player hits over 0.5 and 1.5 thresholds.

## Phase 0: Foundation

- [x] `git init`, `.gitignore` (Python, data, venvs)
- [x] `pyproject.toml` via `poetry init`; dev deps: `pytest`, `ruff`, `httpx`
- [x] `src/mlb_ml_lab/` package scaffold with `__init__.py`
- [x] `AGENTS.md` from existing template

## Phase 1: Data Layer (custom, no library dependencies)

Build thin, testable wrappers around the official public MLB Stats API at `statsapi.mlb.com`. Reference `pybaseball` / `pybaseballstats` / `python-mlb-statsapi` for URL/endpoint patterns only — do not depend on them at runtime.

- [x] `src/mlb_ml_lab/data/client.py` — `httpx`-based HTTP client configures base URL, handles rate limits, returns raw JSON
- [x] `src/mlb_ml_lab/data/schemas.py` — typed dataclasses for player game logs, season stats, roster info
- [x] `src/mlb_ml_lab/data/cache.py` — local JSON-on-disk cache in `data/` to avoid replaying API calls
- [x] Entrypoints to fetch:
  - Player game logs for a season (the core dataset)
  - Season-level batting stats
  - Roster / opponent / venue metadata
  - Game context (weather, venue, boxscore)
  - Statcast expected stats and batters leaderboards (CSV)
  - Park factors (scraped from Baseball Savant HTML)
  - Team pitching stats for opponent features
- [x] `tests/data/test_client.py` — unit tests with fixture data, integration test against live API (marked `@pytest.mark.slow`)

**Data source**: [`https://statsapi.mlb.com/api/v1/`](https://statsapi.mlb.com/api/v1/) — official, free, no key required for read usage. Subject to MLB copyright notice.

## Phase 2: Feature Engineering (complete)

- [x] `src/mlb_ml_lab/features/rolling.py` — rolling averages (L5, L10, L20) of hits, PA, BB/K rate, BABIP; no lookahead
- [x] `src/mlb_ml_lab/features/matchup.py` — opponent team pitching features (ERA, K/9, WHIP, BAA, HR/9)
- [x] `src/mlb_ml_lab/features/context.py` — home/away, rest days, park factors, weather (condition, temp, wind)
- [x] `src/mlb_ml_lab/features/statcast.py` — advanced hitting metrics from Savant leaderboards (xBA, xwOBA, barrel %, exit velo, sweet-spot, FB/LD, GB)
- [x] `src/mlb_ml_lab/features/forecast.py` — NWS weather forecast features (temp, wind, precip, conditions)
- [x] `src/mlb_ml_lab/features/assemble.py` — ``build_feature_matrix()`` merges all extractors on (player_id, game_pk, date); ``describe_features()`` returns metadata
- [x] `src/mlb_ml_lab/features/targets.py` — binary targets for hit thresholds (0.5 and 1.5)
- [x] Unit tests: 117 tests covering correctness, no lookahead, null handling, edge cases

**Architecture**: Feature engineering lives in `src/mlb_ml_lab/features/` as part of the installable package. Each extractor is a registered `FeatureExtractor` subclass; the assembler discovers them via the registry. Optional data sources (`teams`, `game_contexts`, `opponent_pitching`, `statcast_batters`, `expected_stats`) are passed through kwargs and extractors silently default to None/1.0 when absent.  The `pipeline/` directory at project root is reserved for model training and prediction.

## Phase 3: Model Training (baseline complete)

- [x] `src/mlb_ml_lab/models/train.py` — `WalkForwardSplit` (expanding window) + `train_baselines()` for LogisticRegression and XGBoost
- [x] `src/mlb_ml_lab/models/evaluate.py` — metrics: accuracy, log-loss, ROC-AUC, Brier score
- [x] Walk-forward validation (expanding window, not random split)
- [x] End-to-end pipeline (`pipeline/run_end_to_end.py`): fetch real MLB data → featurize → walk-forward train → print metrics
- [x] Hyperparameter tuning (`tune_hyperparameters()`: random search inside walk-forward, per-model default grids, AUC or log-loss optimisation)
- [x] `expected_calibration_error()` — ECE metric over probability buckets
- [ ] Calibration curve + expected profit at market odds

## Phase 4: Backtesting & Odds Integration

- [x] `src/mlb_ml_lab/evaluation/backtest.py` — `walk_forward_predict()` capture, `simulate_bets()` flat-stake, `calibration_buckets()`, `max_drawdown()`, `print_backtest_report()`
- [x] `pipeline/backtest.py` — end-to-end script (fetch → featurize → walk-forward → simulate → report)
- [x] Unit tests: walk-forward prediction, betting simulation, calibration, drawdown
- [ ] Fetch real sportsbook lines (manually or via a free odds API) to evaluate +EV opportunities
- [ ] Kelly / fractional Kelly staking
- [ ] Track model confidence calibration — are probabilities well-calibrated at decision thresholds?

## Phase 5: Iteration & Tooling

- [ ] `experiments/` — Jupyter notebooks for exploration, feature ablation, error analysis
- [ ] CLI entry point (`typer` or `argparse`) — `mlb fetch`, `mlb train`, `mlb predict`
- [ ] Scheduled data refresh (cron / GitHub Actions)
- [ ] Optional: local LLM-powered analysis of mispredictions via LM Studio

## Guiding principles

- **No cloud AI APIs.** Only local LM Studio for optional LLM analysis.
- **Own the data layer.** Library source code is reference material, not runtime dependencies.
- **Walk-forward, never random.** Sports data is temporally dependent — every eval must respect game order.
- **Calibration matters more than accuracy.** For betting, well-calibrated probabilities are worth more than high raw accuracy.
- **Small, testable units.** Each pipeline stage should be verifiable in isolation.
