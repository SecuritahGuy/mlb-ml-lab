# mibl — Roadmap

Experimental models predicting MLB player hits over 0.5 and 1.5 thresholds.

## Phase 0: Foundation

- [x] `git init`, `.gitignore` (Python, data, venvs)
- [x] `pyproject.toml` via `poetry init`; dev deps: `pytest`, `ruff`, `httpx`
- [x] `src/mibl/` package scaffold with `__init__.py`
- [x] `AGENTS.md` from existing template

## Phase 1: Data Layer (custom, no library dependencies)

Build thin, testable wrappers around the official public MLB Stats API at `statsapi.mlb.com`. Reference `pybaseball` / `pybaseballstats` / `python-mlb-statsapi` for URL/endpoint patterns only — do not depend on them at runtime.

- [x] `src/mibl/data/client.py` — `httpx`-based HTTP client configures base URL, handles rate limits, returns raw JSON
- [x] `src/mibl/data/schemas.py` — typed dataclasses for player game logs, season stats, roster info
- [x] `src/mibl/data/cache.py` — local JSON-on-disk cache in `data/` to avoid replaying API calls
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

- [x] `pipeline/rolling.py` — rolling averages (L5, L10, L20) of hits, PA, BB/K rate, BABIP; no lookahead
- [x] `pipeline/matchup.py` — opponent team pitching features (ERA, K/9, WHIP, BAA, HR/9)
- [x] `pipeline/context.py` — home/away, rest days, park factors, weather (condition, temp, wind)
- [x] `pipeline/statcast.py` — advanced hitting metrics from Savant leaderboards (xBA, xwOBA, barrel %, exit velo)
- [x] `pipeline/assemble.py` — ``build_feature_matrix()`` merges all extractors on (player_id, game_pk, date); ``describe_features()`` returns metadata
- [x] `pipeline/targets.py` — binary targets for hit thresholds (0.5 and 1.5)
- [x] Unit tests: 49 pipeline tests covering correctness, no lookahead, null handling, edge cases
- [x] `MlbClient.get_team_pitching_stats()` convenience method

**Architecture**: The `pipeline/` directory is a separable feature engineering package. Each extractor is a registered `FeatureExtractor` subclass; the assembler discovers them via the registry. Optional data sources (`teams`, `game_contexts`, `opponent_pitching`, `statcast_batters`, `expected_stats`) are passed through kwargs and extractors silently default to None/1.0 when absent.

## Phase 3: Model Training

- [ ] `src/mibl/models/train.py` — binary classifiers for two targets:
  - `hits_over_0_5` (~45-55% base rate)
  - `hits_over_1_5` (~20-30% base rate — class imbalance expected)
- [ ] Candidate models: `LogisticRegression` (baseline), `GradientBoostingClassifier`, `XGBoost`
- [ ] `src/mibl/models/evaluate.py` — metrics: accuracy, log-loss, ROC-AUC, calibration curve, expected profit at market odds
- [ ] **Walk-forward validation** (not random split) — time-series-aware train/test splits to prevent lookahead bias

## Phase 4: Backtesting & Odds Integration

- [ ] Fetch real sportsbook lines (manually or via a free odds API) to evaluate +EV opportunities
- [ ] `src/mibl/evaluation/backtest.py` — simulate betting over historical seasons: stake sizing, ROI, max drawdown
- [ ] Track model confidence calibration — are probabilities well-calibrated at decision thresholds?

## Phase 5: Iteration & Tooling

- [ ] `experiments/` — Jupyter notebooks for exploration, feature ablation, error analysis
- [ ] CLI entry point (`typer` or `argparse`) — `mibl fetch`, `mibl train`, `mibl predict`
- [ ] Scheduled data refresh (cron / GitHub Actions)
- [ ] Optional: local LLM-powered analysis of mispredictions via LM Studio

## Guiding principles

- **No cloud AI APIs.** Only local LM Studio for optional LLM analysis.
- **Own the data layer.** Library source code is reference material, not runtime dependencies.
- **Walk-forward, never random.** Sports data is temporally dependent — every eval must respect game order.
- **Calibration matters more than accuracy.** For betting, well-calibrated probabilities are worth more than high raw accuracy.
- **Small, testable units.** Each pipeline stage should be verifiable in isolation.
