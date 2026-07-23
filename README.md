# mlb-ml-lab

MLB prediction models — fetch player and team data, build feature matrices, train models,
and evaluate hit over/under forecasts.

## Features

- **Zero ML dependencies** (no `pybaseball`, `pybaseballstats`, `python-mlb-statsapi`). A
  custom `httpx`-based client wraps `statsapi.mlb.com` and
  `baseballsavant.mlb.com` directly.
- **Typed schemas** throughout — `PlayerGameLog`, `TeamInfo`, `RosterPlayer`, etc. are
  typed dataclasses.
- **Disk caching** with per-key TTL — avoids hammering the MLB API during development.
- **Rate limiting** — built-in token bucket (10 req/s).
- **Park factors** scraped live from Baseball Savant with static fallbacks.
- **NWS weather forecasts** — free, no API key, covers every MLB venue.
- **Feature engineering pipeline** — plugin-based extractors with a registry pattern,
  designed to be extractable as its own package.
- **Walk-forward validation** — no random train/test splits. Sports data is temporally
  dependent.

## Installation

```bash
# Clone the repo
git clone https://github.com/timhollingsworth/mlb-ml-lab
cd mlb-ml-lab

# Install with Poetry
poetry install
```

Requires Python 3.12+.

## Quick Start

### Fetch player game logs

```python
from mlb_ml_lab import MlbClient

client = MlbClient()

# Get all teams
teams = client.get_teams()

# Get roster for a team (Angels = 108)
roster = client.get_roster(108)

# Get game logs for a player (Shohei Ohtani = 660271)
logs = client.get_player_game_log(660271, season=2024)

# Each log has typed fields
for log in logs:
    print(log.date, log.hits, log.at_bats)
```

### Fetch game context (venue, weather, datetime)

```python
# Game feed gives you venue, weather, and game datetime
feed = client.get_game_context(778554)
# → {"venue_id": 4, "venue_name": "Rate Field",
#    "game_datetime": "2025-03-27T20:10:00Z",
#    "weather_condition": "Cloudy", "weather_temp": "68", ...}
```

### Build a feature matrix

```python
from mlb_ml_lab import MlbClient, build_feature_matrix, describe_features, make_targets

client = MlbClient()

# 1. Fetch data
teams = client.get_teams()
logs = client.get_player_game_log(660271, season=2024)
contexts = {778554: client.get_game_context(778554)}

# 2. Assemble features (runs all registered extractors)
matrix = build_feature_matrix(
    logs,
    season=2024,
    teams=teams,
    extra_kwargs={"game_contexts": contexts},
)

# 3. See what features are available
metas = describe_features()
for m in metas:
    print(f"{m.name:40s} {m.source:10s} {m.description}")

# 4. Create target labels
targets = make_targets(logs)
```

### Weather forecast for an upcoming game

```python
from datetime import datetime
from mlb_ml_lab import NwsWeather

nws = NwsWeather()

# Angel Stadium (venue_id=1) at game time
forecast = nws.forecast(1, target_time=datetime(2025, 7, 4, 19, 7))
# → {"temp": 75, "wind_speed": "8 mph", "wind_direction": "SW",
#    "precip_pct": 10, "conditions": "Partly Cloudy", "source": "forecast"}
```

### Park factors

```python
from mlb_ml_lab import ParkFactors

pf = ParkFactors()
# Coors Field (venue_id=19) 2024 wOBA factor
factor = pf.factor(19, "wOBA", season=2024)
print(factor)  # e.g. 1.11 (11% boost)
```

### GPU-accelerated models (Apple Silicon)

A full neural model toolbox runs on Apple's MLX framework (Metal GPU):

| Model | Description | File |
|-------|-------------|------|
| `MlxNNClassifier` | sklearn-compatible MLP | `models/mlx_nn.py` |
| `SequenceHitPredictor` | GRU over 15-game stat windows | `models/sequence.py` |
| `HybridHitPredictor` | GRU + context-feature MLP | `models/sequence.py` |
| `MultiTaskHybridPredictor` | Shared encoder + two heads (0.5/1.5 targets) | `models/sequence.py` |
| `DCNMultiTaskPredictor` | Deep & Cross Network on context features | `models/sequence.py` |
| `TransformerMultiTaskPredictor` | Transformer encoder replacing GRU | `models/sequence.py` |

All plug into the walk-forward training pipeline via `--model mlx` or as
ensemble components. Benchmark: `pipeline/benchmark_mlx.py`.

## Project Structure

```
mlb-ml-lab/
├── src/
│   └── mlb_ml_lab/
│       ├── data/               # Data layer (installable)
│       │   ├── client.py       # MlbClient — MLB Stats API + Baseball Savant
│       │   ├── schemas.py      # Typed dataclasses
│       │   ├── cache.py        # DiskCache (JSON, per-key TTL)
│       │   ├── rate_limiter.py # TokenBucket rate limiter
│       │   ├── parks.py        # ParkFactors (Savant scrape + fallback)
│       │   └── weather.py      # NwsWeather (NWS API, free, no key)
│       └── features/           # Feature engineering (installable)
│           ├── base.py         # FeatureExtractor ABC, registry
│           ├── rolling.py      # Rolling window stats (hits, PA, BABIP)
│           ├── context.py      # Home/away, rest days, park factors, weather
│           ├── matchup.py      # Opponent pitching stats
│           ├── statcast.py     # Statcast advanced metrics
│           ├── forecast.py     # NWS weather forecast features
│           ├── assemble.py     # build_feature_matrix(), describe_features()
│           └── targets.py      # make_targets() for hit thresholds
├── pipeline/                   # Modeling (training, prediction, evaluation)
├── tests/
│   ├── data/                   # Tests for data layer
│   ├── features/               # Tests for feature engineering
│   ├── models/                 # Tests for model training/evaluation
│   └── evaluation/             # Tests for backtesting/calibration
├── data/                       # Raw/processed datasets (gitignored)
│   └── betting/                # P&L tracking (pnl.json)
├── experiments/                # Analysis scripts (not notebooks)
├── pyproject.toml
├── README.md
├── LICENSE
├── AGENTS.md                   # Dev instructions (AI assistant)
└── ROADMAP.md                  # Build-out plan
```

## CLI

A command-line interface is available after install:

```bash
# Fetch data and build feature matrix
mlb fetch --seasons 2024 2025 --max-players 20

# Walk-forward validation training
mlb train --use-cached

# Predict on a season with a saved model
mlb predict --season 2026

# Walk-forward backtest with betting simulation (single model)
mlb backtest --model lgb

# Ensemble backtest (uniform average of all four)
mlb backtest --model lr,xgb,rf,lgb

# Hyperparameter tuning
mlb tune --trials 20

# Quick end-to-end for one team
mlb e2e --team-id 108 --season 2024
```

### Daily betting strategy

The `mlb bet` command generates player-prop bets from a uniform-average
ensemble of LogisticRegression, XGBoost, RandomForest, and LightGBM:

```bash
# Generate today's bets (P(hit ≥ 1) > 0.55, $1 per bet)
mlb bet

# Settle yesterday's bets and update P&L
mlb bet --settle

# View running P&L
mlb bet --pnl

# Custom threshold and stake
mlb bet --threshold 0.60 --stake 5.00

# Use a trained single model instead of ensemble
mlb bet --model-dir data/models/final_0_5

# Specific date
mlb bet --date 2026-07-23
```

#### Backtest results (4-season walk-forward, 2021–2024)

**Target: P(hit ≥ 1)** — Ensemble: 96K bets at 0.55 threshold, **+22.49%
ROI**. AUC 0.639 across 155K out-of-sample predictions.

```
Thresh    Bets  WinRate       ROI    MaxDD
 0.55    96632   0.6416    +22.49%    0.10%
 0.60    71112   0.6658    +27.11%    0.04%
 0.65    43105   0.6907    +31.84%    0.02%
 0.70    18000   0.7191    +37.27%    0.35%
 0.75     4403   0.7533    +43.81%    2.49%
```

**Target: P(hit ≥ 2)** — Not viable: 33 bets total with -36% ROI.

The system is self-calibrated (no market odds required). See the
[ROADMAP](./ROADMAP.md) for full details.

## Development

```bash
# Run fast tests (no live API calls)
poetry run pytest

# Run all tests including live API calls
poetry run pytest --runslow

# Run a single test
poetry run pytest tests/features/test_forecast.py::TestWeatherForecastFeatures::test_indoor_venue_returns_indoor -v

# Lint
poetry run ruff check .

# Format
poetry run ruff format .
```

### Adding a new feature extractor

1. Create a new module in `src/mlb_ml_lab/features/` (e.g. `src/mlb_ml_lab/features/schedule.py`).
2. Subclass `FeatureExtractor`, implement `features` and `extract`.
3. Decorate with `@register`.
4. Import it in `src/mlb_ml_lab/features/__init__.py`.
5. It will automatically be discovered by `build_feature_matrix()`.

## Data Sources

| Source | Endpoint | Key Required | Notes |
|--------|----------|-------------|-------|
| [MLB Stats API](https://statsapi.mlb.com/docs/) | `statsapi.mlb.com/api/v1/` | No | Rate limit ~10 req/s |
| [Baseball Savant](https://baseballsavant.mlb.com/) | `baseballsavant.mlb.com/leaderboard/` | No | CSV download, BOM stripping required |
| [NWS API](https://www.weather.gov/documentation/services-web-api) | `api.weather.gov` | No (User-Agent required) | Free, no key, hourly forecasts |

## License

MIT
