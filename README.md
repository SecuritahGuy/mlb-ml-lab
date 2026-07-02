# mibl

**M**LB **I**nning-**B**ased hit over/under prediction models — fetch, featurize, and predict
whether MLB players will clear 0.5 and 1.5 hit thresholds.

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
git clone https://github.com/timhollingsworth/mibl
cd mibl

# Install with Poetry
poetry install
```

Requires Python 3.12+.

## Quick Start

### Fetch player game logs

```python
from mibl import MlbClient

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
from mibl import MlbClient, build_feature_matrix, describe_features, make_targets

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
from mibl import NwsWeather

nws = NwsWeather()

# Angel Stadium (venue_id=1) at game time
forecast = nws.forecast(1, target_time=datetime(2025, 7, 4, 19, 7))
# → {"temp": 75, "wind_speed": "8 mph", "wind_direction": "SW",
#    "precip_pct": 10, "conditions": "Partly Cloudy", "source": "forecast"}
```

### Park factors

```python
from mibl import ParkFactors

pf = ParkFactors()
# Coors Field (venue_id=19) 2024 wOBA factor
factor = pf.factor(19, "wOBA", season=2024)
print(factor)  # e.g. 1.11 (11% boost)
```

## Project Structure

```
mibl/
├── src/
│   └── mibl/
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
│   └── features/               # Tests for feature engineering
├── data/                       # Raw/processed datasets (gitignored)
├── experiments/                # Notebooks (gitignored)
├── pyproject.toml
├── README.md
├── LICENSE
├── AGENTS.md                   # Dev instructions (AI assistant)
└── ROADMAP.md                  # Build-out plan
```

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

1. Create a new module in `src/mibl/features/` (e.g. `src/mibl/features/schedule.py`).
2. Subclass `FeatureExtractor`, implement `features` and `extract`.
3. Decorate with `@register`.
4. Import it in `src/mibl/features/__init__.py`.
5. It will automatically be discovered by `build_feature_matrix()`.

## Data Sources

| Source | Endpoint | Key Required | Notes |
|--------|----------|-------------|-------|
| [MLB Stats API](https://statsapi.mlb.com/docs/) | `statsapi.mlb.com/api/v1/` | No | Rate limit ~10 req/s |
| [Baseball Savant](https://baseballsavant.mlb.com/) | `baseballsavant.mlb.com/leaderboard/` | No | CSV download, BOM stripping required |
| [NWS API](https://www.weather.gov/documentation/services-web-api) | `api.weather.gov` | No (User-Agent required) | Free, no key, hourly forecasts |

## License

MIT
