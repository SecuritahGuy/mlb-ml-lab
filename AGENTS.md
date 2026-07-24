# mlb-ml-lab — MLB prediction models

Experimental models predicting whether MLB player hits clear 0.5 and 1.5 thresholds.

**Roadmap**: [`ROADMAP.md`](./ROADMAP.md)

## Stack

- Python 3.12, Poetry, `src/mlb_ml_lab/` package layout
- pytest (run: `poetry run pytest`)
- Type hints expected; no strict type checker configured yet
- All AI inference goes through local LM Studio at `http://127.0.0.1:1234/v1` — never call cloud APIs
- MLX (Apple Silicon GPU) for neural models; sklearn (CPU) for LR/XGB/RF/LGBM

## Commands

| Action | Command |
|--------|---------|
| Install deps | `poetry install` |
| Add dep | `poetry add <package>` |
| Add dev dep | `poetry add --group dev <package>` |
| Run all tests | `poetry run pytest` |
| Run single test | `poetry run pytest tests/path/to/test_file.py::test_name -v` |
| Run live API tests | `poetry run pytest --runslow` |
| Run lint | `poetry run ruff check .` |
| Run pylint | `poetry run pylint src/mlb_ml_lab/ tests/ pipeline/` |
| Run formatter | `poetry run ruff format .` |
| Typecheck | Not yet configured |
| Run full backtest (ensemble) | `poetry run python -c "from mlb_ml_lab.cli import main; main()" backtest --model lr,xgb,rf,lgb` |
| Bench MLX GPU speed | `poetry run python pipeline/benchmark_mlx.py` |
| Compute WAR + advanced metrics | `poetry run python experiments/compute_war.py --season 2024 --metrics --archetypes` |
| Compute WAR (all seasons) | `poetry run python experiments/compute_war.py --min-pa 200 --save data/out/war_all.json` |

### CLI

| Command | Description |
|---------|-------------|
| `mlb fetch` | Pull game logs + feature matrix from MLB API |
| `mlb train` | Walk-forward training on cached data |
| `mlb predict` | Generate predictions for a season |
| `mlb backtest` | Walk-forward + betting simulation |
| `mlb bet` | Daily betting (generate/settle/pnl) |
| `mlb tune` | Hyperparameter search |
| `mlb e2e` | Quick end-to-end for one team |

## Conventions

- **No cloud AI APIs.** All model calls go through the local endpoint. Use `openai` SDK with `base_url = http://127.0.0.1:1234/v1`.
- **No MLB data library dependencies.** Reference `pybaseball`/`pybaseballstats`/`python-mlb-statsapi` for endpoint patterns but own the data layer. Thin `httpx` wrappers around `statsapi.mlb.com`.
- `data/` for raw/processed datasets (gitignored if large). Keep small samples in `tests/fixtures/`.
- `src/mlb_ml_lab/models/` for models; `src/mlb_ml_lab/features/` for feature engineering; `experiments/` for scripts.
- Walk-forward validation (never random train/test split) — sports data is temporally dependent.
- Feature engineering lives in `src/mlb_ml_lab/features/` (part of the installable package). The `pipeline/` directory at project root is for model training/prediction only.
- MLX models use Apple Silicon GPU; the `--model mlx` flag plugs them into the walk-forward pipeline.

## Key Results

- **Ensemble (LR+XGB+RF+LGBM) target_0.5**: AUC 0.639, **+22.5% ROI** at 0.55 threshold, **+37.3%** at 0.70 (96K→18K bets)
- **Target_1.5**: not viable (33 bets, -36% ROI)
- **PA-level prediction**: wall at log-loss 1.412 regardless of features
- **Game simulation**: r≈0.2 ceiling (both bottom-up MC and top-down team-regression)
- Decision: ship the betting system; game simulation is dead-ended

## Relevant Files

| File | Purpose |
|------|---------|
| `src/mlb_ml_lab/evaluation/backtest.py` | `walk_forward_predict()`, `simulate_bets()`, `GamePrediction`, `BetResult` |
| `src/mlb_ml_lab/models/train.py` | `load_ensemble()`, `_build_model()`, `WalkForwardSplit` |
| `src/mlb_ml_lab/models/mlx_nn.py` | `MlxNNClassifier` — sklearn-compatible MLP on GPU |
| `src/mlb_ml_lab/models/sequence.py` | GRU, Hybrid, MultiTask, DCN, Transformer models (all MLX) |
| `src/mlb_ml_lab/cli/main.py` | CLI entry point (`backtest`, `bet`, `predict`, `train`) |
| `experiments/betting_strategy.py` | Live betting: generates/settles/tracks P&L |
| `experiments/war_calculator.py` | Hybrid WAR calculator (wOBA+park+baserunning+pos+replacement) |
| `experiments/advanced_metrics.py` | OPS+, wRC+, ISO, BABIP, BB/K%, WAR/162, player archetype classification |
| `experiments/compute_war.py` | CLI entry point for WAR + advanced metrics computation |
| `pipeline/benchmark_mlx.py` | MLX GPU training speed benchmarks |
| `data/models/ensemble_0_5/` | 4 trained ensemble components (gitignored) |

## Gotchas

- `.venv/` is a bare Python 3.13 venv (pip only). After `poetry install`, Poetry creates its own virtualenv; don't mix them.
- MLB Stats API at `statsapi.mlb.com` requires no key for read access. Data use subject to MLB copyright notice.
- FanGraphs has aggressive anti-scraping measures — don't depend on it.
- `experiments/` is gitignored — save exploratory scripts there.
