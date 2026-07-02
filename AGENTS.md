# mlb-ml-lab — MLB prediction models

Experimental models predicting whether MLB player hits clear 0.5 and 1.5 thresholds.

**Roadmap**: [`ROADMAP.md`](./ROADMAP.md)

## Stack

- Python 3.13, Poetry, `src/mlb_ml_lab/` package layout
- pytest (run: `poetry run pytest`)
- Type hints expected; no strict type checker configured yet
- All AI inference goes through local LM Studio at `http://127.0.0.1:1234/v1` — never call cloud APIs

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

## Conventions

- **No cloud AI APIs.** All model calls go through the local endpoint. Use `openai` SDK with `base_url = http://127.0.0.1:1234/v1`.
- **No MLB data library dependencies.** Reference `pybaseball`/`pybaseballstats`/`python-mlb-statsapi` for endpoint patterns but own the data layer. Thin `httpx` wrappers around `statsapi.mlb.com`.
- `data/` for raw/processed datasets (gitignored if large). Keep small samples in `tests/fixtures/`.
- `src/mlb_ml_lab/models/` for models; `src/mlb_ml_lab/features/` for feature engineering; `experiments/` for notebooks.
- Walk-forward validation (never random train/test split) — sports data is temporally dependent.
- Feature engineering lives in `src/mlb_ml_lab/features/` (part of the installable package). The `pipeline/` directory at project root is for model training/prediction only.

## Gotchas

- `.venv/` is a bare Python 3.13 venv (pip only). After `poetry install`, Poetry creates its own virtualenv; don't mix them.
- MLB Stats API at `statsapi.mlb.com` requires no key for read access. Data use subject to MLB copyright notice.
- FanGraphs has aggressive anti-scraping measures — don't depend on it.
