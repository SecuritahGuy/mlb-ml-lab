"""Hyperparameter tuning via random search inside walk-forward validation.

Usage:
    poetry run python pipeline/tune.py

Prints best parameters and AUC for each (model_type, target) combo.
"""

from __future__ import annotations

import warnings

from sklearn.exceptions import ConvergenceWarning

from mlb_ml_lab import load_feature_data
from mlb_ml_lab.models.train import (
    DEFAULT_PARAM_GRIDS,
    tune_hyperparameters,
)

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
N_TRIALS = 12
N_SPLITS = 4

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def main() -> None:
    print(f"Loading cached dataset from {CACHED_DATASET}...")
    feature_matrix, targets, meta = load_feature_data(CACHED_DATASET)
    print(f"  {len(feature_matrix)} feature rows, {len(targets)} target rows")
    print(f"  {meta.get('feature_count', '?')} feature columns\n")

    for model_type in ("lgb", "xgb"):
        for target_col in ("target_0.5", "target_1.5"):
            print(f"{'=' * 60}")
            print(f"Tuning {model_type.upper()} — {target_col}")
            print(f"{'=' * 60}")

            result = tune_hyperparameters(
                feature_matrix,
                targets,
                target_col=target_col,
                model_type=model_type,
                param_grid=DEFAULT_PARAM_GRIDS[model_type],
                n_trials=N_TRIALS,
                n_splits=N_SPLITS,
                metric="auc",
                seed=42,
            )

            if "error" in result:
                print(f"  ERROR: {result['error']}\n")
                continue

            print(f"  Best params:  {result['best_params']}")
            print(f"  Best AUC:     {result['best_score']:.4f}")
            print(f"  Best std:     {result['best_std']:.4f}")
            print(f"  Trials:       {result['n_trials']}")
            print()

            # Show top 5 trials
            sorted_trials = sorted(
                result["trials"],
                key=lambda t: t["auc"],
                reverse=True,
            )
            print("  Top 5 trials:")
            for i, t in enumerate(sorted_trials[:5]):
                print(f"    {i + 1}. AUC={t['auc']:.4f} params={t['params']}")
            print()


if __name__ == "__main__":
    main()
