"""Walk-forward backtest comparing regime-switching vs baseline models.

Trains separate XGBoost models for "hot" (hit_rate_last_10 >= 0.4)
and "cold" regimes, then compares AUC and betting ROI against the
standard single-model baseline.

Usage:
    poetry run python pipeline/backtest_regime.py
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from mlb_ml_lab import load_feature_data
from mlb_ml_lab.models.regime import RegimeSwitchingClassifier
from mlb_ml_lab.models.train import (
    WalkForwardSplit,
    _build_model,
    _feature_columns,
    _merge_features_targets,
    NOISE_FEATURES,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
SEED = 42
N_SPLITS = 4
TARGET_COL = "target_0.5"
DECIMAL_ODDS = 1.909


def _to_array(
    merged: list[dict],
    feat_cols: list[str],
) -> np.ndarray:
    x = np.zeros((len(merged), len(feat_cols)), dtype=np.float64)
    for i, row in enumerate(merged):
        for j, c in enumerate(feat_cols):
            v = row.get(c)
            x[i, j] = float(v) if v is not None else float("nan")
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(x)
    return np.nan_to_num(x, nan=0.0)


def simulate(
    predictions: list[dict[str, Any]],
    min_prob: float = 0.55,
) -> dict[str, float]:
    bets = [p for p in predictions if p["prob"] >= min_prob]
    n = len(bets)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "roi": 0.0}
    wins = sum(1 for p in bets if p["actual"] == 1)
    losses = n - wins
    net = wins * (DECIMAL_ODDS - 1) - losses
    return {
        "n": n,
        "win_rate": round(wins / n, 4),
        "roi": round(net / n * 100, 2),
        "profit": round(net, 2),
    }


def run_regime(
    merged: list[dict],
    feat_cols: list[str],
    dates: list,
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    x_all = _to_array(merged, feat_cols)
    y_all = np.array([r[TARGET_COL] for r in merged], dtype=np.int32)

    splitter = WalkForwardSplit(n_splits=N_SPLITS)
    folds = splitter.split(dates)

    regime_col = feat_cols.index("hit_rate_last_10")
    regime_mask = x_all[:, regime_col] >= threshold

    preds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        model = RegimeSwitchingClassifier(
            base_model_type="xgb",
            regime_feature="hit_rate_last_10",
            threshold=threshold,
            random_state=SEED,
        )
        model.fit(
            x_all[train_idx], y_all[train_idx],
            feature_names=feat_cols,
        )
        probas = model.predict_proba(x_all[test_idx])[:, 1]
        for idx, prob in zip(test_idx, probas.tolist()):
            preds.append({
                "prob": round(float(prob), 4),
                "actual": int(y_all[idx]),
                "in_hot_regime": bool(regime_mask[idx]),
                "fold": fold_idx,
            })
    return preds


def run_baseline(
    merged: list[dict],
    feat_cols: list[str],
    dates: list,
    model_type: str = "xgb",
) -> list[dict[str, Any]]:
    x_all = _to_array(merged, feat_cols)
    y_all = np.array([r[TARGET_COL] for r in merged], dtype=np.int32)

    splitter = WalkForwardSplit(n_splits=N_SPLITS)
    folds = splitter.split(dates)

    preds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        model = _build_model(model_type, SEED)
        model.fit(x_all[train_idx], y_all[train_idx])
        probas = model.predict_proba(x_all[test_idx])[:, 1]
        for idx, prob in zip(test_idx, probas.tolist()):
            preds.append({
                "prob": round(float(prob), 4),
                "actual": int(y_all[idx]),
                "fold": fold_idx,
            })
    return preds


def main() -> None:
    print("=== Regime-Switching Backtest ===\n")

    print("Loading dataset...")
    feature_matrix, targets, _meta = load_feature_data(CACHED_DATASET)
    merged = _merge_features_targets(feature_matrix, targets)
    merged.sort(key=lambda r: r["date"])
    feat_cols = _feature_columns(merged, exclude=NOISE_FEATURES)
    dates = [r["date"] for r in merged]
    print(f"  {len(merged)} merged rows, {len(feat_cols)} features\n")

    # Check regime feature availability
    if "hit_rate_last_10" not in feat_cols:
        print("ERROR: hit_rate_last_10 feature not available. Aborting.")
        return

    regime_values = np.array([
        float(r.get("hit_rate_last_10", 0) or 0) for r in merged
    ])
    hot_pct = (regime_values >= 0.4).mean() * 100
    cold_pct = (regime_values < 0.4).mean() * 100
    print(f"  Hot  (hit_rate >= 0.4): {hot_pct:.1f}% of samples")
    print(f"  Cold (hit_rate <  0.4): {cold_pct:.1f}% of samples\n")

    # ── Baseline ──────────────────────────────────────────────────
    print("=" * 55)
    print("  BASELINE: single XGBoost")
    print("=" * 55)
    base_preds = run_baseline(merged, feat_cols, dates)
    base_y = np.array([p["actual"] for p in base_preds])
    base_p = np.array([p["prob"] for p in base_preds])
    base_auc = roc_auc_score(base_y, base_p)
    print(f"  AUC: {base_auc:.4f}")
    for thresh in (0.55, 0.65, 0.75):
        r = simulate(base_preds, min_prob=thresh)
        if r["n"]:
            print(
                f"  Thresh={thresh:.2f}: {r['n']:>6} bets, "
                f"win={r['win_rate']:.3f}, ROI={r['roi']:+.1f}%"
            )
    print()

    # ── Regime-switching (various thresholds) ──────────────────────
    results: list[dict] = []
    for thresh in (0.3, 0.4, 0.5):
        label = f"regime (hit_rate >= {thresh})"
        print("=" * 55)
        print(f"  {label}")
        print("=" * 55)
        preds = run_regime(merged, feat_cols, dates, threshold=thresh)
        y = np.array([p["actual"] for p in preds])
        p = np.array([p["prob"] for p in preds])
        auc = roc_auc_score(y, p)
        print(f"  AUC: {auc:.4f}")

        # Per-regime AUC
        regime_mask = np.array([q["in_hot_regime"] for q in preds])
        hot_actual = y[regime_mask]
        hot_prob = p[regime_mask]
        cold_actual = y[~regime_mask]
        cold_prob = p[~regime_mask]
        if len(np.unique(hot_actual)) > 1:
            hot_auc = roc_auc_score(hot_actual, hot_prob)
            print(f"    Hot regime AUC:  {hot_auc:.4f} ({len(hot_actual)} samples)")
        if len(np.unique(cold_actual)) > 1:
            cold_auc = roc_auc_score(cold_actual, cold_prob)
            print(f"    Cold regime AUC: {cold_auc:.4f} ({len(cold_actual)} samples)")

        for bt in (0.55, 0.65, 0.75):
            r = simulate(preds, min_prob=bt)
            if r["n"]:
                print(
                    f"  Thresh={bt:.2f}: {r['n']:>6} bets, "
                    f"win={r['win_rate']:.3f}, ROI={r['roi']:+.1f}%"
                )

        results.append({
            "threshold": thresh,
            "auc": round(auc, 4),
            "label": label,
        })
        print()

    # ── Summary ────────────────────────────────────────────────────
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<30} {'AUC':>6}")
    print(f"  {'-'*30} {'-'*6}")
    print(f"  {'Baseline XGB':<30} {base_auc:>6.4f}")
    for r in results:
        print(f"  {r['label']:<30} {r['auc']:>6.4f}")
    print()


if __name__ == "__main__":
    main()
