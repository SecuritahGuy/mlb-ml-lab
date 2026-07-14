"""Compare GRU hidden dim sizes on the last walk-forward fold."""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from sklearn.impute import SimpleImputer

from mlb_ml_lab import PlayerGameLog, load_feature_data, load_game_logs
from mlb_ml_lab.models.evaluate import classification_metrics
from mlb_ml_lab.models.sequence import (
    SEQUENCE_LEN,
    _feat_vec,
    build_hybrid_sequences,
    predict_hybrid_model,
    train_hybrid_model,
)
from mlx.utils import tree_flatten

from mlb_ml_lab.models.train import _build_model, _feature_columns

CACHED_DATASET = "data/datasets/full_2016_2026_30teams"
TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
SEED = 42

_TUNED_XGB = {
    "n_estimators": 500, "max_depth": 5, "learning_rate": 0.01,
    "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1,
}


def _merge_rows(feat_rows, tgt_rows, target_col):
    merged = []
    for fr, tr in zip(feat_rows, tgt_rows):
        row = dict(fr)
        row[target_col] = tr[target_col]
        merged.append(row)
    return merged


def _extract_xgb(merged_rows, target_col, cols=None, imputer=None):
    if cols is None:
        cols = [c for c in _feature_columns(merged_rows) if c != target_col]
    y = np.array([r[target_col] for r in merged_rows], dtype=np.int32)
    x = np.zeros((len(merged_rows), len(cols)), dtype=np.float64)
    for i, r in enumerate(merged_rows):
        for j, c in enumerate(cols):
            v = r.get(c)
            x[i, j] = float(v) if v is not None else float("nan")
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        x = imputer.fit_transform(x)
    else:
        x = imputer.transform(x)
    x = np.nan_to_num(x, nan=0.0)
    return x, y, cols, imputer


# ── Load data ──────────────────────────────────────────────────────
print("Loading data...")
raw_logs = load_game_logs(CACHED_DATASET)
feature_matrix, targets_list, meta = load_feature_data(CACHED_DATASET)

game_logs: list[PlayerGameLog] = []
for d in raw_logs:
    game_logs.append(PlayerGameLog(**{
        k: v for k, v in d.items() if k in PlayerGameLog.__dataclass_fields__
    }))

targets: list[dict] = targets_list
features: list[dict] = feature_matrix

# Index for alignment
feat_by_key = {(f["player_id"], f["game_pk"]): f for f in features}
tgt_by_key = {(t["player_id"], t["game_pk"]): t for t in targets}

aligned_logs = []
aligned_feats = []
aligned_tgts = []
for lg in game_logs:
    key = (lg.player_id, lg.game_pk)
    fr = feat_by_key.get(key)
    tr = tgt_by_key.get(key)
    if fr is not None and tr is not None:
        aligned_logs.append(lg)
        aligned_feats.append(fr)
        aligned_tgts.append(tr)

# Fold: train ≤2024, test=2025
train_cutoff = 2024
test_season = 2025

train_logs = [lg for lg, t in zip(aligned_logs, aligned_tgts)
              if int(t["date"][:4]) <= train_cutoff]
test_logs = [lg for lg, t in zip(aligned_logs, aligned_tgts)
             if int(t["date"][:4]) == test_season]
train_tgt = [t for t in aligned_tgts if int(t["date"][:4]) <= train_cutoff]
test_tgt = [t for t in aligned_tgts if int(t["date"][:4]) == test_season]
train_feat = [f for f, t in zip(aligned_feats, aligned_tgts)
              if int(t["date"][:4]) <= train_cutoff]
test_feat = [f for f, t in zip(aligned_feats, aligned_tgts)
             if int(t["date"][:4]) == test_season]

print(f"Train ≤{train_cutoff}: {len(train_logs)} logs")
print(f"Test  {test_season}: {len(test_logs)} logs")

# Build sequences once (shared)
Xs_tr, Xc_tr, y_tr, sm, ss, fm, fs = build_hybrid_sequences(
    train_logs, train_feat, train_tgt, target_col="target_0.5",
)
Xs_te, Xc_te, y_te, _, _, _, _ = build_hybrid_sequences(
    test_logs, test_feat, test_tgt,
    stats_mean=sm, stats_std=ss, feat_mean=fm, feat_std=fs,
    target_col="target_0.5",
)
print(f"Hybrid train: {len(Xs_tr)}, test: {len(Xs_te)}")

# XGB baseline
train_merged = _merge_rows(train_feat, train_tgt, "target_0.5")
test_merged = _merge_rows(test_feat, test_tgt, "target_0.5")
x_train, y_tr_xgb, _xgb_cols, xgb_imputer = _extract_xgb(train_merged, "target_0.5")
x_test, y_te_xgb, _, _ = _extract_xgb(test_merged, "target_0.5", cols=_xgb_cols, imputer=xgb_imputer)

xgb_model = _build_model("xgb", SEED, params=_TUNED_XGB)
xgb_model.fit(x_train, y_tr_xgb)
xgb_te_proba = xgb_model.predict_proba(x_test)[:, 1]
xgb_auc = classification_metrics(y_te_xgb.tolist(), (xgb_te_proba > 0.5).astype(int).tolist(), xgb_te_proba.tolist()).get("auc", 0)
print(f"XGB AUC: {xgb_auc:.4f}")

# Map XGB probas for ensemble
xgb_proba_map = {(r["player_id"], r["game_pk"]): float(xgb_te_proba[i]) for i, r in enumerate(test_merged)}

# Build key index for hybrid test rows
_keys_te = []
_feat_idx = {(f["player_id"], f["game_pk"]): f for f in test_feat}
_grouped: dict = defaultdict(list)
for i, lg in enumerate(test_logs):
    _grouped[(lg.player_id, str(lg.season))].append((i, lg))
for (pid, season), entries in _grouped.items():
    entries.sort(key=lambda e: e[1].date)
    indices = [e[0] for e in entries]
    vecs = [_feat_vec(e[1]) for e in entries]
    for pos in range(SEQUENCE_LEN, len(vecs)):
        idx = indices[pos]
        lg = test_logs[idx]
        if _feat_idx.get((lg.player_id, lg.game_pk)):
            _keys_te.append((lg.player_id, lg.game_pk))

# ── Compare hidden dims ──────────────────────────────────────────
for hidden_dim, n_layers in [(64, 2), (128, 2), (256, 2), (128, 3)]:
    label = f"hidden={hidden_dim}, L={n_layers}"
    print(f"\n  Train: {label}...")
    t0 = time.time()

    model, meta = train_hybrid_model(
        Xs_tr, Xc_tr, y_tr,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=60,
        batch_size=512,
        verbose=False,
    )
    train_time = time.time() - t0
    hybrid_te_proba = predict_hybrid_model(model, Xs_te, Xc_te)
    hybrid_pred = (hybrid_te_proba > 0.5).astype(int)
    hybrid_metrics = classification_metrics(
        y_te.tolist(), hybrid_pred.tolist(), hybrid_te_proba.tolist(),
    )
    hybrid_auc = hybrid_metrics.get("auc", 0)

    # Ensemble
    ensemble_probas = []
    for i, (pid, gpk) in enumerate(_keys_te):
        hp = float(hybrid_te_proba[i])
        xp = xgb_proba_map.get((pid, gpk), hp)
        ensemble_probas.append((hp + xp) / 2.0)
    ensemble_pred = (np.array(ensemble_probas) > 0.5).astype(int)
    ens_metrics = classification_metrics(
        y_te.tolist(), ensemble_pred.tolist(), ensemble_probas,
    )
    ens_auc = ens_metrics.get("auc", 0)

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"    Hybrid AUC={hybrid_auc:.4f}  Ensemble AUC={ens_auc:.4f}  "
          f"Train time={train_time:.1f}s  Params={n_params:,}")
