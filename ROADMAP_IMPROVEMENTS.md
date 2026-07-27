# mlb-ml-lab — Improvements Roadmap

Higher-value directions beyond the current baseline: better tabular learners,
unified probabilistic target formulation, and stricter calibration validation.

Derived from an external research review of the project (July 2026).

---

## Current State Assessment

### What works well

- Expanding-window validation (not random split), temporal gap, no-lookahead
- 6 model families: LR, RF, XGBoost, LightGBM, CatBoost, MLX MLP
- Ensemble AUC ~0.639 (uniform averaging ≈ stacking)
- Feature registry, imputation, hyperparameter search, model persistence
- CLI workflows (fetch/train/predict/backtest/bet/tune/e2e)
- Isotonic calibration, ECE tracking
- 27 registered feature extractors, ~166 columns

### Key findings from the review

1. **CatBoost is not fully tested** — the feature pipeline drops non-numeric columns and excludes `player_id`, preventing CatBoost from using its core strength (leakage-resistant high-cardinality categorical handling).
2. **Sequence model failures are informative, not disappointing** — GRU/Transformer AUC (0.568–0.583) show that raw 15-game box-score sequences lack unique signal beyond engineered rolling features. Process-level PA data (pitch velocity, type, location, swing decisions) would be needed for a fair sequence-model test.
3. **Reported ROI (22%–44%) is hypothetical** — player hit-prop lines are unavailable from free sources. The backtest assumes -110 odds. Win-rate ranking is real, but edge vs actual market prices is unvalidated.
4. **Calibration ECE (0.001–0.002) deserves an explicit leakage audit** — verify each calibrator is fit only on pre-evaluation predictions, with independent samples per bin, and tested with adaptive-bin ECE + calibration intercept/slope (not fixed-bin ECE alone).

---

## Sprint 1: Low-Cost, High-Information Diversity

1. [ ] **Add Explainable Boosting Machine** (`interpret` framework)
     - Main effects only → 5, 10, 20 interactions
     - Monotonic constraints on clearly directional features
     - Bagged EBM seed ensemble
     - Optimize for log loss, Brier, AUC, calibration slope
     - *Effort: Low. Upside: Medium-High. Interpretability: Very high.*
2. [ ] **Add Extra Trees** (sklearn) — cheap ensemble diversity
     - 500–1500 trees, `min_samples_leaf` 5–100, `max_features` 0.3–1.0
     - Check out-of-fold residual correlation with XGBoost/LGBM
     - *Effort: Very low. Upside: Low-Medium. Diversity: Potentially useful.*
3. [ ] **Add HistGradientBoostingClassifier** (sklearn) — fast histogram-based boosting, native missing-value handling, monotonic constraints
     - *Effort: Very low. Upside: Low. Benchmark value: High.*
4. [ ] **Audit calibration splitting** — verify per-season isotonic calibrators are:
     - Fit only on predictions from before the evaluation period
     - Never fit on the same season/fold where ECE is reported
     - Evaluated on ≥~50 independent samples per probability bin
     - Tested with adaptive-bin ECE + calibration intercept/slope
     - Compared with constant base-rate model
     - Consider nested pattern: train seasons t-2, calibrate on t-1, evaluate on t
5. [ ] **Produce residual-correlation matrix** across all models (LR, RF, XGB, LGBM, CB, MLX MLP, EBM, ET, HGB)
6. [ ] **Compare uniform averaging vs constrained nonnegative blending** (e.g. ensemble weights learned via CV on log loss)

---

## Sprint 2: Unlock CatBoost Properly ✅

7. [x] **Build dedicated CatBoost adapter** that preserves categorical columns
8. [x] **Categorical fields implemented** (IdentityFeatures + CATEGORICAL_FEATURES set):
     - `team_id`, `opponent_id`, `venue_id` (identity/park extractors)
     - `month`, `position_code` (identity extractor)
     - `opp_pitcher_id`, `hp_umpire_id` (existing high-cardinality, now categorical)
     - `position_cat`, `is_home`, `il_flag`, `same_hand_advantage`
     - `bats_left`, `bats_right`, `throws_left`, `throws_right`
     - Future: `player_id` excluded from numeric matrix, available for CatBoost via categorical path
9. [x] **Pipeline changes**: `_build_catboost_matrix()` creates combined numeric+categorical matrix
     - Object-array approach preserves int32 dtype for categorical columns
     - Missing categorical values encoded as -1 (CatBoost-native missing handling)
     - Integrated into `walk_forward_predict()`, `train_baselines()`, `tune_hyperparameters()`, `train_final()`
     - Non-CatBoost models unaffected (continue with numeric-only matrix)
10. [x] **Backtest results** (on cached dataset without new identity features):
     - **Categorical CatBoost standalone: AUC 0.6384** (+0.0012 vs numeric-only 0.6372)
     - **Ensemble (LR+XGB+RF+LGBM+CB with categorical CB): AUC 0.6378** (essentially unchanged)
     - The gain is from treating existing features (position_cat, opp_pitcher_id, etc.) as categorical
     - Larger lift expected after rebuilding dataset with `team_id`, `opponent_id`, `month`, `venue_id`
11. [ ] **Evaluate cold-start cohorts independently**:
     - Established players (≥200 prior games)
     - Players with <50 prior games
     - Rookies (no prior MLB games)
     - Players changing teams
     - Previously unseen pitchers
12. [ ] **Assess memorization vs generalization** — does categorical player ID help on rookies?

*Effort: Medium. Upside: High (+0.12 bps AUC lift from existing features alone). Main risk: Memorization and cold-start degradation.*

---

## Sprint 3: Reformulate the Prediction Target

13. [ ] **Build multi-class hit-count labels**: 0, 1, 2, 3+ hits
14. [ ] **Test ordinal multiclass classifier** — predictions automatically maintain `P(H≥2) ≤ P(H≥1)`
15. [ ] **Test Poisson / negative-binomial count model** (LightGBM Poisson, XGBoost count, `statsmodels` GLM)
16. [ ] **Test hurdle model** — P(hit ≥ 1) × conditional distribution of additional hits
17. [ ] **Derive both 0.5 and 1.5 probabilities from one distribution**
18. [ ] **Verify logical consistency**: `P(H≥2) ≤ P(H≥1)` for every prediction (ordinal models guarantee this)
19. [ ] **Evaluate impact on target 1.5** — the current 33-bet / -36% ROI situation is the primary motivation

*Effort: Medium. Upside: High. Scientific value: Very high.*

---

## Sprint 4: Focused Neural Benchmark

20. [ ] **Add PyTorch Tabular** as a dependency
21. [ ] **Test GANDALF** — gated feature-learning, well-aligned with medium tabular datasets
22. [ ] **Test FT-Transformer** — column-as-token formulation (more relevant than temporal sequence Transformer)
23. [ ] **Test NODE** (Neural Oblivious Decision Ensembles) — bridges neural and tree models
24. [ ] **Run all through identical**: train/test dates, feature sets, seeds, early stopping, probability metrics
25. [ ] **Stop if none beats MLX MLP on log loss or adds ensemble diversity**

*Effort: Medium-High. Upside: Medium. Important: Do not turn into a large open-ended architecture search.*

---

## Framework Improvements

### Optuna for temporal hyperparameter optimization

- [ ] Replace/add to random-search with Optuna
- [ ] Conditional parameter spaces, trial pruning, persistent storage
- [ ] Multiobjective: log loss, AUC, Brier, calibration slope, inference cost
- [ ] *Don't tune for hypothetical ROI until real odds are available.*

### Experiment tracking (lightweight)

- [ ] Structured experiment records (Parquet/JSON or MLflow)
- [ ] Per-run: git commit, dataset hash, feature set, date boundaries, model params, fold metrics, calibration params, training time, peak memory

### SHAP + residual analysis

- [ ] SHAP for XGBoost, LightGBM, CatBoost
- [ ] Pair with **error cohorts**, not only global feature importance
- [ ] Analyze false/costly predictions by: batter experience, lineup position, home/away, SP quality, park, month, favorite/underdog, probability decile, missing-data pattern

### Better calibration suite

- [ ] Add: Platt scaling, beta calibration, temperature scaling, Venn-Abers prediction
- [ ] Report: calibration intercept, calibration slope, adaptive ECE, max calibration error, Brier decomposition
- [ ] Reliability diagrams with confidence intervals
- [ ] One independent calibration season or rolling calibration window

---

## Lower-Priority Directions

| Model | Why | Effort | Upside |
|-------|-----|--------|--------|
| Hierarchical Bayesian (PyMC/Bambi) | Partial pooling for player/pitcher/park; explicit uncertainty; strong scientific baseline | High | Medium |
| TabPFN (tabular foundation model) | Residual specialist on difficult cases; high research value | Medium | Uncertain |
| NGBoost (probabilistic gradient boosting) | Full conditional distributions for expected hits, PA, total bases | Medium | Experimental |
| Additional GRU/LSTM/Transformer variants | Low expected return without process-level PA sequence data | Medium | Low |

---

## Bottom Line

> **Categorical CatBoost + EBM + a unified count-distribution target, evaluated with nested temporal calibration.**

That combination has the best chance of improving both discrimination and the trustworthiness of resulting probabilities. The project already demonstrates that boosted tabular models beat generic deep sequence learning for this dataset. The next strong moves are not larger Transformers.

---

## Conventions

- Each item is a checkbox — mark `[x]` when implemented and tested.
- Model experiments should run through the existing walk-forward pipeline with shared train/test dates, feature sets, and metrics.
- All AI inference stays local (no cloud APIs).
- Experiments directory (`experiments/`) is gitignored — save exploratory scripts there.
- After each sprint, update `AGENTS.md` with new commands/config.
- After each sprint, update `ROADMAP.md`"Phase N" if the direction becomes permanent.
