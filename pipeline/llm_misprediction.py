"""Analyze model mispredictions using a local LLM.

Runs walk-forward ensemble predictions on the cached dataset,
samples high-confidence mispredictions, and asks a local LLM
to identify patterns in the errors.

Usage:
    poetry run python pipeline/llm_misprediction.py
    poetry run python pipeline/llm_misprediction.py --provider ollama --model gemma4:e4b

Requires LM Studio (http://127.0.0.1:1234/v1) or Ollama (http://localhost:11434/v1).
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from mlb_ml_lab import PlayerIdResolver, load_feature_data, load_game_logs
from mlb_ml_lab.evaluation.backtest import GamePrediction, walk_forward_predict

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

CACHED_DATASET = "data/datasets/full_2021_2026_30teams"
N_SAMPLES = 30
RANDOM_SEED = 42

PROVIDERS = {
    "lm-studio": {"base_url": "http://127.0.0.1:1234/v1", "default_model": ""},
    "ollama": {"base_url": "http://localhost:11434/v1", "default_model": "gemma4:e4b"},
}

KEY_FEATURES = [
    "hits_last_10",
    "hit_rate_last_10",
    "player_age",
    "years_experience",
    "career_avg",
    "career_obp",
    "career_slg",
    "opp_era",
    "opp_k_per_9",
    "opp_pitcher_era",
    "opp_pitcher_k_per_9",
    "is_home",
    "rest_days",
    "park_wOBA",
    "hitting_streak",
    "bullpen_era",
    "league_avg",
]


def _build_feature_index(
    feature_matrix: list[dict],
) -> dict[tuple[int, int], dict]:
    return {(r["player_id"], r["game_pk"]): r for r in feature_matrix}


def sample_mispredictions(
    predictions: list[GamePrediction],
    feature_index: dict[tuple[int, int], dict],
    n_samples: int = 30,
) -> list[tuple[GamePrediction, dict]]:
    fp: list[tuple[GamePrediction, dict]] = []
    fn: list[tuple[GamePrediction, dict]] = []
    tp: list[tuple[GamePrediction, dict]] = []
    tn: list[tuple[GamePrediction, dict]] = []

    for gp in predictions:
        if gp.target_col != "target_0.5":
            continue
        feat = feature_index.get((gp.player_id, gp.game_pk))
        if feat is None:
            continue
        entry = (gp, feat)
        if gp.predicted_prob >= 0.65:
            if gp.actual == 0:
                fp.append(entry)
            else:
                tp.append(entry)
        elif gp.predicted_prob <= 0.35:
            if gp.actual == 1:
                fn.append(entry)
            else:
                tn.append(entry)

    rng = np.random.default_rng(RANDOM_SEED)
    per_group = max(1, n_samples // 4)
    sampled: list[tuple[GamePrediction, dict]] = []
    for pool in [fp, fn, tp, tn]:
        pool_sorted = sorted(pool, key=lambda x: abs(x[0].predicted_prob - 0.5))
        if len(pool_sorted) > per_group:
            idx = rng.choice(len(pool_sorted), size=per_group, replace=False)
            sampled.extend(pool_sorted[int(i)] for i in idx)
        else:
            sampled.extend(pool_sorted)

    rng.shuffle(sampled)
    return sampled


def _fmt(feat: dict, key: str) -> str:
    val = feat.get(key)
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def build_prompt(
    samples: list[tuple[GamePrediction, dict]],
    resolver: PlayerIdResolver,
) -> str:
    lines: list[str] = [
        "You are analyzing a machine learning model that predicts whether an MLB player will",
        "get at least 1 hit in a game. Below are out-of-sample predictions where the model",
        "was either confidently right or confidently wrong.",
        "",
        "Features used by the model:",
        "- Rolling hit stats: hits_last_10 (total hits in last 10 games),",
        "  hit_rate_last_10 (fraction of games with 1+ hit in last 10)",
        "- Player context: player_age, years_experience, career_avg/obp/slg",
        "- Opponent pitching: opp_era, opp_k_per_9 (team level),",
        "  opp_pitcher_era, opp_pitcher_k_per_9 (starting pitcher level)",
        "- Game context: is_home (1=home, 0=away), rest_days, park_wOBA",
        "- Momentum: hitting_streak (consecutive games with 1+ hit)",
        "- bullpen_era, league_avg (league-wide batting average)",
        "",
        "For each case:",
        "- predicted_prob: model's confidence (0-1) that player gets 1+ hits",
        "- actual: 1 if player got 1+ hits, 0 if not",
        "- 'WRONG' = model was confidently wrong, 'CORRECT' = model was right",
        "",
        "Analyze these cases and identify:",
        "1. Common patterns in the mispredictions",
        "2. What features seem to drive the model's errors",
        "3. What additional context or features could help",
        "4. Specific player types or game situations where the model fails",
        "",
    ]

    for i, (gp, feat) in enumerate(samples, 1):
        player = resolver.resolve(gp.player_id, source="mlbam")
        if player:
            name = (
                f"{player.get('name_first', '')} {player.get('name_last', '')}"
            ).strip()
        else:
            name = f"player_{gp.player_id}"

        label = "CORRECT" if gp.actual == 1 else "WRONG"
        lines.append(f"Case {i}: {name}")
        lines.append(
            f"  date={gp.date}  predicted_prob={gp.predicted_prob:.3f}  "
            f"actual={gp.actual} ({label})"
        )
        for k in KEY_FEATURES:
            lines.append(f"  {k}={_fmt(feat, k)}")
        lines.append("")

    lines.append(
        "Give me your analysis of the patterns. Be specific and reference "
        "feature values."
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze model mispredictions using a local LLM"
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: provider-specific, e.g. gemma4:e4b for ollama)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    provider = PROVIDERS[args.provider]
    model_name = args.model or provider["default_model"]
    base_url = provider["base_url"]

    print(f"=== LLM Misprediction Analysis (provider={args.provider}) ===")
    print()

    print("[1/5] Loading cached dataset...")
    try:
        feature_matrix, targets, _meta = load_feature_data(CACHED_DATASET)
        game_logs = load_game_logs(CACHED_DATASET)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {CACHED_DATASET}")
        print("Run `mlb fetch` first to generate cached data.")
        sys.exit(1)
    print(
        f"       {len(feature_matrix)} feature rows, "
        f"{len(targets)} target rows, "
        f"{len(game_logs)} game logs"
    )

    print("[2/5] Running walk-forward ensemble (LR+XGB+RF+LGBM)...")
    print("       This may take a minute...")
    predictions = walk_forward_predict(
        feature_matrix,
        targets,
        target_col="target_0.5",
        model_type=["lr", "xgb", "rf", "lgb"],
        n_splits=4,
        seed=42,
    )
    print(f"       {len(predictions)} out-of-sample predictions")

    if predictions:
        y_true = np.array([gp.actual for gp in predictions])
        y_prob = np.array([gp.predicted_prob for gp in predictions])
        auc = roc_auc_score(y_true, y_prob)
        print(f"       Overall AUC: {auc:.4f}")

        fp_count = sum(
            1 for gp in predictions if gp.predicted_prob >= 0.65 and gp.actual == 0
        )
        fn_count = sum(
            1 for gp in predictions if gp.predicted_prob <= 0.35 and gp.actual == 1
        )
        print(f"       High-confidence FPs (prob>=0.65, actual=0): {fp_count}")
        print(f"       High-confidence FNs (prob<=0.35, actual=1): {fn_count}")

    print("[3/5] Initializing player ID resolver...")
    resolver = PlayerIdResolver()
    try:
        resolver.sync()
        print(f"       Resolver loaded ({len(resolver)} players)")
    except (OSError, ConnectionError) as exc:
        print(f"       WARNING: Failed to sync Chadwick register: {exc}")
        print(f"       Using bundled sample ({len(resolver)} players)")

    print("[4/5] Building feature index...")
    feature_index = _build_feature_index(feature_matrix)
    print(f"       Index built ({len(feature_index)} unique (player, game) pairs)")
    print("       Sampling mispredictions...")
    samples = sample_mispredictions(predictions, feature_index, n_samples=N_SAMPLES)
    print(f"       {len(samples)} cases selected")

    print(f"[5/5] Calling {args.provider} (model={model_name})...")
    prompt = build_prompt(samples, resolver)

    with open("data/llm_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    try:
        from openai import OpenAI, APIConnectionError

        client = OpenAI(base_url=base_url, api_key="not-needed")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192,
        )
        analysis = response.choices[0].message.content
    except APIConnectionError:
        print(f"\nERROR: Cannot reach {args.provider} at {base_url}")
        print("Make sure it is running with a model loaded.")
        print("Prompt saved to data/llm_prompt.txt for manual review")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("LLM ANALYSIS OF MISPREDICTIONS")
    print("=" * 60)
    print(analysis)

    with open("data/llm_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis)
    print("\nAnalysis saved to data/llm_analysis.txt")
    print("Prompt saved to data/llm_prompt.txt")


if __name__ == "__main__":
    main()
