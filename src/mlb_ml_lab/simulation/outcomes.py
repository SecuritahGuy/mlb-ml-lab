from __future__ import annotations

import json
import os
from collections import defaultdict, Counter



OUTCOME_CLASSES = [
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "strikeout",
    "other",
]


def load_pbp_dataset(path: str) -> list[dict]:
    pas: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pas.append(json.loads(line))
    return pas


def compute_league_averages(
    pas: list[dict],
) -> dict[str, float]:
    """Compute MLB-wide PA outcome distribution."""
    counter: Counter[str] = Counter()
    for pa in pas:
        counter[pa["event_type"]] += 1
    total = sum(counter.values())
    return {cls: counter.get(cls, 0) / total for cls in OUTCOME_CLASSES}


def compute_player_outcomes(
    pas: list[dict],
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute per-player outcome distributions.

    Returns:
        ``{"batter": {player_id: {outcome: prob}}, "pitcher": ...}``
    """
    batter_outcomes: dict[int, Counter[str]] = defaultdict(Counter)
    pitcher_outcomes: dict[int, Counter[str]] = defaultdict(Counter)

    for pa in pas:
        et = pa["event_type"]
        batter_outcomes[pa["batter_id"]][et] += 1
        pitcher_outcomes[pa["pitcher_id"]][et] += 1

    def _normalize(counter: Counter) -> dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {cls: 0.0 for cls in OUTCOME_CLASSES}
        return {cls: counter.get(cls, 0) / total for cls in OUTCOME_CLASSES}

    return {
        "batter": {pid: _normalize(c) for pid, c in batter_outcomes.items()},
        "pitcher": {pid: _normalize(c) for pid, c in pitcher_outcomes.items()},
    }


def blend_outcomes(
    batter_dist: dict[str, float],
    pitcher_dist: dict[str, float],
    league_avg: dict[str, float],
    batter_weight: float = 0.5,
    pitcher_weight: float = 0.3,
    league_weight: float = 0.2,
) -> dict[str, float]:
    """Blend batter, pitcher, and league outcome distributions.

    Args:
        batter_dist: Batter's historical outcome distribution.
        pitcher_dist: Pitcher's historical outcome distribution.
        league_avg: MLB-wide outcome distribution.
        batter_weight: Weight for batter's history.
        pitcher_weight: Weight for pitcher's history.
        league_weight: Weight for league average.

    Returns:
        Blended probability distribution.
    """
    blended: dict[str, float] = {}
    for cls in OUTCOME_CLASSES:
        blended[cls] = (
            batter_dist[cls] * batter_weight
            + pitcher_dist[cls] * pitcher_weight
            + league_avg[cls] * league_weight
        )
    total = sum(blended.values())
    if total > 0:
        for cls in blended:
            blended[cls] /= total
    return blended


def save_outcome_distributions(
    output_dir: str,
    league_avg: dict[str, float],
    player_outcomes: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "league_avg.json"), "w") as f:
        json.dump(league_avg, f, indent=2)
    for role, data in player_outcomes.items():
        fname = f"{role}_outcomes.json"
        ser = {str(pid): dist for pid, dist in data.items()}
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(ser, f, indent=2)


def load_outcome_distributions(
    output_dir: str,
) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    with open(os.path.join(output_dir, "league_avg.json")) as f:
        league_avg: dict[str, float] = json.load(f)
    with open(os.path.join(output_dir, "batter_outcomes.json")) as f:
        batter_raw: dict[str, dict[str, float]] = json.load(f)
    with open(os.path.join(output_dir, "pitcher_outcomes.json")) as f:
        pitcher_raw: dict[str, dict[str, float]] = json.load(f)
    batter_outcomes = {int(k): v for k, v in batter_raw.items()}
    pitcher_outcomes = {int(k): v for k, v in pitcher_raw.items()}
    return league_avg, batter_outcomes, pitcher_outcomes
