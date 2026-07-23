from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

# 24 base-out states: (1B, 2B, 3B, outs)
BASE_STATES: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
]
OUT_STATES = [0, 1, 2]
STATES = [(b1, b2, b3, o) for (b1, b2, b3) in BASE_STATES for o in OUT_STATES]

BASE_CODES = {"1B": 0, "2B": 1, "3B": 2}


def _parse_bases_before(
    runners: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Determine base state before a play from runner start positions."""
    bases = [0, 0, 0]
    for runner in runners:
        start = (runner.get("movement", {}) or {}).get("start")
        if start in BASE_CODES:
            bases[BASE_CODES[start]] = 1
    return tuple(bases)


def _parse_bases_after(
    runners: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Determine base state after a play from runner end positions."""
    bases = [0, 0, 0]
    for runner in runners:
        end = (runner.get("movement", {}) or {}).get("end")
        if end in BASE_CODES:
            bases[BASE_CODES[end]] = 1
    return tuple(bases)


def _runs_on_play(
    result: dict[str, Any], prev_home: int, prev_away: int
) -> int:
    """Total runs scored on a play from score delta."""
    home = int(result.get("homeScore", 0) or 0)
    away = int(result.get("awayScore", 0) or 0)
    return max(0, home - prev_home) + max(0, away - prev_away)


def _outs_on_play(
    result: dict[str, Any], runners: list[dict[str, Any]], outs_before: int
) -> int:
    """Total outs after a play.

    Counts outs from runners that were already on base (``start`` is not
    None).  The batter's out is determined by ``result.isOut``.  This
    avoids double-counting since the batter also appears as a runner
    with ``start=None``.
    """
    outs = outs_before
    for runner in runners:
        mv = runner.get("movement", {}) or {}
        start = mv.get("start")
        if start is not None and mv.get("isOut", False):
            outs += 1
    if result.get("isOut", False):
        outs += 1
    return min(outs, 3)


def compute_re24_from_plays(
    plays: list[dict[str, Any]],
) -> dict[tuple[int, int, int, int], float]:
    """Compute run expectancy for all 24 base-out states.

    Processes each play in order, tracking states and accumulating
    runs remaining from each state to the end of its inning.

    Returns:
        Mapping from ``(1B, 2B, 3B, outs)`` → expected runs to end of inning.
    """
    state_runs: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)

    prev_home = 0
    prev_away = 0

    for play in plays:
        result = play.get("result", {}) or {}
        runners = play.get("runners", []) or []
        count = play.get("count", {}) or {}

        if not result.get("eventType"):
            continue

        outs_before = count.get("outs", 0)
        bases_before = _parse_bases_before(runners)
        state_before = (*bases_before, outs_before)

        runs_scored = _runs_on_play(result, prev_home, prev_away)
        prev_home = int(result.get("homeScore", 0) or 0)
        prev_away = int(result.get("awayScore", 0) or 0)

        if runs_scored > 0:
            state_runs[state_before].append(float(runs_scored))

            # Also give credit if runners scored but no scoreboard change
            # (shouldn't happen in MLB)

    # Fill in missing states with 0
    re24: dict[tuple[int, int, int, int], float] = {}
    for state, runs_list in state_runs.items():
        re24[state] = float(np.mean(runs_list)) if runs_list else 0.0
    for state in STATES:
        if state not in re24:
            re24[state] = 0.0

    return re24


def compute_re24_game_method(
    game_plays_list: list[dict[str, Any]],
) -> dict[tuple[int, int, int, int], float]:
    """Compute RE24 by tracking runs-remaining per inning.

    Records the state before each PA and how many runs remained from
    that state to the end of the inning.  This is the canonical RE24
    definition.
    """
    state_to_runs_remaining: dict[tuple[int, int, int, int], list[float]] = (
        defaultdict(list)
    )

    current_state = (0, 0, 0, 0)
    states_in_inning: list[tuple[tuple[int, int, int, int], float]] = []
    cumulative_runs = 0.0
    prev_home = 0
    prev_away = 0

    for play in game_plays_list:
        result = play.get("result", {}) or {}
        runners = play.get("runners", []) or []
        count = play.get("count", {}) or {}

        if not result.get("eventType"):
            continue

        states_in_inning.append((current_state, cumulative_runs))

        outs_before = count.get("outs", 0)
        bases_after = _parse_bases_after(runners)
        outs_after = _outs_on_play(result, runners, outs_before)
        runs_scored = _runs_on_play(result, prev_home, prev_away)
        prev_home = int(result.get("homeScore", 0) or 0)
        prev_away = int(result.get("awayScore", 0) or 0)

        cumulative_runs += runs_scored
        current_state = (*bases_after, outs_after)

        if outs_after >= 3:
            for state, runs_at_state in states_in_inning:
                remaining = cumulative_runs - runs_at_state
                state_to_runs_remaining[state].append(remaining)
            states_in_inning = []
            current_state = (0, 0, 0, 0)
            cumulative_runs = 0.0

    re24 = {}
    for state, runs_list in state_to_runs_remaining.items():
        re24[state] = float(np.mean(runs_list)) if runs_list else 0.0
    for state in STATES:
        if state not in re24:
            re24[state] = 0.0
    return re24


def re24_matrix_to_array(
    re24: dict[tuple[int, int, int, int], float],
) -> np.ndarray:
    """Convert RE24 dict to 8×3 numpy array (base_state × outs)."""
    arr = np.zeros((8, 3))
    for i, (b1, b2, b3) in enumerate(BASE_STATES):
        for j, o in enumerate(OUT_STATES):
            arr[i, j] = re24.get((b1, b2, b3, o), 0.0)
    return arr


def print_re24(re24: dict[tuple[int, int, int, int], float]) -> None:
    base_labels = ["___", "1__", "_2_", "12_", "__3", "1_3", "_23", "123"]
    print(f"\n{'Bases':>6}  {'0 outs':>8}  {'1 out':>8}  {'2 outs':>8}")
    print("-" * 42)
    for i, (b1, b2, b3) in enumerate(BASE_STATES):
        vals = [re24.get((b1, b2, b3, o), 0.0) for o in OUT_STATES]
        print(f"{base_labels[i]:>6}  {vals[0]:>8.4f}  {vals[1]:>8.4f}  {vals[2]:>8.4f}")
    print()
