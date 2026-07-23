from __future__ import annotations

import json
from collections import deque

import numpy as np

from mlb_ml_lab.simulation.outcomes import OUTCOME_CLASSES

OUTCOME_TO_CLASS = {oc: i for i, oc in enumerate(OUTCOME_CLASSES)}

BATTER_WINDOWS = [10, 20, 50, 100]
PITCHER_WINDOWS = [20, 50, 100]

PLATOON_FEATURES = ["batter_hand_R", "batter_hand_L", "pitcher_hand_R", "pitcher_hand_L", "platoon_advantage"]

GAME_CONTEXT_FEATURES = ["park_wOBA", "park_HR", "temp", "wind_speed", "indoor"]

_PA_SAMPLE_RATE = 1.0  # can reduce for faster dev

_handedness: dict[int, dict[str, str]] | None = None


def _load_handedness(path: str = "data/simulation/player_handedness.json") -> dict[int, dict[str, str]]:
    global _handedness
    if _handedness is None:
        with open(path) as f:
            raw = json.load(f)
        _handedness = {int(k): v for k, v in raw.items()}
    return _handedness


_game_context: dict[str, dict] | None = None


def _load_game_context(path: str = "data/simulation/game_context.json") -> dict[str, dict]:
    global _game_context
    if _game_context is None:
        with open(path) as f:
            _game_context = json.load(f)
    return _game_context


def _window_key(prefix: str, window: int, outcome: str) -> str:
    return f"{prefix}_last{window}_{outcome}_rate"


def _window_count_key(prefix: str, window: int) -> str:
    return f"{prefix}_last{window}_count"


def _all_feature_names(include_context: bool = True, include_platoon: bool = True, include_game_context: bool = True) -> list[str]:
    names: list[str] = []
    for w in BATTER_WINDOWS:
        for o in OUTCOME_CLASSES:
            names.append(_window_key("batter", w, o))
        names.append(_window_count_key("batter", w))
    for w in PITCHER_WINDOWS:
        for o in OUTCOME_CLASSES:
            names.append(_window_key("pitcher", w, o))
        names.append(_window_count_key("pitcher", w))
    if include_platoon:
        names.extend(PLATOON_FEATURES)
    if include_game_context:
        names.extend(GAME_CONTEXT_FEATURES)
    if include_context:
        names.extend(["half_inning_top", "inning", "outs_before", "balls", "strikes"])
    return names


def compute_pbp_features(
    pas: list[dict],
    game_dates: dict[int, str],
    sample_rate: float = 1.0,
    include_context: bool = True,
    include_platoon: bool = True,
    include_game_context: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Compute rolling features for every PA in the dataset.

    Args:
        pas: List of PA dicts from PBP JSONL.
        game_dates: Map of ``game_pk → YYYY-MM-DD``.
        sample_rate: Fraction of PAs to include (1.0 = all).
        include_context: Include in-game features (inning, outs, count).

    Returns:
        ``(X, y, feature_names, game_pks)`` where:
        - X: (n_samples, n_features) float array
        - y: (n_samples,) int array of outcome class indices
        - feature_names: list of column names
        - game_pks: list of game_pk for each sample
    """
    dated_pas = []
    for pa in pas:
        gpk = pa["game_pk"]
        date = game_dates.get(gpk, "")
        if date:
            dated_pas.append((date, pa))

    dated_pas.sort(key=lambda x: (x[0], x[1]["game_pk"], x[1]["at_bat_index"]))

    batter_windows: dict[int, dict[int, deque[str]]] = {}
    pitcher_windows: dict[int, dict[int, deque[str]]] = {}

    feature_names = _all_feature_names(include_context=include_context, include_platoon=include_platoon, include_game_context=include_game_context)
    n_features = len(feature_names)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    gpk_list: list[int] = []

    total = len(dated_pas)
    for i, (date, pa) in enumerate(dated_pas):
        if sample_rate < 1.0 and np.random.random() > sample_rate:
            _update_windows(batter_windows, pitcher_windows, pa)
            continue

        if (i + 1) % 50000 == 0:
            print(f"  [{i + 1}/{total}]")

        bid = pa["batter_id"]
        pid = pa["pitcher_id"]
        et = pa["event_type"]
        cls_idx = OUTCOME_TO_CLASS.get(et, OUTCOME_TO_CLASS["other"])

        features = np.zeros(n_features, dtype=np.float64)
        col = 0

        for w in BATTER_WINDOWS:
            hist = batter_windows.get(bid, {}).get(w, deque())
            cnt = len(hist)
            for oi, o in enumerate(OUTCOME_CLASSES):
                rate = sum(1 for e in hist if e == o) / max(cnt, 1)
                features[col + oi] = rate
            features[col + len(OUTCOME_CLASSES)] = float(cnt)
            col += len(OUTCOME_CLASSES) + 1

        for w in PITCHER_WINDOWS:
            hist = pitcher_windows.get(pid, {}).get(w, deque())
            cnt = len(hist)
            for oi, o in enumerate(OUTCOME_CLASSES):
                rate = sum(1 for e in hist if e == o) / max(cnt, 1)
                features[col + oi] = rate
            features[col + len(OUTCOME_CLASSES)] = float(cnt)
            col += len(OUTCOME_CLASSES) + 1

        if include_platoon:
            hd = _load_handedness()
            bh = hd.get(bid, {"batSide": "?"})["batSide"]
            ph = hd.get(pid, {"pitchHand": "?"})["pitchHand"]
            features[col] = 1.0 if bh == "R" else 0.0
            features[col + 1] = 1.0 if bh == "L" else 0.0
            features[col + 2] = 1.0 if ph == "R" else 0.0
            features[col + 3] = 1.0 if ph == "L" else 0.0
            if bh == "S" or bh == "?":
                features[col + 4] = 1.0
            elif ph == "?" or ph == "S":
                features[col + 4] = 0.0
            else:
                features[col + 4] = 1.0 if bh != ph else 0.0
            col += len(PLATOON_FEATURES)

        if include_game_context:
            gc = _load_game_context()
            gpk_str = str(pa["game_pk"])
            ctx = gc.get(gpk_str, {})
            features[col] = float(ctx.get("park_wOBA", 1.0))
            features[col + 1] = float(ctx.get("park_HR", 1.0))
            features[col + 2] = float(ctx.get("temp", 72.0))
            features[col + 3] = float(ctx.get("wind_speed", 0.0))
            features[col + 4] = float(ctx.get("indoor", 0.0))
            col += len(GAME_CONTEXT_FEATURES)

        if include_context:
            features[col] = 1.0 if pa.get("half_inning") == "top" else 0.0
            features[col + 1] = float(pa.get("inning", 1))
            features[col + 2] = float(pa.get("outs_before", 0))
            features[col + 3] = float(pa.get("balls", 0))
            features[col + 4] = float(pa.get("strikes", 0))

        X_list.append(features)
        y_list.append(cls_idx)
        gpk_list.append(pa["game_pk"])

        _update_windows(batter_windows, pitcher_windows, pa)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    return X, y, feature_names, gpk_list


class RollingState:
    """Maintain rolling windows and produce feature vectors for any matchup."""

    def __init__(self, game_dates: dict[int, str], handedness_path: str = "data/simulation/player_handedness.json",
                 game_context_path: str = "data/simulation/game_context.json"):
        self.game_dates = game_dates
        self.batter_windows: dict[int, dict[int, deque[str]]] = {}
        self.pitcher_windows: dict[int, dict[int, deque[str]]] = {}
        self._processed: set[int] = set()
        self._handedness = _load_handedness(handedness_path)
        self._game_context = _load_game_context(game_context_path)

    def replay_until(self, pas: list[dict], target_date: str) -> None:
        """Replay all PAs before *target_date* to populate rolling windows."""
        dated = []
        for pa in pas:
            gpk = pa["game_pk"]
            if gpk in self._processed:
                continue
            date = self.game_dates.get(gpk, "")
            if date and date < target_date:
                dated.append((date, pa))

        dated.sort(key=lambda x: (x[0], x[1]["game_pk"], x[1]["at_bat_index"]))
        for date, pa in dated:
            _update_windows(self.batter_windows, self.pitcher_windows, pa)
            self._processed.add(pa["game_pk"])

    def feature_vector(
        self,
        batter_id: int,
        pitcher_id: int,
        half_inning: str = "top",
        inning: int = 1,
        outs_before: int = 0,
        balls: int = 0,
        strikes: int = 0,
        include_context: bool = False,
        include_platoon: bool = True,
        include_game_context: bool = True,
        game_pk: int = 0,
    ) -> np.ndarray:
        """Build feature vector for a batter-pitcher matchup."""
        n_bat = len(BATTER_WINDOWS) * (len(OUTCOME_CLASSES) + 1)
        n_pit = len(PITCHER_WINDOWS) * (len(OUTCOME_CLASSES) + 1)
        n_plt = len(PLATOON_FEATURES) if include_platoon else 0
        n_gcx = len(GAME_CONTEXT_FEATURES) if include_game_context else 0
        n_ctx = 5 if include_context else 0
        features = np.zeros(n_bat + n_pit + n_plt + n_gcx + n_ctx, dtype=np.float64)

        col = 0
        for w in BATTER_WINDOWS:
            hist = self.batter_windows.get(batter_id, {}).get(w, deque())
            cnt = len(hist)
            for oi, o in enumerate(OUTCOME_CLASSES):
                rate = sum(1 for e in hist if e == o) / max(cnt, 1)
                features[col + oi] = rate
            features[col + len(OUTCOME_CLASSES)] = float(cnt)
            col += len(OUTCOME_CLASSES) + 1

        for w in PITCHER_WINDOWS:
            hist = self.pitcher_windows.get(pitcher_id, {}).get(w, deque())
            cnt = len(hist)
            for oi, o in enumerate(OUTCOME_CLASSES):
                rate = sum(1 for e in hist if e == o) / max(cnt, 1)
                features[col + oi] = rate
            features[col + len(OUTCOME_CLASSES)] = float(cnt)
            col += len(OUTCOME_CLASSES) + 1

        if include_platoon:
            hd = self._handedness
            bh = hd.get(batter_id, {"batSide": "?"})["batSide"]
            ph = hd.get(pitcher_id, {"pitchHand": "?"})["pitchHand"]
            features[col] = 1.0 if bh == "R" else 0.0
            features[col + 1] = 1.0 if bh == "L" else 0.0
            features[col + 2] = 1.0 if ph == "R" else 0.0
            features[col + 3] = 1.0 if ph == "L" else 0.0
            if bh == "S" or bh == "?":
                features[col + 4] = 1.0
            elif ph == "?" or ph == "S":
                features[col + 4] = 0.0
            else:
                features[col + 4] = 1.0 if bh != ph else 0.0
            col += len(PLATOON_FEATURES)

        if include_game_context:
            gc = self._game_context
            ctx = gc.get(str(game_pk), {})
            features[col] = float(ctx.get("park_wOBA", 1.0))
            features[col + 1] = float(ctx.get("park_HR", 1.0))
            features[col + 2] = float(ctx.get("temp", 72.0))
            features[col + 3] = float(ctx.get("wind_speed", 0.0))
            features[col + 4] = float(ctx.get("indoor", 0.0))
            col += len(GAME_CONTEXT_FEATURES)

        if include_context:
            features[col] = 1.0 if half_inning == "top" else 0.0
            features[col + 1] = float(inning)
            features[col + 2] = float(outs_before)
            features[col + 3] = float(balls)
            features[col + 4] = float(strikes)

        return features


def _update_windows(
    batter_windows: dict[int, dict[int, deque[str]]],
    pitcher_windows: dict[int, dict[int, deque[str]]],
    pa: dict,
) -> None:
    bid = pa["batter_id"]
    pid = pa["pitcher_id"]
    et = pa["event_type"]

    for w in BATTER_WINDOWS:
        bw = batter_windows.setdefault(bid, {}).setdefault(w, deque(maxlen=w))
        bw.append(et)

    for w in PITCHER_WINDOWS:
        pw = pitcher_windows.setdefault(pid, {}).setdefault(w, deque(maxlen=w))
        pw.append(et)
