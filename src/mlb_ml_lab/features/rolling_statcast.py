from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


_BARREL_MIN_EV = 95.0
_BARREL_MIN_LA = 24.0
_BARREL_MAX_LA = 32.0
_HARD_HIT_MIN_EV = 95.0
_SWEET_SPOT_MIN_LA = 8.0
_SWEET_SPOT_MAX_LA = 32.0


@register
class RollingStatcastFeatures(FeatureExtractor):
    """Rolling statcast metrics from per-game batted-ball data.

    Expects ``statcast_pitch_data`` in ``extract()`` kwargs::

        extra_kwargs={
            "statcast_pitch_data": client.get_statcast_search_data(
                start_date, end_date, player_ids
            ),
        }
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self._windows = windows or [10, 20]

    @property
    def features(self) -> list[FeatureMeta]:
        cols: list[FeatureMeta] = []
        for w in self._windows:
            for name, desc in [
                ("sc_avg_ev", "Average exit velocity on batted balls"),
                ("sc_hardhit_rate", "Hard hit rate (95+ mph EV)"),
                ("sc_barrel_rate", "Barrel rate (95+ EV, 24-32 LA)"),
                ("sc_avg_la", "Average launch angle"),
                ("sc_sweet_spot_rate", "Sweet-spot contact rate (8-32 LA)"),
                ("sc_avg_xba", "Average xBA per batted ball"),
                ("sc_avg_xwoba", "Average xwOBA per batted ball"),
                ("sc_avg_distance", "Average batted ball distance (ft)"),
                ("sc_fbld_rate", "Fly ball + line drive rate"),
                ("sc_gb_rate", "Ground ball rate"),
                ("sc_bbe_count", "Batted ball events in window"),
            ]:
                cols.append(
                    FeatureMeta(
                        name=f"{name}_{w}",
                        description=f"{desc} (last {w} games)",
                        source="statcast_search",
                    )
                )
        return cols

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        pitch_data: list[dict[str, str]] | None = kwargs.get("statcast_pitch_data")
        if not pitch_data:
            return []

        game_stats = _compute_per_game_stats(pitch_data)

        by_player: dict[int, list[_GameStat]] = defaultdict(list)
        for gs in game_stats:
            by_player[gs.player_id].append(gs)
        for gs_list in by_player.values():
            gs_list.sort(key=lambda x: x.date)

        rows: list[dict[str, Any]] = []
        for pid, gs_list in by_player.items():
            _process_player(pid, gs_list, game_logs, self._windows, rows)

        return rows


# ---------------------------------------------------------------------------
# Player-level rolling computation
# ---------------------------------------------------------------------------


def _process_player(
    pid: int,
    gs_list: list[_GameStat],
    game_logs: list[Any],
    windows: list[int],
    rows: list[dict[str, Any]],
) -> None:
    bufs: dict[int, _StatcastRollingBuffer] = {w: _StatcastRollingBuffer(w) for w in windows}

    gs_index = 0
    n_gs = len(gs_list)

    logs_for_player = sorted(
        (log for log in game_logs if log.player_id == pid),
        key=lambda x: x.date,
    )

    for log in logs_for_player:
        gs_index = _advance_buffer(bufs, gs_list, gs_index, n_gs, log.date)

        row: dict[str, Any] = {
            "player_id": pid,
            "game_pk": log.game_pk,
            "date": log.date,
        }
        for w in windows:
            buf = bufs[w]
            n = buf.bbe_count
            if n == 0:
                row.update({
                    f"sc_avg_ev_{w}": None,
                    f"sc_hardhit_rate_{w}": None,
                    f"sc_barrel_rate_{w}": None,
                    f"sc_avg_la_{w}": None,
                    f"sc_sweet_spot_rate_{w}": None,
                    f"sc_avg_xba_{w}": None,
                    f"sc_avg_xwoba_{w}": None,
                    f"sc_avg_distance_{w}": None,
                    f"sc_fbld_rate_{w}": None,
                    f"sc_gb_rate_{w}": None,
                    f"sc_bbe_count_{w}": 0,
                })
                continue

            row.update({
                f"sc_avg_ev_{w}": buf.avg_ev,
                f"sc_hardhit_rate_{w}": round(buf.hardhit_count / n, 3),
                f"sc_barrel_rate_{w}": round(buf.barrel_count / n, 3),
                f"sc_avg_la_{w}": buf.avg_la,
                f"sc_sweet_spot_rate_{w}": round(buf.sweet_spot_count / n, 3),
                f"sc_avg_xba_{w}": buf.avg_xba,
                f"sc_avg_xwoba_{w}": buf.avg_xwoba,
                f"sc_avg_distance_{w}": buf.avg_distance,
                f"sc_fbld_rate_{w}": round((buf.fb_count + buf.ld_count) / n, 3),
                f"sc_gb_rate_{w}": round(buf.gb_count / n, 3),
                f"sc_bbe_count_{w}": n,
            })

        rows.append(row)


def _advance_buffer(
    bufs: dict[int, _StatcastRollingBuffer],
    gs_list: list[_GameStat],
    gs_index: int,
    n_gs: int,
    log_date: str,
) -> int:
    while gs_index < n_gs and gs_list[gs_index].date < log_date:
        gs = gs_list[gs_index]
        for buf in bufs.values():
            buf.append(gs)
        gs_index += 1
    return gs_index


# ---------------------------------------------------------------------------
# Per-game statcast aggregate computation
# ---------------------------------------------------------------------------


class _GameStat:
    __slots__ = ("player_id", "game_pk", "date", "avg_ev", "hardhit_count",
                 "barrel_count", "avg_la", "sweet_spot_count", "avg_xba",
                 "avg_xwoba", "avg_distance", "fb_count", "ld_count",
                 "gb_count", "bunt_count", "popup_count", "bbe_count")

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _StatcastRollingBuffer:
    """Accumulates per-game statcast stats across a rolling window.

    Maintains running sums for counts and running sums-of-values
    for averages, so appending/poping is O(1).
    """

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._ev_vals: deque[float] = deque(maxlen=maxlen)
        self._la_vals: deque[float] = deque(maxlen=maxlen)
        self._xba_vals: deque[float] = deque(maxlen=maxlen)
        self._xwoba_vals: deque[float] = deque(maxlen=maxlen)
        self._dist_vals: deque[float] = deque(maxlen=maxlen)
        self._hardhit_counts: deque[int] = deque(maxlen=maxlen)
        self._barrel_counts: deque[int] = deque(maxlen=maxlen)
        self._sweet_spot_counts: deque[int] = deque(maxlen=maxlen)
        self._fb_counts: deque[int] = deque(maxlen=maxlen)
        self._ld_counts: deque[int] = deque(maxlen=maxlen)
        self._gb_counts: deque[int] = deque(maxlen=maxlen)
        self._bbe_counts: deque[int] = deque(maxlen=maxlen)
        # Running sums
        self._ev_sum = 0.0
        self._la_sum = 0.0
        self._xba_sum = 0.0
        self._xwoba_sum = 0.0
        self._dist_sum = 0.0
        self._hardhit_sum = 0
        self._barrel_sum = 0
        self._sweet_spot_sum = 0
        self._fb_sum = 0
        self._ld_sum = 0
        self._gb_sum = 0
        self._bbe_sum = 0

    @property
    def bbe_count(self) -> int:
        return self._bbe_sum

    @property
    def hardhit_count(self) -> int:
        return self._hardhit_sum

    @property
    def barrel_count(self) -> int:
        return self._barrel_sum

    @property
    def sweet_spot_count(self) -> int:
        return self._sweet_spot_sum

    @property
    def fb_count(self) -> int:
        return self._fb_sum

    @property
    def ld_count(self) -> int:
        return self._ld_sum

    @property
    def gb_count(self) -> int:
        return self._gb_sum

    @property
    def avg_ev(self) -> float | None:
        if not self._ev_vals:
            return None
        return round(self._ev_sum / len(self._ev_vals), 1)

    @property
    def avg_la(self) -> float | None:
        if not self._la_vals:
            return None
        return round(self._la_sum / len(self._la_vals), 1)

    @property
    def avg_xba(self) -> float | None:
        if not self._xba_vals:
            return None
        return round(self._xba_sum / len(self._xba_vals), 3)

    @property
    def avg_xwoba(self) -> float | None:
        if not self._xwoba_vals:
            return None
        return round(self._xwoba_sum / len(self._xwoba_vals), 3)

    @property
    def avg_distance(self) -> float | None:
        if not self._dist_vals:
            return None
        return round(self._dist_sum / len(self._dist_vals), 1)

    def append(self, gs: _GameStat) -> None:
        n = gs.bbe_count
        if n == 0:
            return

        ev = gs.avg_ev
        la = gs.avg_la
        xba = gs.avg_xba
        xwoba = gs.avg_xwoba
        dist = gs.avg_distance

        # If buffer is full, subtract the oldest values
        if len(self._ev_vals) >= self._maxlen:
            old_ev = self._ev_vals[0]
            old_la = self._la_vals[0]
            old_xba = self._xba_vals[0]
            old_xwoba = self._xwoba_vals[0]
            old_dist = self._dist_vals[0]
            self._ev_sum -= old_ev
            self._la_sum -= old_la
            self._xba_sum -= old_xba
            self._xwoba_sum -= old_xwoba
            self._dist_sum -= old_dist
            self._hardhit_sum -= self._hardhit_counts[0]
            self._barrel_sum -= self._barrel_counts[0]
            self._sweet_spot_sum -= self._sweet_spot_counts[0]
            self._fb_sum -= self._fb_counts[0]
            self._ld_sum -= self._ld_counts[0]
            self._gb_sum -= self._gb_counts[0]
            self._bbe_sum -= self._bbe_counts[0]

        self._ev_vals.append(ev)
        self._la_vals.append(la)
        self._xba_vals.append(xba)
        self._xwoba_vals.append(xwoba)
        self._dist_vals.append(dist)
        self._hardhit_counts.append(gs.hardhit_count)
        self._barrel_counts.append(gs.barrel_count)
        self._sweet_spot_counts.append(gs.sweet_spot_count)
        self._fb_counts.append(gs.fb_count)
        self._ld_counts.append(gs.ld_count)
        self._gb_counts.append(gs.gb_count)
        self._bbe_counts.append(n)

        self._ev_sum += ev
        self._la_sum += la
        self._xba_sum += xba
        self._xwoba_sum += xwoba
        self._dist_sum += dist
        self._hardhit_sum += gs.hardhit_count
        self._barrel_sum += gs.barrel_count
        self._sweet_spot_sum += gs.sweet_spot_count
        self._fb_sum += gs.fb_count
        self._ld_sum += gs.ld_count
        self._gb_sum += gs.gb_count
        self._bbe_sum += n


def _compute_per_game_stats(
    pitch_data: list[dict[str, str]],
) -> list[_GameStat]:
    groups: dict[tuple[int, str, int], list[_BBE]] = defaultdict(list)

    for row in pitch_data:
        bbe = _parse_bbe(row)
        if bbe is not None:
            pid = int(row.get("batter", 0))
            if pid == 0:
                continue
            gd = row.get("game_date", "")
            try:
                pk = int(row.get("game_pk", 0))
            except (ValueError, TypeError):
                continue
            groups[(pid, gd, pk)].append(bbe)

    results: list[_GameStat] = []
    for (pid, gd, pk), bbes in groups.items():
        n = len(bbes)
        avg_ev = _mean_or_none([b.ev for b in bbes])
        avg_la = _mean_or_none([b.la for b in bbes])
        avg_xba = _mean_or_none([b.xba for b in bbes])
        avg_xwoba = _mean_or_none([b.xwoba for b in bbes])
        avg_dist = _mean_or_none([b.distance for b in bbes])

        results.append(_GameStat(
            player_id=pid,
            game_pk=pk,
            date=gd,
            avg_ev=avg_ev,
            hardhit_count=sum(1 for b in bbes if b.is_hardhit),
            barrel_count=sum(1 for b in bbes if b.is_barrel),
            avg_la=avg_la,
            sweet_spot_count=sum(1 for b in bbes if b.is_sweet_spot),
            avg_xba=avg_xba,
            avg_xwoba=avg_xwoba,
            avg_distance=avg_dist,
            fb_count=sum(1 for b in bbes if b.bb_type == "fly_ball"),
            ld_count=sum(1 for b in bbes if b.bb_type == "line_drive"),
            gb_count=sum(1 for b in bbes if b.bb_type == "ground_ball"),
            bunt_count=sum(1 for b in bbes if b.bb_type == "bunt"),
            popup_count=sum(1 for b in bbes if b.bb_type == "popup"),
            bbe_count=n,
        ))

    results.sort(key=lambda x: (x.player_id, x.date))
    return results


# ---------------------------------------------------------------------------
# Batted-ball event helpers
# ---------------------------------------------------------------------------


class _BBE:
    __slots__ = ("ev", "la", "xba", "xwoba", "distance", "bb_type",
                 "is_hardhit", "is_barrel", "is_sweet_spot")

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _parse_bbe(row: dict[str, str]) -> _BBE | None:
    ev_str = row.get("launch_speed", "")
    if not ev_str:
        return None
    try:
        ev = float(ev_str)
    except (ValueError, TypeError):
        return None

    la = _safe_float(row.get("launch_angle", ""))
    xba = _safe_float(row.get("estimated_ba_using_speedangle", ""))
    xwoba = _safe_float(row.get("estimated_woba_using_speedangle", ""))
    distance = _safe_float(row.get("hit_distance_sc", ""))
    bb_type = row.get("bb_type", "")

    return _BBE(
        ev=ev,
        la=la or 0.0,
        xba=xba or 0.0,
        xwoba=xwoba or 0.0,
        distance=distance or 0.0,
        bb_type=bb_type,
        is_hardhit=ev >= _HARD_HIT_MIN_EV,
        is_barrel=(ev >= _BARREL_MIN_EV and la is not None
                   and _BARREL_MIN_LA <= la <= _BARREL_MAX_LA)
        if la is not None else False,
        is_sweet_spot=(la is not None and _SWEET_SPOT_MIN_LA <= la
                       <= _SWEET_SPOT_MAX_LA)
        if la is not None else False,
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _safe_float(val: str) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _mean_or_none(vals: list[float | None]) -> float | None:
    filtered = [v for v in vals if v is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 3)
