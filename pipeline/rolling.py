"""Rolling-average features computed from player game logs.

Each feature is a windowed statistic (e.g. hits in last 10 games) that
uses only data available before the target game date — no lookahead.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from mibl.data.schemas import PlayerGameLog

from pipeline.base import FeatureExtractor, FeatureMeta, register


@register
class RollingHits(FeatureExtractor):
    """Rolling hit totals and averages over configurable windows."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self._windows = windows or [5, 10, 20]

    @property
    def features(self) -> list[FeatureMeta]:
        cols: list[FeatureMeta] = []
        for w in self._windows:
            cols.append(
                FeatureMeta(
                    name=f"hits_last_{w}",
                    description=f"Total hits in last {w} games",
                    source="game_log",
                )
            )
            cols.append(
                FeatureMeta(
                    name=f"hit_rate_last_{w}",
                    description=f"Hits per game in last {w} games",
                    source="game_log",
                )
            )
        return cols

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        # Group logs by player, sorted by date ascending
        by_player: dict[int, list[PlayerGameLog]] = defaultdict(list)
        for log in game_logs:
            by_player[log.player_id].append(log)

        for logs in by_player.values():
            logs.sort(key=lambda x: x.date)

        rows: list[dict[str, Any]] = []
        for pid, logs in by_player.items():
            window_buffers: dict[int, deque[int]] = {
                w: deque(maxlen=w) for w in self._windows
            }
            for log in logs:
                row: dict[str, Any] = {
                    "player_id": pid,
                    "game_pk": log.game_pk,
                    "date": log.date,
                }
                for w in self._windows:
                    buf = window_buffers[w]
                    total = sum(buf)
                    row[f"hits_last_{w}"] = total
                    row[f"hit_rate_last_{w}"] = round(total / w, 3) if len(buf) == w else None
                rows.append(row)
                # Append this game's hits to all windows *after* computing
                for w in self._windows:
                    window_buffers[w].append(log.hits)

        return rows


@register
class RollingPlateAppearances(FeatureExtractor):
    """Rolling PA, walk, and strikeout rate features."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self._windows = windows or [10, 20]

    @property
    def features(self) -> list[FeatureMeta]:
        cols: list[FeatureMeta] = []
        for w in self._windows:
            cols.append(
                FeatureMeta(
                    name=f"avg_pa_last_{w}",
                    description=f"Average plate appearances in last {w} games",
                    source="game_log",
                )
            )
            cols.append(
                FeatureMeta(
                    name=f"bb_rate_last_{w}",
                    description=f"Walk rate in last {w} games",
                    source="game_log",
                )
            )
            cols.append(
                FeatureMeta(
                    name=f"k_rate_last_{w}",
                    description=f"Strikeout rate in last {w} games",
                    source="game_log",
                )
            )
        return cols

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        by_player: dict[int, list[PlayerGameLog]] = defaultdict(list)
        for log in game_logs:
            by_player[log.player_id].append(log)
        for logs in by_player.values():
            logs.sort(key=lambda x: x.date)

        rows: list[dict[str, Any]] = []
        for pid, logs in by_player.items():
            pa_buf: dict[int, deque[int]] = {w: deque(maxlen=w) for w in self._windows}
            bb_buf: dict[int, deque[int]] = {w: deque(maxlen=w) for w in self._windows}
            k_buf: dict[int, deque[int]] = {w: deque(maxlen=w) for w in self._windows}

            for log in logs:
                row: dict[str, Any] = {
                    "player_id": pid,
                    "game_pk": log.game_pk,
                    "date": log.date,
                }
                for w in self._windows:
                    pa_vals = list(pa_buf[w])
                    bb_vals = list(bb_buf[w])
                    k_vals = list(k_buf[w])
                    total_pa = sum(pa_vals)
                    total_bb = sum(bb_vals)
                    total_k = sum(k_vals)
                    row[f"avg_pa_last_{w}"] = (
                        round(total_pa / w, 1) if len(pa_vals) == w else None
                    )
                    row[f"bb_rate_last_{w}"] = (
                        round(total_bb / total_pa, 3) if total_pa > 0 else None
                    )
                    row[f"k_rate_last_{w}"] = (
                        round(total_k / total_pa, 3) if total_pa > 0 else None
                    )

                rows.append(row)

                pa_walks = log.plate_appearances
                for w in self._windows:
                    pa_buf[w].append(pa_walks)
                    bb_buf[w].append(log.walks)
                    k_buf[w].append(log.strikeouts)

        return rows


@register
class RollingBABIP(FeatureExtractor):
    """Rolling BABIP (batting average on balls in play)."""

    def __init__(self, window: int = 20) -> None:
        self._window = window

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="babip_last_20",
                description="Rolling BABIP over last 20 games",
                source="game_log",
            )
        ]

    def extract(self, game_logs: list[PlayerGameLog], **kwargs: Any) -> list[dict[str, Any]]:
        by_player: dict[int, list[PlayerGameLog]] = defaultdict(list)
        for log in game_logs:
            by_player[log.player_id].append(log)
        for logs in by_player.values():
            logs.sort(key=lambda x: x.date)

        rows: list[dict[str, Any]] = []
        for pid, logs in by_player.items():
            hits_buf: deque[int] = deque(maxlen=self._window)
            abs_buf: deque[int] = deque(maxlen=self._window)
            ks_buf: deque[int] = deque(maxlen=self._window)
            bbs_buf: deque[int] = deque(maxlen=self._window)

            for log in logs:
                babip: float | None = None
                if len(hits_buf) == self._window:
                    total_h = sum(hits_buf)
                    total_ab = sum(abs_buf)
                    total_k = sum(ks_buf)
                    total_bb = sum(bbs_buf)
                    bip = total_ab - total_k - total_bb
                    if bip > 0:
                        babip = round(total_h / bip, 3)
                rows.append({
                    "player_id": pid,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    "babip_last_20": babip,
                })
                hits_buf.append(log.hits)
                abs_buf.append(log.at_bats)
                ks_buf.append(log.strikeouts)
                bbs_buf.append(log.walks)

        return rows
