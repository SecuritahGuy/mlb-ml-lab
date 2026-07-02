"""Target variable construction for hit over/under prediction."""

from __future__ import annotations

from typing import Any


def make_targets(
    game_logs: list[Any],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build target variables from game-log rows.

    Each row contains the actual hit count for that game.  This function
    labels each row with binary targets (hits > threshold) for one or
    more thresholds.

    Args:
        game_logs: PlayerGameLog objects or dicts with ``hits``, ``player_id``,
                   ``game_pk``, ``date``.
        thresholds: Hit thresholds to binarize against.  Defaults to
                    ``[0.5, 1.5]``.

    Returns:
        List of dicts with ``player_id``, ``game_pk``, ``date``, ``hits``,
        and ``target_{threshold}`` keys.
    """
    thresholds = thresholds or [0.5, 1.5]

    rows: list[dict[str, Any]] = []
    for log in game_logs:
        hits = log.hits if hasattr(log, "hits") else log.get("hits", 0)
        row: dict[str, Any] = {
            "player_id": log.player_id,
            "game_pk": log.game_pk,
            "date": log.date,
            "hits": hits,
        }
        for t in thresholds:
            row[f"target_{t}"] = 1 if hits > t else 0
        rows.append(row)

    return rows
