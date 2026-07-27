from __future__ import annotations

from typing import Any

from mlb_ml_lab.data.client import MlbClient
from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


def build_player_timelines(
    transactions: list[dict[str, Any]],
) -> dict[int, list[tuple[str, str]]]:
    """Build a per-player timeline of IL events from raw transactions.

    Args:
        transactions: Raw transaction dicts from
            ``MlbClient.get_transactions(season)``.

    Returns:
        Dict mapping ``player_id`` → sorted list of
        ``(iso_date, event_type)`` tuples where event_type
        is ``"il_placement"`` or ``"il_activation"``.
    """
    classify = MlbClient.classify_transaction
    events: dict[int, list[tuple[str, str]]] = {}
    for txn in transactions:
        cat = classify(txn)
        if cat not in ("il_placement", "il_activation"):
            continue
        person = txn.get("person")
        if person is None:
            continue
        pid = person["id"]
        date = txn.get("effectiveDate") or txn["date"]
        events.setdefault(pid, []).append((date, cat))
    for tl in events.values():
        tl.sort(key=lambda x: x[0])
    return events


def _status_at_date(
    timeline: list[tuple[str, str]], date_str: str
) -> tuple[int, int | None]:
    """Compute IL status for a player at a given date.

    Returns ``(days_since_il_deactivation, days_on_il)`` where one is
    always ``None`` (player is either on or off IL at that date).
    """
    if not timeline:
        return (None, None)

    current = timeline[0][1]
    last_date = timeline[0][0]
    for event_date, event_type in timeline:
        if event_date > date_str:
            break
        current = event_type
        last_date = event_date

    from datetime import date

    game_date = date.fromisoformat(date_str)
    event_date_obj = date.fromisoformat(last_date) if last_date else game_date

    if current == "il_placement":
        return (None, (game_date - event_date_obj).days)
    return ((game_date - event_date_obj).days, None)


def _days_on_il(timeline: list[tuple[str, str]], date_str: str) -> int | None:
    result = _status_at_date(timeline, date_str)
    return result[1]  # days_on_il


def _days_since_il(timeline: list[tuple[str, str]], date_str: str) -> int | None:
    result = _status_at_date(timeline, date_str)
    return result[0]  # days_since_il_deactivation


@register
class InjuryFeatures(FeatureExtractor):
    """Player injury/IL status features.

    Expects ``injury_data`` kwarg — a dict with key
    ``\"player_timelines"`` returned by
    :func:`build_player_timelines`.

    Source data can be fetched via
    ``MlbClient.get_transactions(season)``.
    """

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="days_on_il",
                description="Days player has been on IL (None if active)",
                source="injuries",
            ),
            FeatureMeta(
                name="days_since_il",
                description="Days since last IL activation (None if on IL)",
                source="injuries",
            ),
            FeatureMeta(
                name="il_flag",
                description="1 if player is on IL at game date",
                source="injuries",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs: list[Any] = kwargs.get("game_logs", [])
        injury_data: dict[str, Any] | None = kwargs.get("injury_data")
        if not injury_data or not game_logs:
            return _empty_rows(game_logs)

        timelines: dict[int, list[tuple[str, str]]] = injury_data.get(
            "player_timelines", {}
        )

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            pid = log.player_id
            date_str = (
                log.date.isoformat()
                if hasattr(log.date, "isoformat")
                else str(log.date)
            )
            tl = timelines.get(pid, [])
            d_il = _days_on_il(tl, date_str)
            d_since = _days_since_il(tl, date_str)

            rows.append(
                {
                    "player_id": pid,
                    "game_pk": log.game_pk,
                    "date": log.date,
                    "days_on_il": d_il,
                    "days_since_il": d_since,
                    "il_flag": 1 if d_il is not None else 0,
                }
            )
        return rows


def _empty_rows(game_logs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": log.player_id,
            "game_pk": log.game_pk,
            "date": log.date,
            "days_on_il": None,
            "days_since_il": None,
            "il_flag": 0,
        }
        for log in game_logs
    ]
