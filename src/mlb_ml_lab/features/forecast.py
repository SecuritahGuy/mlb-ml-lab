"""Weather forecast features via NWS API.

Unlike ``WeatherFeatures`` (which reads historical conditions from the
MLB Stats API game feed), this extractor fetches live forecast data
from the National Weather Service for each game's venue and time.

Usage from ``build_feature_matrix``::

    matrix = build_feature_matrix(
        game_logs,
        season=2025,
        teams=teams,
        extra_kwargs={
            "game_contexts": {game_pk: {"game_datetime": "..."}, ...},
        },
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mlb_ml_lab.data.weather import INDOOR_VENUES, NwsWeather

from mlb_ml_lab.features.base import FeatureExtractor, FeatureMeta, register


@register
class WeatherForecastFeatures(FeatureExtractor):
    """NWS forecast weather at game time for each game's venue.

    Requires:
        * ``teams`` — list of team dicts (to resolve venue_id).
        * ``game_contexts`` — dict mapping ``game_pk`` → dict with
          ``game_datetime`` (ISO-8601 string).

    Accepts an optional ``nws`` kwarg in ``extract()`` (an
    ``NwsWeather`` instance) for dependency injection in tests.
    If absent, a default is created.
    """

    def __init__(self) -> None:
        self._nws: NwsWeather | None = None

    @property
    def features(self) -> list[FeatureMeta]:
        return [
            FeatureMeta(
                name="forecast_temp",
                description="Forecast temperature (°F) at game time",
                source="forecast",
            ),
            FeatureMeta(
                name="forecast_wind_speed",
                description="Forecast wind speed (string, e.g. '10 mph')",
                source="forecast",
            ),
            FeatureMeta(
                name="forecast_wind_direction",
                description="Forecast wind direction (string, e.g. 'SW')",
                source="forecast",
            ),
            FeatureMeta(
                name="forecast_precip_pct",
                description="Forecast precipitation probability (0-100)",
                source="forecast",
            ),
            FeatureMeta(
                name="forecast_conditions",
                description="Forecast conditions label (e.g. 'Partly Cloudy')",
                source="forecast",
            ),
        ]

    def extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        game_logs = kwargs.get("game_logs", [])
        teams: list[dict[str, Any]] | None = kwargs.get("teams")
        contexts: dict[int, dict[str, Any]] | None = kwargs.get("game_contexts")
        venue_map = _resolve_venue_map(teams)

        nws: NwsWeather = kwargs.get("nws") or self._nws or NwsWeather()
        if self._nws is None:
            self._nws = nws

        rows: list[dict[str, Any]] = []
        for log in game_logs:
            ctx = (contexts or {}).get(log.game_pk, {}) if contexts else {}
            game_datetime_str: str | None = ctx.get("game_datetime")

            home_team_id = log.team_id if log.is_home else log.opponent_id
            venue_id = venue_map.get(home_team_id)

            forecast: dict[str, Any] | None = None
            if venue_id is not None and game_datetime_str:
                if venue_id in INDOOR_VENUES:
                    forecast = {
                        "temp": 72,
                        "wind_speed": "0 mph",
                        "wind_direction": "",
                        "precip_pct": 0,
                        "conditions": "Indoor",
                        "source": "indoor",
                    }
                else:
                    try:
                        dt = datetime.fromisoformat(game_datetime_str)
                        result = nws.forecast(venue_id, target_time=dt)
                        if result:
                            forecast = result
                    except (ValueError, TypeError):
                        pass
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass

            rows.append({
                "player_id": log.player_id,
                "game_pk": log.game_pk,
                "date": log.date,
                "forecast_temp": (forecast or {}).get("temp"),
                "forecast_wind_speed": (forecast or {}).get("wind_speed"),
                "forecast_wind_direction": (forecast or {}).get("wind_direction"),
                "forecast_precip_pct": (forecast or {}).get("precip_pct"),
                "forecast_conditions": (forecast or {}).get("conditions"),
            })

        return rows

    def close(self) -> None:
        if self._nws:
            self._nws.close()


def _resolve_venue_map(
    teams: list[dict[str, Any]] | None,
) -> dict[int, int]:
    if not teams:
        return {}
    mapping: dict[int, int] = {}
    for t in teams:
        venue = t.get("venue") or {}
        venue_id = venue.get("id")
        if venue_id is not None:
            mapping[t["id"]] = venue_id
    return mapping
