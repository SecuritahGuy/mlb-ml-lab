"""Tests for WeatherForecastFeatures extractor."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mlb_ml_lab.data.schemas import PlayerGameLog
from mlb_ml_lab.data.weather import NwsWeather
from mlb_ml_lab.features.assemble import build_feature_matrix, describe_features
from mlb_ml_lab.features.forecast import WeatherForecastFeatures


def _log(**kw: Any) -> PlayerGameLog:
    defaults: dict[str, Any] = {
        "player_id": 1,
        "player_name": "A",
        "team_id": 108,
        "opponent_id": 145,
        "date": "2025-04-01",
        "game_pk": 1000,
        "is_home": True,
        "is_win": True,
        "game_type": "R",
        "season": "2025",
        "hits": 1,
        "at_bats": 4,
        "plate_appearances": 4,
        "walks": 0,
        "strikeouts": 1,
    }
    defaults.update(kw)
    return PlayerGameLog(**defaults)


def _mock_nws(fixed_result: dict[str, Any] | None = None) -> MagicMock:
    nws = MagicMock(spec=NwsWeather)
    nws.forecast.return_value = fixed_result or {
        "temp": 72,
        "wind_speed": "10 mph",
        "wind_direction": "SW",
        "precip_pct": 20,
        "conditions": "Partly Cloudy",
        "source": "forecast",
    }
    return nws


class TestWeatherForecastFeatures:
    def test_features_metadata(self):
        feature_names = {f.name for f in WeatherForecastFeatures().features}
        expected = {
            "forecast_temp",
            "forecast_wind_speed",
            "forecast_wind_direction",
            "forecast_precip_pct",
            "forecast_conditions",
        }
        assert feature_names == expected

    def test_returns_forecast_for_each_game(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log()]
        contexts = {
            1000: {
                "weather_condition": "Clear",
                "weather_temp": "75",
                "game_datetime": "2025-04-01T18:10:00Z",
            }
        }
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["player_id"] == 1
        assert r["game_pk"] == 1000
        assert r["forecast_temp"] == 72
        assert r["forecast_wind_speed"] == "10 mph"
        assert r["forecast_wind_direction"] == "SW"
        assert r["forecast_precip_pct"] == 20
        assert r["forecast_conditions"] == "Partly Cloudy"

    def test_away_game_uses_opponent_venue(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log(team_id=108, opponent_id=145, is_home=False)]
        contexts = {
            1000: {"game_datetime": "2025-04-01T18:10:00Z"},
        }
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        extractor.extract(game_logs=logs, teams=teams, game_contexts=contexts, nws=nws)
        nws.forecast.assert_called_once()
        args, _ = nws.forecast.call_args
        assert args[0] == 680

    def test_home_game_uses_own_venue(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log(team_id=108, opponent_id=145, is_home=True)]
        contexts = {
            1000: {"game_datetime": "2025-04-01T18:10:00Z"},
        }
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        extractor.extract(game_logs=logs, teams=teams, game_contexts=contexts, nws=nws)
        nws.forecast.assert_called_once()
        args, _ = nws.forecast.call_args
        assert args[0] == 19

    def test_indoor_venue_returns_indoor(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log(team_id=141, opponent_id=145, is_home=True)]
        contexts = {
            1000: {"game_datetime": "2025-04-01T18:10:00Z"},
        }
        teams = [{"id": 141, "venue": {"id": 14}}, {"id": 145, "venue": {"id": 680}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        r = rows[0]
        assert r["forecast_conditions"] == "Indoor"
        assert r["forecast_temp"] == 72
        assert r["forecast_precip_pct"] == 0
        nws.forecast.assert_not_called()

    def test_no_contexts_defaults_to_none(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log()]
        rows = extractor.extract(
            game_logs=logs, teams=[{"id": 108, "venue": {"id": 19}}], nws=nws,
        )
        r = rows[0]
        assert r["forecast_temp"] is None
        assert r["forecast_wind_speed"] is None
        assert r["forecast_precip_pct"] is None
        assert r["forecast_conditions"] is None
        nws.forecast.assert_not_called()

    def test_no_game_datetime_skips_forecast(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log()]
        contexts = {1000: {"weather_condition": "Clear"}}
        teams = [{"id": 108, "venue": {"id": 19}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        r = rows[0]
        assert r["forecast_temp"] is None
        nws.forecast.assert_not_called()

    def test_no_teams_defaults_to_none(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log()]
        contexts = {1000: {"game_datetime": "2025-04-01T18:10:00Z"}}
        rows = extractor.extract(game_logs=logs, game_contexts=contexts, nws=nws)
        r = rows[0]
        assert r["forecast_temp"] is None
        nws.forecast.assert_not_called()

    def test_unknown_venue_defaults_to_none(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log(team_id=999, opponent_id=145, is_home=True)]
        contexts = {1000: {"game_datetime": "2025-04-01T18:10:00Z"}}
        teams = [{"id": 999, "venue": None}, {"id": 145, "venue": {"id": 680}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        r = rows[0]
        assert r["forecast_temp"] is None
        nws.forecast.assert_not_called()

    def test_invalid_datetime_skips_forecast(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [_log()]
        contexts = {1000: {"game_datetime": "not-a-datetime"}}
        teams = [{"id": 108, "venue": {"id": 19}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        r = rows[0]
        assert r["forecast_temp"] is None
        nws.forecast.assert_not_called()

    def test_nws_exception_returns_none_for_that_game(self):
        nws = _mock_nws()
        nws.forecast.side_effect = RuntimeError("API down")
        extractor = WeatherForecastFeatures()
        logs = [_log(game_pk=1000), _log(game_pk=1001)]
        contexts = {
            1000: {"game_datetime": "2025-04-01T18:10:00Z"},
            1001: {"game_datetime": "2025-04-02T18:10:00Z"},
        }
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        for r in rows:
            assert r["forecast_temp"] is None

    def test_multiple_games_multiple_contexts(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [
            _log(player_id=1, game_pk=1000, date="2025-04-01"),
            _log(player_id=2, game_pk=1001, date="2025-04-02"),
        ]
        contexts = {
            1000: {"game_datetime": "2025-04-01T18:10:00Z"},
            1001: {"game_datetime": "2025-04-02T18:10:00Z"},
        }
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        assert len(rows) == 2
        assert rows[0]["player_id"] == 1
        assert rows[1]["player_id"] == 2
        assert nws.forecast.call_count == 2

    def test_mixed_availability(self):
        nws = _mock_nws()
        extractor = WeatherForecastFeatures()
        logs = [
            _log(player_id=1, game_pk=1000, date="2025-04-01"),
            _log(player_id=2, game_pk=1001, date="2025-04-02"),
            _log(player_id=3, game_pk=1002, date="2025-04-03"),
        ]
        contexts = {
            1000: {"game_datetime": "2025-04-01T18:10:00Z"},
            1001: {"weather_condition": "Clear"},
        }
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        rows = extractor.extract(
            game_logs=logs, teams=teams, game_contexts=contexts, nws=nws,
        )
        assert len(rows) == 3
        assert rows[0]["forecast_temp"] == 72
        assert rows[1]["forecast_temp"] is None
        assert rows[2]["forecast_temp"] is None
        assert nws.forecast.call_count == 1

    def test_defaults_to_live_nws_when_not_injected(self):
        extractor = WeatherForecastFeatures()
        rows = extractor.extract(game_logs=[_log()])
        assert rows[0]["forecast_temp"] is None


class TestForecastInFullAssembly:
    def test_forecast_features_in_matrix(self):
        nws = _mock_nws()
        logs = [_log()]
        contexts = {
            1000: {"game_datetime": "2025-04-01T18:10:00Z"},
        }
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        matrix = build_feature_matrix(
            logs,
            teams=teams,
            extra_kwargs={
                "game_contexts": contexts,
                "nws": nws,
            },
        )
        assert len(matrix) == 1
        r = matrix[0]
        assert r["forecast_temp"] == 72

    def test_forecast_features_in_describe(self):
        metas = describe_features()
        names = {m.name for m in metas}
        assert "forecast_temp" in names
        assert "forecast_conditions" in names
