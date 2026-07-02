from mibl.data.schemas import PlayerGameLog
from mibl.features.context import (
    HomeAwayFeature,
    RestDaysFeature,
    ParkFactorFeatures,
    WeatherFeatures,
)


def _log(**kwargs) -> PlayerGameLog:
    defaults = dict(
        player_id=1, player_name="A", team_id=108, opponent_id=145,
        date="2025-04-01", game_pk=1000, is_home=True, is_win=True,
        game_type="R", season="2025",
        hits=0, at_bats=4, plate_appearances=4,
    )
    defaults.update(kwargs)
    return PlayerGameLog(**defaults)


class TestHomeAwayFeature:
    def test_home_is_1(self):
        log = _log(is_home=True)
        rows = HomeAwayFeature().extract(game_logs=[log])
        assert rows[0]["is_home"] == 1

    def test_away_is_0(self):
        log = _log(is_home=False)
        rows = HomeAwayFeature().extract(game_logs=[log])
        assert rows[0]["is_home"] == 0


class TestRestDaysFeature:
    def test_first_game_no_rest(self):
        log = _log(date="2025-04-01")
        rows = RestDaysFeature().extract(game_logs=[log])
        assert rows[0]["rest_days"] is None

    def test_consecutive_days_rest(self):
        logs = [
            _log(date="2025-04-01", game_pk=1),
            _log(date="2025-04-02", game_pk=2),
        ]
        rows = RestDaysFeature().extract(game_logs=logs)
        assert rows[1]["rest_days"] == 0

    def test_one_day_off(self):
        logs = [
            _log(date="2025-04-01", game_pk=1),
            _log(date="2025-04-03", game_pk=2),
        ]
        rows = RestDaysFeature().extract(game_logs=logs)
        assert rows[1]["rest_days"] == 1

    def test_per_player(self):
        logs = [
            _log(player_id=1, date="2025-04-01", game_pk=1),
            _log(player_id=2, date="2025-04-01", game_pk=2),
            _log(player_id=1, date="2025-04-03", game_pk=3),
        ]
        rows = {r["game_pk"]: r for r in RestDaysFeature().extract(game_logs=logs)}
        assert rows[1]["rest_days"] is None
        assert rows[3]["rest_days"] == 1

    def test_invalid_date_format(self):
        logs = [
            _log(date="2025-04-01", game_pk=1),
            _log(date="bad-date", game_pk=2, player_id=1),
        ]
        rows = RestDaysFeature().extract(game_logs=logs)
        assert rows[1]["rest_days"] is None


class TestParkFactorFeatures:
    def test_returns_all_keys(self):
        log = _log()
        rows = ParkFactorFeatures().extract(game_logs=[log])
        r = rows[0]
        assert "park_wOBA" in r
        assert "park_HR" in r
        assert "park_1B" in r

    def test_no_teams_defaults_to_neutral(self):
        log = _log()
        rows = ParkFactorFeatures().extract(game_logs=[log])
        assert rows[0]["park_wOBA"] == 1.0

    def test_team_without_venue_defaults(self):
        teams = [{"id": 108, "venue": None}]
        log = _log(team_id=108, is_home=True)
        rows = ParkFactorFeatures().extract(game_logs=[log], teams=teams)
        assert rows[0]["park_wOBA"] == 1.0

    def test_home_game_uses_player_team_venue(self):
        teams = [{"id": 108, "venue": {"id": 19}}]
        log = _log(team_id=108, opponent_id=145, is_home=True)
        rows = ParkFactorFeatures().extract(game_logs=[log], teams=teams, season=2025)
        assert rows[0]["park_wOBA"] > 1.0

    def test_away_game_uses_opponent_venue(self):
        teams = [
            {"id": 108, "venue": {"id": 19}},
            {"id": 145, "venue": {"id": 680}},
        ]
        log = _log(team_id=108, opponent_id=145, is_home=False)
        rows = ParkFactorFeatures().extract(game_logs=[log], teams=teams, season=2025)
        assert rows[0]["park_HR"] < 1.0

    def test_resolve_venue_map_returns_expected_mapping(self):
        teams = [
            {"id": 108, "venue": {"id": 1}},
            {"id": 145, "venue": {"id": 4}},
        ]
        mapping = ParkFactorFeatures._resolve_venue_map(teams)
        assert mapping[108] == 1
        assert mapping[145] == 4

    def test_resolve_venue_map_empty_for_no_teams(self):
        assert ParkFactorFeatures._resolve_venue_map(None) == {}
        assert ParkFactorFeatures._resolve_venue_map([]) == {}


class TestWeatherFeatures:
    def test_no_contexts_all_none(self):
        log = _log()
        rows = WeatherFeatures().extract(game_logs=[log])
        r = rows[0]
        assert r["weather_condition"] is None
        assert r["weather_temp"] is None
        assert r["weather_wind"] is None

    def test_game_pk_in_contexts(self):
        contexts = {
            1000: {
                "weather_condition": "Cloudy",
                "weather_temp": "72",
                "weather_wind": "10 mph",
            }
        }
        log = _log(game_pk=1000)
        rows = WeatherFeatures().extract(game_logs=[log], game_contexts=contexts)
        r = rows[0]
        assert r["weather_condition"] == "Cloudy"
        assert r["weather_temp"] == "72"
        assert r["weather_wind"] == "10 mph"

    def test_game_pk_not_in_contexts(self):
        contexts = {9999: {"weather_condition": "Clear"}}
        log = _log(game_pk=1000)
        rows = WeatherFeatures().extract(game_logs=[log], game_contexts=contexts)
        assert rows[0]["weather_condition"] is None

    def test_partial_weather_data(self):
        contexts = {1000: {"weather_temp": "85"}}
        log = _log(game_pk=1000)
        rows = WeatherFeatures().extract(game_logs=[log], game_contexts=contexts)
        r = rows[0]
        assert r["weather_temp"] == "85"
        assert r["weather_condition"] is None
        assert r["weather_wind"] is None

    def test_multiple_games(self):
        contexts = {
            1000: {"weather_condition": "Clear"},
            1001: {"weather_condition": "Rain"},
        }
        logs = [
            _log(game_pk=1000, date="2025-04-01"),
            _log(game_pk=1001, date="2025-04-02"),
        ]
        rows = WeatherFeatures().extract(game_logs=logs, game_contexts=contexts)
        assert rows[0]["weather_condition"] == "Clear"
        assert rows[1]["weather_condition"] == "Rain"

    def test_returns_all_keys(self):
        log = _log()
        rows = WeatherFeatures().extract(game_logs=[log])
        assert "weather_condition" in rows[0]
        assert "weather_temp" in rows[0]
        assert "weather_wind" in rows[0]
