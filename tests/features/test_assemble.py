from mlb_ml_lab.data.schemas import PlayerGameLog
from mlb_ml_lab.features.assemble import build_feature_matrix, describe_features


def _log(**kw) -> PlayerGameLog:
    defaults = {
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


class TestBuildFeatureMatrix:
    def test_single_game_merges_features(self):
        logs = [_log()]
        matrix = build_feature_matrix(logs)
        assert len(matrix) == 1
        r = matrix[0]
        assert r["player_id"] == 1
        assert r["game_pk"] == 1000
        assert r["date"] == "2025-04-01"
        assert "hits_last_5" in r
        assert "hit_rate_last_5" in r
        assert "is_home" in r
        assert "rest_days" in r
        assert "park_wOBA" in r
        assert "weather_condition" in r
        assert "opp_era" in r
        assert "xba" in r

    def test_multiple_games(self):
        logs = [
            _log(date="2025-04-01", game_pk=1),
            _log(date="2025-04-02", game_pk=2, player_id=2),
        ]
        matrix = build_feature_matrix(logs)
        assert len(matrix) == 2

    def test_no_duplicate_rows(self):
        logs = [_log(date="2025-04-01", game_pk=1)]
        matrix = build_feature_matrix(logs)
        pks = [r["game_pk"] for r in matrix]
        assert len(pks) == len(set(pks))

    def test_with_teams_data(self):
        teams = [
            {"id": 108, "venue": {"id": 19}},
            {"id": 145, "venue": {"id": 680}},
        ]
        logs = [_log(team_id=108, opponent_id=145, is_home=True)]
        matrix = build_feature_matrix(logs, teams=teams, season=2025)
        assert matrix[0]["park_wOBA"] > 1.0

    def test_with_game_contexts(self):
        contexts = {
            1000: {
                "weather_condition": "Clear",
                "weather_temp": "75",
                "weather_wind": "5 mph",
            }
        }
        logs = [_log(game_pk=1000)]
        matrix = build_feature_matrix(logs, extra_kwargs={"game_contexts": contexts})
        assert matrix[0]["weather_condition"] == "Clear"

    def test_with_opponent_pitching(self):
        pitching = {
            145: {
                "era": 3.50,
                "k_per_9": 9.2,
                "whip": 1.20,
                "ba_against": 0.235,
                "hr_per_9": 1.1,
            },
        }
        logs = [_log(team_id=108, opponent_id=145)]
        matrix = build_feature_matrix(
            logs, extra_kwargs={"opponent_pitching": pitching}
        )
        assert matrix[0]["opp_era"] == 3.50

    def test_all_optional_data_combined(self):
        teams = [{"id": 108, "venue": {"id": 19}}, {"id": 145, "venue": {"id": 680}}]
        contexts = {
            1000: {
                "weather_condition": "Cloudy",
                "weather_temp": "68",
                "weather_wind": "8 mph",
            },
        }
        pitching = {
            145: {
                "era": 4.00,
                "k_per_9": 8.5,
                "whip": 1.30,
                "ba_against": 0.250,
                "hr_per_9": 1.0,
            },
        }
        logs = [_log(team_id=108, opponent_id=145, is_home=True, game_pk=1000)]
        matrix = build_feature_matrix(
            logs,
            season=2025,
            teams=teams,
            extra_kwargs={
                "game_contexts": contexts,
                "opponent_pitching": pitching,
            },
        )
        r = matrix[0]
        assert r["park_wOBA"] > 1.0
        assert r["weather_condition"] == "Cloudy"
        assert r["opp_era"] == 4.00

    def test_without_data_defaults_to_none(self):
        logs = [_log()]
        matrix = build_feature_matrix(logs)
        r = matrix[0]
        assert r["park_wOBA"] == 1.0
        assert r["weather_condition"] is None
        assert r["opp_era"] is None
        assert r["xba"] is None


class TestDescribeFeatures:
    def test_returns_all_metas(self):
        metas = describe_features()
        assert len(metas) > 0
        for m in metas:
            assert m.name
            assert m.source

    def test_includes_new_features(self):
        metas = describe_features()
        names = {m.name for m in metas}
        assert "weather_condition" in names
        assert "opp_era" in names
