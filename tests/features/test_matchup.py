from mibl.data.schemas import PlayerGameLog
from mibl.features.matchup import TeamPitchingFeatures


def _log(**kwargs) -> PlayerGameLog:
    defaults = {
        "player_id": 1, "player_name": "A", "team_id": 108, "opponent_id": 145,
        "date": "2025-04-01", "game_pk": 1000, "is_home": True, "is_win": True,
        "game_type": "R", "season": "2025",
        "hits": 0, "at_bats": 4, "plate_appearances": 4,
    }
    defaults.update(kwargs)
    return PlayerGameLog(**defaults)


class TestTeamPitchingFeatures:
    def test_no_pitching_data_all_none(self):
        log = _log()
        rows = TeamPitchingFeatures().extract(game_logs=[log])
        r = rows[0]
        assert r["opp_era"] is None
        assert r["opp_k_per_9"] is None
        assert r["opp_whip"] is None
        assert r["opp_ba_against"] is None
        assert r["opp_hr_per_9"] is None

    def test_opponent_lookup_by_opponent_id(self):
        pitching = {
            145: {
                "era": 3.50,
                "k_per_9": 9.2,
                "whip": 1.20,
                "ba_against": 0.235,
                "hr_per_9": 1.1,
            }
        }
        log = _log(team_id=108, opponent_id=145)
        rows = TeamPitchingFeatures().extract(game_logs=[log], opponent_pitching=pitching)
        r = rows[0]
        assert r["opp_era"] == 3.50
        assert r["opp_k_per_9"] == 9.2
        assert r["opp_whip"] == 1.20
        assert r["opp_ba_against"] == 0.235
        assert r["opp_hr_per_9"] == 1.1

    def test_uses_opponent_not_player_team(self):
        pitching = {
            108: {"era": 2.00},
            145: {"era": 4.50},
        }
        log = _log(team_id=108, opponent_id=145)
        rows = TeamPitchingFeatures().extract(game_logs=[log], opponent_pitching=pitching)
        assert rows[0]["opp_era"] == 4.50

    def test_team_not_in_dict_all_none(self):
        pitching = {999: {"era": 3.00}}
        log = _log(opponent_id=145)
        rows = TeamPitchingFeatures().extract(game_logs=[log], opponent_pitching=pitching)
        assert rows[0]["opp_era"] is None

    def test_partial_stat_dict(self):
        pitching = {145: {"era": 4.00}}
        log = _log(opponent_id=145)
        rows = TeamPitchingFeatures().extract(game_logs=[log], opponent_pitching=pitching)
        r = rows[0]
        assert r["opp_era"] == 4.00
        assert r["opp_k_per_9"] is None
        assert r["opp_whip"] is None

    def test_multiple_games_different_opponents(self):
        pitching = {
            145: {"era": 3.00},
            108: {"era": 4.20},
        }
        logs = [
            _log(game_pk=1, team_id=108, opponent_id=145),
            _log(game_pk=2, team_id=145, opponent_id=108),
        ]
        rows = TeamPitchingFeatures().extract(game_logs=logs, opponent_pitching=pitching)
        assert rows[0]["opp_era"] == 3.00
        assert rows[1]["opp_era"] == 4.20

    def test_returns_all_keys_present(self):
        log = _log()
        rows = TeamPitchingFeatures().extract(game_logs=[log])
        r = rows[0]
        assert "opp_era" in r
        assert "opp_k_per_9" in r
        assert "opp_whip" in r
        assert "opp_ba_against" in r
        assert "opp_hr_per_9" in r
