from __future__ import annotations

from mlb_ml_lab.features.log5 import _log5, Log5Features
from mlb_ml_lab.features.base import get_registry


def _log(
    player_id: int = 1,
    game_pk: int = 100,
    date: str = "2024-04-01",
    team_id: int = 111,
    opponent_id: int = 222,
    hits: int = 1,
    plate_appearances: int = 4,
    season: str = "2024",
):
    return type(
        "GameLog",
        (),
        {
            "player_id": player_id,
            "player_name": "Test Player",
            "game_pk": game_pk,
            "date": date,
            "team_id": team_id,
            "opponent_id": opponent_id,
            "season": season,
            "is_home": True,
            "hits": hits,
            "at_bats": plate_appearances,
            "plate_appearances": plate_appearances,
            "home_runs": 0,
            "doubles": 0,
            "triples": 0,
            "runs": 0,
            "walks": 0,
            "strikeouts": 0,
            "avg": ".250",
            "obp": ".250",
            "slg": ".250",
            "position_abbr": "CF",
            "innings_pitched": "0.0",
            "earned_runs": 0,
            "era": "0.00",
            "whip": "0.00",
            "batters_faced": 0,
            "games_started": 0,
            "wins": 0,
            "losses": 0,
            "saves": 0,
        },
    )()


class TestLog5Formula:
    def test_equal_rates(self):
        assert _log5(0.250, 0.250) == 0.5

    def test_batter_better(self):
        p = _log5(0.300, 0.250)
        assert p > 0.5

    def test_pitcher_better(self):
        p = _log5(0.200, 0.250)
        assert p < 0.5

    def test_extreme_batter(self):
        p = _log5(0.400, 0.200)
        assert p > 0.5

    def test_very_low_rates(self):
        p = _log5(0.100, 0.050)
        assert 0 < p < 1

    def test_zero_denom_returns_half(self):
        assert _log5(0.0, 0.0) == 0.5


class TestLog5Features:
    def test_registered(self):
        assert "Log5Features" in get_registry()

    def test_metadata(self):
        names = {m.name for m in Log5Features().features}
        assert "log5_hit_prob_5" in names
        assert "log5_hit_prob_10" in names
        assert "log5_hit_prob_20" in names
        assert "log5_vs_league" in names

    def test_empty_logs_returns_empty(self):
        rows = Log5Features().extract(game_logs=[])
        assert not rows

    def test_single_game_defaults_to_half(self):
        logs = [_log(player_id=1, game_pk=100, date="2024-04-01")]
        rows = Log5Features().extract(game_logs=logs)
        assert len(rows) == 1
        for col in ("log5_hit_prob_5", "log5_hit_prob_10", "log5_hit_prob_20", "log5_vs_league"):
            assert 0 <= rows[0][col] <= 1

    def test_rolling_rates_computed(self):
        logs = [
            _log(player_id=1, game_pk=100, date="2024-04-01", hits=0, plate_appearances=4),
            _log(player_id=1, game_pk=101, date="2024-04-02", hits=2, plate_appearances=4),
            _log(player_id=1, game_pk=102, date="2024-04-03", hits=1, plate_appearances=4),
        ]
        rows = Log5Features().extract(game_logs=logs)
        # Row 0: no prior games → default 0.5 batter rate
        # Row 1: prior 1 game has 0/4 = 0.0 → log5(0.0, opp_rate)
        # Row 2: prior games have (0+2)/(4+4) = 2/8 = 0.25 → log5(0.25, opp_rate)
        assert len(rows) == 3

    def test_opponent_rate_affects_prob(self):
        logs = [
            _log(player_id=1, game_pk=100, date="2024-04-01", team_id=111, opponent_id=222,
                 hits=0, plate_appearances=4),
            _log(player_id=2, game_pk=100, date="2024-04-01", team_id=222, opponent_id=111,
                 hits=0, plate_appearances=4),
            _log(player_id=1, game_pk=101, date="2024-04-02", team_id=111, opponent_id=222,
                 hits=1, plate_appearances=4),
            _log(player_id=2, game_pk=101, date="2024-04-02", team_id=222, opponent_id=111,
                 hits=0, plate_appearances=4),
            _log(player_id=1, game_pk=102, date="2024-04-03", team_id=111, opponent_id=222,
                 hits=1, plate_appearances=4),
        ]
        rows = Log5Features().extract(game_logs=logs)
        # Player 1's 3rd game: batter hit rate = (1+0)/(4+4) = 0.125 vs opp team
        # Opponent team (222) hit rate allowed = (0+0)/(4+4) = 0.0
        row_2 = [r for r in rows if r["game_pk"] == 102 and r["player_id"] == 1]
        assert len(row_2) == 1
        assert 0 <= row_2[0]["log5_hit_prob_5"] <= 1
