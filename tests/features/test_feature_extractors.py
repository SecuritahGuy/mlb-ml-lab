from __future__ import annotations

from typing import Any

from mlb_ml_lab.features.base import get_registry

_reg = get_registry()


def _log(
    player_id: int = 1,
    game_pk: int = 100,
    date: str = "2024-04-01",
    team_id: int = 111,
    opponent_id: int = 222,
    season: str = "2024",
) -> Any:
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
            "hits": 1,
            "at_bats": 4,
            "plate_appearances": 4,
            "home_runs": 0,
            "doubles": 0,
            "triples": 0,
            "runs": 0,
            "walks": 0,
            "strikeouts": 1,
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


def _extract(cls_name: str, **extra: Any) -> list[dict[str, Any]]:
    return _reg[cls_name]().extract(game_logs=[_log()], **extra)


class TestBullpenQualityFeatures:
    def test_metadata(self):
        assert "bullpen_era" in {
            m.name for m in _reg["BullpenQualityFeatures"]().features
        }

    def test_extract(self):
        rows = _extract("BullpenQualityFeatures", bullpen_stats={222: {"era": 3.5}})
        assert rows[0]["bullpen_era"] == 3.5

    def test_defaults_none_when_missing(self):
        assert _extract("BullpenQualityFeatures")[0]["bullpen_era"] is None


class TestGamePaceFeature:
    def test_metadata(self):
        assert len(_reg["GamePaceFeature"]().features) > 0

    def test_extract(self):
        rows = _extract(
            "GamePaceFeature", game_pace_stats={222: {"time_per_game": "2:35"}}
        )
        assert rows[0]["opp_pace_time_per_game"] == "2:35"

    def test_defaults_none_when_missing(self):
        assert _extract("GamePaceFeature")[0]["opp_pace_time_per_game"] is None


class TestInjuryFeatures:
    def test_metadata(self):
        assert "il_flag" in {m.name for m in _reg["InjuryFeatures"]().features}

    def test_on_il_at_game_date(self):
        rows = _extract(
            "InjuryFeatures",
            injury_data={"player_timelines": {1: [("2024-03-20", "il_placement")]}},
        )
        assert rows[0]["il_flag"] == 1

    def test_activated_before_game_date(self):
        rows = _extract(
            "InjuryFeatures",
            injury_data={
                "player_timelines": {
                    1: [("2024-03-20", "il_placement"), ("2024-03-28", "il_activation")]
                }
            },
        )
        assert rows[0]["il_flag"] == 0

    def test_empty_data_defaults(self):
        assert _extract("InjuryFeatures")[0]["il_flag"] == 0


class TestLeagueContextFeatures:
    def test_metadata(self):
        assert len(_reg["LeagueContextFeatures"]().features) > 0

    def test_extract(self):
        rows = _extract("LeagueContextFeatures", league_stats={"avg": 0.248})
        assert rows[0]["league_avg"] == 0.248


class TestOddsFeatures:
    def test_metadata(self):
        assert len(_reg["OddsFeatures"]().features) > 0

    def test_extract(self):
        rows = _extract("OddsFeatures", odds_by_game={(111, 100): {"team_ml": -150}})
        assert rows[0]["team_ml"] == -150


class TestStartingPitcherFeatures:
    def test_metadata(self):
        assert "opp_pitcher_k_per_9" in {
            m.name for m in _reg["StartingPitcherFeatures"]().features
        }

    def test_extract(self):
        rows = _extract(
            "StartingPitcherFeatures",
            game_contexts={
                100: {"home_probable_pitcher_id": 500, "away_probable_pitcher_id": 501}
            },
            pitcher_stats={
                501: {
                    "strikeoutsPer9Inn": 9.5,
                    "whip": 1.2,
                    "avg": ".230",
                    "homeRunsPer9": 0.8,
                    "strikeOuts": 80,
                    "battersFaced": 300,
                    "inningsPitched": "84.2",
                }
            },
            player_details={
                500: {"id": 500, "fullName": "Ace", "pitchHand": {"code": "R"}},
                1: {"fullName": "Batter", "batSide": {"code": "R"}},
            },
        )
        assert rows[0]["opp_pitcher_k_per_9"] == 9.5


class TestPlayerQualityFeatures:
    def test_metadata(self):
        assert "player_age" in {
            m.name for m in _reg["PlayerQualityFeatures"]().features
        }

    def test_extract(self):
        rows = _extract(
            "PlayerQualityFeatures",
            player_details={
                1: {"id": 1, "fullName": "Test", "birthDate": "1995-06-01"}
            },
        )
        assert rows[0]["player_age"] is not None


class TestRollingAdvancedMetrics:
    def test_metadata(self):
        assert "rolling_woba_10" in {
            m.name for m in _reg["RollingAdvancedMetrics"]().features
        }

    def test_extract_returns_row_per_log(self):
        rows = _reg["RollingAdvancedMetrics"]().extract(game_logs=[_log()])
        assert len(rows) == 1


class TestRollingStatcastFeatures:
    def test_metadata(self):
        assert len(_reg["RollingStatcastFeatures"]().features) > 0

    def test_extract(self):
        rows = _reg["RollingStatcastFeatures"]().extract(
            game_logs=[_log(date="2024-04-01")],
            statcast_pitch_data=[
                {
                    "batter": "1",
                    "game_date": "2024-03-25",
                    "game_pk": 99,
                    "launch_speed": "95.0",
                    "launch_angle": "12.0",
                }
            ],
        )
        assert rows[0]["sc_avg_ev_10"] == 95.0


class TestScheduleDensityFeatures:
    def test_metadata(self):
        assert len(_reg["ScheduleDensityFeatures"]().features) > 0

    def test_extract(self):
        rows = _extract(
            "ScheduleDensityFeatures",
            season_schedule=[{"game_pk": 100, "home_team_id": 111}],
        )
        assert len(rows) == 1


class TestStatcastAdvancedFeatures:
    def test_metadata(self):
        assert "brl_pa" in {m.name for m in _reg["StatcastAdvancedFeatures"]().features}

    def test_extract(self):
        rows = _extract(
            "StatcastAdvancedFeatures",
            statcast_batters=[{"player_id": "1", "brl_pa": "2.5"}],
        )
        assert rows[0]["brl_pa"] == 2.5

    def test_defaults_none_when_missing(self):
        assert _extract("StatcastAdvancedFeatures")[0]["brl_pa"] is None


class TestStreaksFeature:
    def test_metadata(self):
        assert len(_reg["StreaksFeature"]().features) > 0

    def test_extract(self):
        rows = _extract(
            "StreaksFeature", streaks_stats={1: {"hitting": 5, "onbase": 8}}
        )
        assert rows[0]["hitting_streak"] == 5
        assert rows[0]["onbase_streak"] == 8


class TestTeamLeadersFeature:
    def test_metadata(self):
        assert len(_reg["TeamLeadersFeature"]().features) > 0

    def test_extract(self):
        rows = _extract("TeamLeadersFeature", team_leaders={222: {"top_avg": 0.280}})
        assert rows[0]["opp_top_avg"] == 0.280


class TestTeamTrendFeatures:
    def test_metadata(self):
        assert len(_reg["TeamTrendFeatures"]().features) > 0

    def test_extract(self):
        assert len(_reg["TeamTrendFeatures"]().extract(game_logs=[_log()])) == 1


class TestUmpireFeatures:
    def test_metadata(self):
        names = {m.name for m in _reg["UmpireFeatures"]().features}
        assert "hp_umpire_id" in names and "umpire_game_count" in names

    def test_extract_with_context(self):
        rows = _extract(
            "UmpireFeatures",
            game_contexts={100: {"hp_umpire_id": 123}},
            umpire_game_counts={123: 15},
        )
        assert rows[0]["hp_umpire_id"] == 123 and rows[0]["umpire_game_count"] == 15

    def test_extract_missing_context_defaults_none(self):
        rows = _extract("UmpireFeatures")
        assert rows[0]["hp_umpire_id"] is None and rows[0]["umpire_game_count"] is None

    def test_extract_no_game_counts(self):
        rows = _extract("UmpireFeatures", game_contexts={100: {"hp_umpire_id": 123}})
        assert rows[0]["hp_umpire_id"] == 123 and rows[0]["umpire_game_count"] is None


class TestIdentityFeatures:
    def test_metadata(self):
        names = {m.name for m in _reg["IdentityFeatures"]().features}
        assert "team_id" in names and "month" in names and "position_code" in names

    def test_extract(self):
        rows = _extract("IdentityFeatures")
        assert rows[0]["team_id"] == 111
        assert rows[0]["opponent_id"] == 222
        assert rows[0]["month"] == 4
        assert rows[0]["position_code"] == "CF"

    def test_month_parsing(self):
        rows = _reg["IdentityFeatures"]().extract(game_logs=[_log(date="2024-10-15")])
        assert rows[0]["month"] == 10


class TestParkFactorFeaturesVenueId:
    def test_venue_id_present(self):
        rows = _extract(
            "ParkFactorFeatures",
            teams=[{"id": 111, "venue": {"id": 555}}],
        )
        assert rows[0]["venue_id"] is not None

    def test_venue_id_none_without_teams(self):
        rows = _extract("ParkFactorFeatures")
        assert rows[0]["venue_id"] is None
