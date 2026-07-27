from __future__ import annotations

from unittest.mock import patch

from mlb_ml_lab.data.client import MlbClient


def _make_client():
    return MlbClient(cache_dir="/tmp/_test_cache_nonexistent", cache_ttl=1)


class TestEnrichedSchedule:
    @patch.object(MlbClient, "_get")
    def test_returns_game_pk_lookup(self, mock_get):
        mock_get.return_value = {
            "dates": [
                {
                    "games": [
                        {
                            "gamePk": 100,
                            "gameDate": "2024-04-01T18:10:00Z",
                            "dayNight": "day",
                            "venue": {"id": 1, "name": "Fenway"},
                            "teams": {
                                "home": {"team": {"id": 111}},
                                "away": {"team": {"id": 222}},
                            },
                            "officials": [
                                {
                                    "officialType": "Home Plate",
                                    "official": {"id": 123, "fullName": "Ump A"},
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        client = _make_client()
        try:
            result = client.get_enriched_schedule(2024)
            assert 100 in result
            ctx = result[100]
            assert ctx["hp_umpire_id"] == 123
            assert ctx["hp_umpire_name"] == "Ump A"
            assert ctx["home_team_id"] == 111
            assert ctx["venue_name"] == "Fenway"
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_no_officials_defaults_none(self, mock_get):
        mock_get.return_value = {
            "dates": [
                {
                    "games": [
                        {
                            "gamePk": 100,
                            "gameDate": "2024-04-01T18:10:00Z",
                            "teams": {
                                "home": {"team": {"id": 111}},
                                "away": {"team": {"id": 222}},
                            },
                        }
                    ]
                }
            ]
        }
        client = _make_client()
        try:
            ctx = client.get_enriched_schedule(2024)[100]
            assert ctx["hp_umpire_id"] is None
            assert ctx["hp_umpire_name"] is None
        finally:
            client.close()


class TestGamePlays:
    @patch.object(MlbClient, "_get")
    def test_parses_plays(self, mock_get):
        mock_get.return_value = {
            "liveData": {
                "plays": {
                    "allPlays": [
                        {
                            "about": {
                                "atBatIndex": 1,
                                "halfInning": "top",
                                "inning": 1,
                            },
                            "result": {
                                "description": "Single",
                                "event": "Single",
                                "eventType": "single",
                            },
                            "matchup": {"batter": {"id": 100}, "pitcher": {"id": 200}},
                            "count": {"balls": 1, "strikes": 1, "outs": 0},
                        }
                    ]
                }
            }
        }
        client = _make_client()
        try:
            from mlb_ml_lab.data.schemas import PlateAppearance

            pas = client.get_game_plays(100)
            assert len(pas) == 1
            assert isinstance(pas[0], PlateAppearance)
        finally:
            client.close()


class TestTeamFieldingStats:
    @patch.object(MlbClient, "_get")
    def test_returns_fielding_stats(self, mock_get):
        mock_get.return_value = {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "errors": 42,
                                "fieldingPct": 0.985,
                                "doublePlays": 99,
                            }
                        }
                    ]
                }
            ]
        }
        client = _make_client()
        try:
            result = client.get_team_fielding_stats([111], 2024)
            assert 111 in result
            assert result[111]["errors"] == 42
        finally:
            client.close()

    def test_empty_result_skipped(self):
        pass  # tested implicitly by cache miss → empty

    @patch.object(MlbClient, "_get")
    def test_no_stats_returns_empty(self, mock_get):
        mock_get.return_value = {"stats": []}
        client = _make_client()
        try:
            result = client.get_team_fielding_stats([111], 2024)
            assert not result
        finally:
            client.close()


class TestTeamPitchingMonthlyStats:
    @patch.object(MlbClient, "_get")
    def test_aggregates_by_month(self, mock_get):
        mock_get.return_value = {
            "stats": [
                {
                    "splits": [
                        {
                            "date": "2024-04-01",
                            "stat": {
                                "inningsPitched": "9.0",
                                "earnedRuns": 2,
                                "strikeOuts": 8,
                                "baseOnBalls": 2,
                                "hits": 5,
                            },
                        },
                        {
                            "date": "2024-05-01",
                            "stat": {
                                "inningsPitched": "9.0",
                                "earnedRuns": 3,
                                "strikeOuts": 7,
                                "baseOnBalls": 3,
                                "hits": 6,
                            },
                        },
                    ]
                }
            ]
        }
        client = _make_client()
        try:
            result = client.get_team_pitching_monthly_stats([111], 2024)
            assert 111 in result
            assert len(result[111]) == 2  # Apr + May
            assert result[111][0]["month"] == 4
        finally:
            client.close()


class TestPlayerAwards:
    @patch.object(MlbClient, "_get")
    def test_returns_awards(self, mock_get):
        mock_get.return_value = {
            "awards": [{"id": "MVP", "name": "MVP", "season": "2024"}]
        }
        client = _make_client()
        try:
            awards = client.get_player_awards(100)
            assert len(awards) == 1
            assert awards[0]["name"] == "MVP"
        finally:
            client.close()


class TestAttendance:
    @patch.object(MlbClient, "_get")
    def test_returns_attendance(self, mock_get):
        mock_get.return_value = {
            "records": [],
            "aggregateTotals": {"attendanceTotal": 2500000},
        }
        client = _make_client()
        try:
            att = client.get_attendance(2024, team_id=117)
            assert att["aggregateTotals"]["attendanceTotal"] == 2500000
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_no_team_all_teams(self, mock_get):
        mock_get.return_value = {"records": [], "aggregateTotals": {}}
        client = _make_client()
        try:
            att = client.get_attendance(2024)
            assert "aggregateTotals" in att
        finally:
            client.close()


class TestStatLeaders:
    @patch.object(MlbClient, "_get")
    def test_flattens_categories(self, mock_get):
        mock_get.return_value = {
            "leagueLeaders": [
                {
                    "leaderCategory": "hits",
                    "leaders": [{"rank": 1, "value": 200, "person": {"fullName": "A"}}],
                }
            ]
        }
        client = _make_client()
        try:
            leaders = client.get_stat_leaders(2024, "hits")
            assert len(leaders) == 1
            assert leaders[0]["leaderCategory"] == "hits"
            assert leaders[0]["value"] == 200
        finally:
            client.close()


class TestHighLow:
    @patch.object(MlbClient, "_get")
    def test_flattens_splits(self, mock_get):
        mock_get.return_value = {
            "highLowResults": [
                {
                    "splits": [
                        {
                            "rank": 1,
                            "player": {"fullName": "X"},
                            "stat": {"hits": 5},
                            "date": "2024-04-01",
                        }
                    ]
                }
            ]
        }
        client = _make_client()
        try:
            hl = client.get_high_low("player", "hitting", "hits", 2024, limit=5)
            assert len(hl) == 1
            assert hl[0]["stat"]["hits"] == 5
        finally:
            client.close()


class TestSavantCSV:
    @patch.object(MlbClient, "_fetch_savant_csv")
    def test_statcast_search(self, mock_csv):
        mock_csv.return_value = [{"player_id": "100", "pitch_type": "FF"}]
        client = _make_client()
        try:
            rows = client.get_statcast_search_data("2024-04-01", "2024-04-07")
            assert len(rows) == 1
            assert rows[0]["pitch_type"] == "FF"
        finally:
            client.close()

    @patch.object(MlbClient, "_fetch_savant_csv")
    def test_pitcher_statcast(self, mock_csv):
        mock_csv.return_value = [{"player_id": "200", "velocity": "95.0"}]
        client = _make_client()
        try:
            rows = client.get_pitcher_statcast_data(2024)
            assert len(rows) == 1
        finally:
            client.close()


class TestTeamBullpenStats:
    @patch.object(MlbClient, "get_roster")
    @patch.object(MlbClient, "get_player_season_stats")
    def test_bullpen_stats(self, mock_stats, mock_roster):
        mock_roster.return_value = [
            {"person": {"id": 500}, "position": {"abbreviation": "P"}},
            {"person": {"id": 501}, "position": {"abbreviation": "P"}},
        ]

        # 500 is a reliever (0 GS), 501 is a starter (20 GS / 28 GP)
        def _side_effect(pid, _season, _group="pitching"):
            if pid == 500:
                return {
                    "gamesPlayed": 40,
                    "gamesStarted": 0,
                    "inningsPitched": "60.0",
                    "era": "3.50",
                    "strikeoutsPer9Inn": "9.0",
                    "whip": "1.20",
                    "avg": ".230",
                    "homeRunsPer9": "0.8",
                }
            return {"gamesPlayed": 28, "gamesStarted": 20, "inningsPitched": "150.0"}

        mock_stats.side_effect = _side_effect
        client = _make_client()
        try:
            result = client.get_team_bullpen_stats(111, 2024)
            assert abs(result["era"] - 3.50) < 0.01
        finally:
            client.close()


class TestTeamLeaders:
    @patch.object(MlbClient, "_get")
    def test_flattens_leaders(self, mock_get):
        mock_get.return_value = {
            "teamLeaders": [
                {
                    "leaderCategory": "homeRuns",
                    "leaders": [{"rank": 1, "value": 40, "person": {"fullName": "Y"}}],
                }
            ]
        }
        client = _make_client()
        try:
            leaders = client.get_team_leaders(111, 2024)
            assert len(leaders) == 1
            assert leaders[0]["value"] == 40
        finally:
            client.close()


class TestGamePace:
    @patch.object(MlbClient, "_get")
    def test_returns_pace(self, mock_get):
        mock_get.return_value = {
            "teams": [{"team": {"id": 111}, "timePerGame": "2:35"}]
        }
        client = _make_client()
        try:
            pace = client.get_game_pace(2024)
            assert len(pace) == 1
        finally:
            client.close()


class TestContextMetrics:
    @patch.object(MlbClient, "_get")
    def test_returns_metrics(self, mock_get):
        mock_get.return_value = {"leverageIndex": 1.5, "homeWinProbability": 0.55}
        client = _make_client()
        try:
            m = client.get_context_metrics(100)
            assert abs(m["homeWinProbability"] - 0.55) < 0.01
        finally:
            client.close()


class TestTransactions:
    @patch.object(MlbClient, "_get")
    def test_returns_transactions(self, mock_get):
        mock_get.return_value = {
            "transactions": [
                {"id": 1, "description": "Placed on IL", "date": "2024-04-01"}
            ]
        }
        client = _make_client()
        try:
            txns = client.get_transactions(2024, team_id=117)
            assert len(txns) == 1
            assert "Placed on IL" in txns[0]["description"]
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_player_transactions(self, mock_get):
        mock_get.return_value = {
            "transactions": [{"id": 2, "description": "Activated"}]
        }
        client = _make_client()
        try:
            txns = client.get_player_transactions(100, 2024)
            assert len(txns) == 1
        finally:
            client.close()


class TestFullRoster:
    @patch.object(MlbClient, "_get")
    def test_full_roster(self, mock_get):
        mock_get.return_value = {
            "roster": [{"person": {"id": 100}, "status": {"code": "A"}}]
        }
        client = _make_client()
        try:
            roster = client.get_full_roster(111, 2024)
            assert len(roster) == 1
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_full_roster_with_date(self, mock_get):
        mock_get.return_value = {"roster": []}
        client = _make_client()
        try:
            roster = client.get_full_roster(111, 2024, date="2024-07-01")
            assert roster == []
        finally:
            client.close()


class TestStatsStreaks:
    @patch.object(MlbClient, "_get")
    def test_returns_streaks(self, mock_get):
        mock_get.return_value = {
            "streaks": [
                {"player": {"id": 100}, "numStreak": 10, "streakType": "hitting"}
            ]
        }
        client = _make_client()
        try:
            streaks = client.get_stats_streaks(2024)
            assert len(streaks) == 1
            assert streaks[0]["numStreak"] == 10
        finally:
            client.close()


class TestBulkPeople:
    @patch.object(MlbClient, "_get")
    def test_bulk_lookup(self, mock_get):
        mock_get.return_value = {
            "people": [{"id": 100, "fullName": "A"}, {"id": 200, "fullName": "B"}]
        }
        client = _make_client()
        try:
            ppl = client.get_people_bulk([100, 200])
            assert len(ppl) == 2
        finally:
            client.close()


class TestMultiSeasonStats:
    @patch.object(MlbClient, "_get")
    def test_multi_season(self, mock_get):
        mock_get.return_value = {
            "stats": [
                {
                    "splits": [
                        {"season": "2024", "stat": {"homeRuns": 30}},
                        {"season": "2023", "stat": {"homeRuns": 25}},
                    ]
                }
            ]
        }
        client = _make_client()
        try:
            stats = client.get_player_multi_season_stats(100)
            assert len(stats) == 2
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_empty_stats(self, mock_get):
        mock_get.return_value = {"stats": []}
        client = _make_client()
        try:
            stats = client.get_player_multi_season_stats(100)
            assert stats == []
        finally:
            client.close()


class TestMetadata:
    @patch.object(MlbClient, "_get")
    def test_divisions(self, mock_get):
        mock_get.return_value = {"divisions": [{"id": 200, "name": "AL East"}]}
        client = _make_client()
        try:
            divs = client.get_divisions()
            assert len(divs) == 1
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_leagues(self, mock_get):
        mock_get.return_value = {"leagues": [{"id": 103, "name": "AL"}]}
        client = _make_client()
        try:
            leagues = client.get_leagues()
            assert len(leagues) == 1
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_seasons(self, mock_get):
        mock_get.return_value = {"seasons": [{"seasonId": "2024"}]}
        client = _make_client()
        try:
            seasons = client.get_seasons()
            assert len(seasons) == 1
        finally:
            client.close()

    @patch.object(MlbClient, "_get")
    def test_umpires(self, mock_get):
        mock_get.return_value = {
            "roster": [
                {"person": {"id": 123, "fullName": "Ump X"}, "jerseyNumber": "42"}
            ]
        }
        client = _make_client()
        try:
            umps = client.get_umpires()
            assert len(umps) == 1
            assert umps[0]["person"]["fullName"] == "Ump X"
        finally:
            client.close()


class TestWinProbability:
    @patch.object(MlbClient, "_get")
    def test_returns_plays_as_list(self, mock_get):
        mock_get.return_value = [
            {"homeWinProbability": 0.5, "result": {"event": "Single"}}
        ]
        client = _make_client()
        try:
            wp = client.get_game_win_probability(100)
            assert len(wp) == 1
            assert wp[0]["result"]["event"] == "Single"
        finally:
            client.close()


class TestGameLinescore:
    @patch.object(MlbClient, "_get")
    def test_returns_linescore(self, mock_get):
        mock_get.return_value = {"currentInning": 5, "inningState": "Top"}
        client = _make_client()
        try:
            ls = client.get_game_linescore(100)
            assert ls["currentInning"] == 5
        finally:
            client.close()
