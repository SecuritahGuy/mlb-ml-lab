from mibl.data.schemas import PlayerGameLog, RosterPlayer, TeamInfo


class TestTeamInfo:
    def test_from_dict(self, teams_data):
        raw = teams_data["teams"][0]
        team = TeamInfo.from_dict(raw)
        assert team.id == raw["id"]
        assert team.name == raw["name"]
        assert team.abbreviation == raw["abbreviation"]


class TestRosterPlayer:
    def test_from_dict(self, roster_data):
        raw = roster_data["roster"][0]
        player = RosterPlayer.from_dict(raw, team_id=108)
        assert player.id == raw["person"]["id"]
        assert player.full_name == raw["person"]["fullName"]
        assert player.team_id == 108


class TestPlayerGameLog:
    def test_from_split_dict(self, gamelog_data):
        split = gamelog_data["stats"][0]["splits"][0]
        log = PlayerGameLog.from_split_dict(split)
        assert log.player_id == 545361
        assert log.player_name == "Mike Trout"
        assert log.hits == 0
        assert log.at_bats == 2
        assert log.summary == "0-2 | BB, HBP"

    def test_all_splits_parse(self, gamelog_data):
        splits = gamelog_data["stats"][0]["splits"]
        logs = [PlayerGameLog.from_split_dict(s) for s in splits]
        assert len(logs) > 0
        for log in logs:
            assert log.player_id == 545361
            assert log.date
            assert log.game_pk

    def test_hits_over(self):
        log = PlayerGameLog(
            player_id=1, player_name="Test", team_id=108, opponent_id=145,
            date="2025-04-01", game_pk=1, is_home=True, is_win=True,
            game_type="R", season="2025", hits=2,
        )
        assert log.hits_over(0.5) is True
        assert log.hits_over(1.5) is True
        assert log.hits_over(2.5) is False
