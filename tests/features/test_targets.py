from mibl.data.schemas import PlayerGameLog
from mibl.features.targets import make_targets


def _log(hits: int, **kw) -> PlayerGameLog:
    defaults = dict(
        player_id=1, player_name="A", team_id=108, opponent_id=145,
        date="2025-04-01", game_pk=1000, is_home=True, is_win=True,
        game_type="R", season="2025",
        at_bats=4, plate_appearances=4,
    )
    defaults.update(kw)
    return PlayerGameLog(hits=hits, **defaults)


class TestMakeTargets:
    def test_default_thresholds(self):
        logs = [_log(hits=0), _log(hits=1), _log(hits=2)]
        rows = make_targets(logs)
        assert rows[0]["target_0.5"] == 0
        assert rows[0]["target_1.5"] == 0
        assert rows[1]["target_0.5"] == 1
        assert rows[1]["target_1.5"] == 0
        assert rows[2]["target_0.5"] == 1
        assert rows[2]["target_1.5"] == 1

    def test_custom_thresholds(self):
        logs = [_log(hits=2)]
        rows = make_targets(logs, thresholds=[1.0, 2.5])
        assert rows[0]["target_1.0"] == 1
        assert rows[0]["target_2.5"] == 0

    def test_includes_raw_hits(self):
        logs = [_log(hits=3)]
        rows = make_targets(logs)
        assert rows[0]["hits"] == 3

    def test_has_identifying_keys(self):
        logs = [_log(hits=0, player_id=42, game_pk=777, date="2025-04-15")]
        rows = make_targets(logs)
        r = rows[0]
        assert r["player_id"] == 42
        assert r["game_pk"] == 777
        assert r["date"] == "2025-04-15"
