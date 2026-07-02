from mibl.data.schemas import PlayerGameLog
from mibl.features.rolling import RollingHits, RollingPlateAppearances, RollingBABIP


def _make_logs() -> list[PlayerGameLog]:
    logs: list[PlayerGameLog] = []
    dates = ["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05", "2025-04-06"]
    for i, d in enumerate(dates):
        logs.append(PlayerGameLog(
            player_id=1, player_name="A", team_id=108, opponent_id=145,
            date=d, game_pk=1000 + i, is_home=True, is_win=True,
            game_type="R", season="2025",
            hits=i % 3, at_bats=4, plate_appearances=5,
            walks=i % 2, strikeouts=1, home_runs=0,
        ))
    for i, d in enumerate(dates[:4]):
        logs.append(PlayerGameLog(
            player_id=2, player_name="B", team_id=145, opponent_id=108,
            date=d, game_pk=2000 + i, is_home=False, is_win=False,
            game_type="R", season="2025",
            hits=1, at_bats=4, plate_appearances=4,
            walks=0, strikeouts=1, home_runs=0,
        ))
    return logs


class TestRollingHits:
    def test_window_length(self):
        logs = _make_logs()
        extractor = RollingHits(windows=[3])
        rows = extractor.extract(game_logs=logs)
        by_pk = {r["game_pk"]: r for r in rows}
        assert by_pk[1000]["hits_last_3"] == 0
        assert by_pk[1000]["hit_rate_last_3"] is None
        assert by_pk[1003]["hits_last_3"] == 3
        assert by_pk[1003]["hit_rate_last_3"] == 1.0
        assert by_pk[1005]["hits_last_3"] == 3

    def test_includes_all_games(self):
        logs = _make_logs()
        rows = RollingHits(windows=[3]).extract(game_logs=logs)
        assert len(rows) == len(logs)

    def test_multiple_windows(self):
        logs = _make_logs()
        rows = RollingHits(windows=[2, 5]).extract(game_logs=logs)
        r = rows[0]
        assert "hits_last_2" in r
        assert "hits_last_5" in r
        assert "hit_rate_last_2" in r
        assert "hit_rate_last_5" in r

    def test_two_players(self):
        logs = _make_logs()
        rows = RollingHits(windows=[3]).extract(game_logs=logs)
        player_ids = {r["player_id"] for r in rows}
        assert player_ids == {1, 2}


class TestRollingPlateAppearances:
    def test_avg_pa_computed(self):
        logs = _make_logs()
        rows = RollingPlateAppearances(windows=[3]).extract(game_logs=logs)
        by_pk = {r["game_pk"]: r for r in rows}
        assert by_pk[1003]["avg_pa_last_3"] == 5.0

    def test_bb_rate(self):
        logs = _make_logs()
        rows = RollingPlateAppearances(windows=[3]).extract(game_logs=logs)
        by_pk = {r["game_pk"]: r for r in rows}
        r = by_pk[1003]
        assert r["bb_rate_last_3"] is not None

    def test_null_before_warmup(self):
        logs = _make_logs()
        rows = RollingPlateAppearances(windows=[5]).extract(game_logs=logs)
        p2_rows = [r for r in rows if r["player_id"] == 2]
        assert p2_rows[-1]["avg_pa_last_5"] is None


class TestRollingBABIP:  # pylint: disable=too-few-public-methods
    def test_babip_computed(self):
        logs = _make_logs()
        rows = RollingBABIP(window=3).extract(game_logs=logs)
        by_pk = {r["game_pk"]: r for r in rows}
        r = by_pk[1003]
        assert "babip_last_20" in r
