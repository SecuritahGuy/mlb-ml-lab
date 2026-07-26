from __future__ import annotations

from mlb_ml_lab.data.player_ids import PlayerIdResolver


class TestPlayerIdResolver:
    def test_lookup_by_mlbam(self):
        r = PlayerIdResolver()
        ids = r.lookup(mlbam=660271)
        assert ids is not None
        assert ids["name_last"] == "Ohtani"
        assert ids["fangraphs"] == 19755

    def test_lookup_by_fangraphs(self):
        r = PlayerIdResolver()
        ids = r.lookup(fangraphs=19755)
        assert ids is not None
        assert ids["mlbam"] == 660271

    def test_lookup_by_bref(self):
        r = PlayerIdResolver()
        ids = r.lookup(bref="troutmi01")
        assert ids is not None
        assert ids["mlbam"] == 545361

    def test_lookup_by_retrosheet(self):
        r = PlayerIdResolver()
        ids = r.lookup(retrosheet="troum001")
        assert ids is not None
        assert ids["mlbam"] == 545361

    def test_resolve_method(self):
        r = PlayerIdResolver()
        ids = r.resolve(660271, source="mlbam")
        assert ids is not None
        assert ids["name_first"] == "Shohei"

    def test_lookup_unknown_returns_none(self):
        r = PlayerIdResolver()
        assert r.lookup(mlbam=999999999) is None

    def test_search_by_name(self):
        r = PlayerIdResolver()
        results = r.search("Ohtani")
        assert len(results) >= 1
        assert any(p["mlbam"] == 660271 for p in results)

    def test_search_partial_name(self):
        r = PlayerIdResolver()
        results = r.search("trout")
        assert len(results) >= 1

    def test_len(self):
        r = PlayerIdResolver()
        assert len(r) >= 10
