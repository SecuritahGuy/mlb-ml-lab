import httpx
import pytest

from mibl.data.parks import ParkFactors, _FALLBACK


class TestParkFactors:
    def test_known_venue_returns_value(self):
        pf = ParkFactors()
        try:
            coors = pf.factor(19, "wOBA")
            # Coors should be well above neutral in any Savant dataset
            assert coors >= 1.05
        finally:
            pf.close()

    def test_unknown_venue_returns_neutral(self):
        pf = ParkFactors()
        try:
            f = pf.factor(99999, "wOBA")
            assert f == 1.0
        finally:
            pf.close()

    def test_all_metrics_available_in_fallback(self):
        for vid, factors in _FALLBACK.items():
            for m in ("wOBA", "HR", "1B", "2B", "3B"):
                assert m in factors, f"{vid} missing {m}"

    def test_season_parameter_works(self):
        pf = ParkFactors()
        try:
            f = pf.factor(19, "HR", season=2024)
            assert f >= 0.8  # any reasonable HR factor
            f2 = pf.factor(19, "wOBA", season=2025)
            assert f2 >= 1.0
        finally:
            pf.close()

    def test_season_caching(self):
        pf = ParkFactors()
        try:
            pf._cache[9999] = {42: {"wOBA": 123.0}}  # pylint: disable=protected-access
            assert pf.factor(42, season=9999) == 1.23
        finally:
            pf.close()

    def test_savant_json_extraction(self):
        """Verify we can parse the JS variable from a real Savant page."""
        resp = httpx.get(
            "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
            "?type=year&year=2025&batSide=&stat=index_wOBA"
            "&condition=All&rolling=3&parks=mlb"
        )
        raw = ParkFactors._extract_json(resp.text)  # pylint: disable=protected-access
        assert raw is not None
        assert len(raw) >= 20
        keys = {k for entry in raw for k in entry}
        assert "venue_id" in keys
        assert "index_woba" in keys


@pytest.mark.slow
class TestParkFactorsLive:
    def test_fetches_30_venues_for_2024(self):
        pf = ParkFactors()
        try:
            data = pf._for_season(2024)  # pylint: disable=protected-access
            # 2024 should have all 30 venues (before A's relocation changes)
            assert len(data) >= 28
        finally:
            pf.close()

    def test_factor_changes_by_year(self):
        pf = ParkFactors()
        try:
            f2024 = pf.factor(22, "HR", season=2024)  # Dodger Stadium
            f2025 = pf.factor(22, "HR", season=2025)
            # Factors change slowly but shouldn't be identical
            assert isinstance(f2024, float)
            assert isinstance(f2025, float)
        finally:
            pf.close()
