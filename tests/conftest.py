from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from mlb_ml_lab.data.client import MlbClient

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> dict[str, Any]:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


def load_csv(name: str) -> list[dict[str, str]]:
    with open(FIXTURES / name, encoding="utf-8-sig") as f:
        text = f.read()
    return list(csv.DictReader(io.StringIO(text)))


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow (live API) tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that hit the live MLB API")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---- shared fixtures ----


@pytest.fixture
def teams_data() -> dict[str, Any]:
    return load_json("teams.json")


@pytest.fixture
def roster_data() -> dict[str, Any]:
    return load_json("angels_roster_2025.json")


@pytest.fixture
def gamelog_data() -> dict[str, Any]:
    return load_json("trout_gamelog_2025.json")


@pytest.fixture
def game_feed_data() -> dict[str, Any]:
    return load_json("game_feed_778554.json")


@pytest.fixture
def statcast_data() -> list[dict[str, str]]:
    return load_csv("statcast_batters_2025.csv")


@pytest.fixture
def expected_stats_data() -> list[dict[str, str]]:
    return load_csv("expected_stats_2025.csv")


@pytest.fixture
def client_with_fixtures(tmp_path) -> MlbClient:
    """MlbClient with fixtures pre-loaded into its cache."""
    # pylint: disable=protected-access
    cache_dir = tmp_path / "cache"
    client = MlbClient(cache_dir=str(cache_dir), cache_ttl=86400)

    seeds = {
        "/teams?sportId=1": "teams.json",
        "/teams/108/roster?season=2025&rosterType=40Man": "angels_roster_2025.json",
        "/people/545361/stats?stats=gameLog&group=hitting&season=2025":
            "trout_gamelog_2025.json",
        "/game/778554/feed/live?": None,
        "savant:/leaderboard/statcast?csv=true&min=q&type=batter&year=2025":
            "statcast_batters_2025.csv",
        "savant:/leaderboard/expected_statistics?csv=true&min=q&type=batter&year=2025":
            "expected_stats_2025.csv",
    }

    for cache_key, fixture_name in seeds.items():
        if fixture_name is None:
            continue
        if fixture_name.endswith(".json"):
            client._cache.set(cache_key, load_json(fixture_name))
        else:
            client._cache.set(cache_key, load_csv(fixture_name))

    # Seed schedule from live API (one-time)
    resp = httpx.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "season": 2025, "gameType": "R"},
    )
    client._cache.set(
        "/schedule?sportId=1&gameType=R&season=2025",
        resp.json(),
    )
    # Seed game feed
    client._cache.set(
        "/game/778554/feed/live?",
        load_json("game_feed_778554.json"),
    )

    return client
