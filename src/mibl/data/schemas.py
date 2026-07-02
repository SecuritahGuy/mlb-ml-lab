from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TeamInfo:
    id: int
    name: str
    abbreviation: str
    team_name: str
    location_name: str
    league_id: int | None = None
    division_id: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TeamInfo:
        return cls(
            id=d["id"],
            name=d["name"],
            abbreviation=d.get("abbreviation", ""),
            team_name=d.get("teamName", ""),
            location_name=d.get("locationName", ""),
            league_id=d.get("league", {}).get("id"),
            division_id=d.get("division", {}).get("id"),
        )


@dataclass
class RosterPlayer:
    id: int
    full_name: str
    position: str
    team_id: int
    status_code: str = ""
    status_description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], team_id: int) -> RosterPlayer:
        person = d.get("person", {})
        pos = d.get("position", {})
        status = d.get("status", {})
        return cls(
            id=person["id"],
            full_name=person.get("fullName", ""),
            position=pos.get("abbreviation", ""),
            team_id=team_id,
            status_code=status.get("code", ""),
            status_description=status.get("description", ""),
        )


@dataclass
class PlayerDetail:
    """Full player details from the /people endpoint."""

    id: int
    full_name: str
    first_name: str
    last_name: str
    primary_number: str = ""
    birth_date: str = ""
    current_age: int = 0
    birth_city: str = ""
    birth_state_province: str = ""
    birth_country: str = ""
    height: str = ""
    weight: int = 0
    active: bool = True
    primary_position: str = ""
    bats: str = ""
    throws: str = ""
    mlb_debut_date: str = ""
    draft_year: int | None = None
    current_team_id: int | None = None
    current_team_name: str = ""
    strike_zone_top: float = 0.0
    strike_zone_bottom: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlayerDetail:
        pos = d.get("primaryPosition", {}) or {}
        bats = d.get("batSide", {}) or {}
        throws = d.get("pitchHand", {}) or {}
        team = d.get("currentTeam", {}) or {}
        return cls(
            id=d["id"],
            full_name=d.get("fullName", ""),
            first_name=d.get("firstName", ""),
            last_name=d.get("lastName", ""),
            primary_number=d.get("primaryNumber", ""),
            birth_date=d.get("birthDate", ""),
            current_age=d.get("currentAge", 0),
            birth_city=d.get("birthCity", ""),
            birth_state_province=d.get("birthStateProvince", ""),
            birth_country=d.get("birthCountry", ""),
            height=d.get("height", ""),
            weight=d.get("weight", 0),
            active=d.get("active", True),
            primary_position=pos.get("abbreviation", ""),
            bats=bats.get("code", ""),
            throws=throws.get("code", ""),
            mlb_debut_date=d.get("mlbDebutDate", ""),
            draft_year=d.get("draftYear"),
            current_team_id=team.get("id"),
            current_team_name=team.get("name", ""),
            strike_zone_top=d.get("strikeZoneTop", 0.0),
            strike_zone_bottom=d.get("strikeZoneBottom", 0.0),
        )


@dataclass
class VenueInfo:
    """Venue metadata from the MLB Stats API.

    Note: the API returns minimal venue data (name, location, active status).
    Fields like capacity/surface are only available through team data or
    external sources.
    """

    id: int
    name: str
    location_city: str = ""
    location_state: str = ""
    location_country: str = ""
    active: bool = True
    season: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VenueInfo:
        loc = d.get("location", {}) or {}
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            location_city=loc.get("city", ""),
            location_state=loc.get("state", ""),
            location_country=loc.get("country", ""),
            active=d.get("active", True),
            season=d.get("season", ""),
        )


@dataclass
class StandingRecord:
    team_id: int
    team_name: str
    team_abbreviation: str
    league_id: int
    division_id: int
    division_name: str
    wins: int
    losses: int
    win_pct: str
    games_back: str
    wild_card_games_back: str
    division_rank: int
    league_rank: int
    runs_scored: int
    runs_allowed: int
    streak: str = ""
    clinch_indicator: str = ""
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    last_10_wins: int = 0
    last_10_losses: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StandingRecord:
        team = d.get("team", {}) or {}
        league_record = d.get("leagueRecord", {}) or {}
        records = d.get("records", {}) or {}
        home_rec = records.get("home", {}) or {}
        away_rec = records.get("away", {}) or {}
        last10_rec = records.get("lastTen", {}) or {}
        return cls(
            team_id=team.get("id", 0),
            team_name=team.get("name", ""),
            team_abbreviation=team.get("abbreviation", ""),
            league_id=d.get("league", {}).get("id", 0),
            division_id=d.get("division", {}).get("id", 0),
            division_name=d.get("division", {}).get("name", ""),
            wins=league_record.get("wins", 0),
            losses=league_record.get("losses", 0),
            win_pct=league_record.get("pct", ""),
            games_back=d.get("gamesBack", ""),
            wild_card_games_back=d.get("wildCardGamesBack", ""),
            division_rank=int(d.get("divisionRank", 0)),
            league_rank=int(d.get("leagueRank", 0)),
            runs_scored=d.get("runsScored", 0),
            runs_allowed=d.get("runsAllowed", 0),
            streak=d.get("streak", {}).get("description", ""),
            clinch_indicator=d.get("clinchIndicator", ""),
            home_wins=home_rec.get("wins", 0),
            home_losses=home_rec.get("losses", 0),
            away_wins=away_rec.get("wins", 0),
            away_losses=away_rec.get("losses", 0),
            last_10_wins=last10_rec.get("wins", 0),
            last_10_losses=last10_rec.get("losses", 0),
        )


@dataclass
class BoxscorePlayer:
    """Player entry in a game boxscore.

    ``batting_order`` is a raw integer from the API (e.g. 600 = 6th spot).
    ``is_substitute`` is True when the player entered mid-game.
    """

    player_id: int
    full_name: str
    position: str
    batting_order: int | None
    is_current_batter: bool = False
    is_current_pitcher: bool = False
    is_substitute: bool = False
    is_on_bench: bool = False
    # Game stats
    at_bats: int = 0
    runs: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    rbi: int = 0
    walks: int = 0
    strikeouts: int = 0
    avg: str = ""
    obp: str = ""
    slg: str = ""
    ops: str = ""

    @classmethod
    def from_dict(cls, pid: str, d: dict[str, Any]) -> BoxscorePlayer:
        person = d.get("person", {}) or {}
        pos = d.get("position", {}) or {}
        stats = d.get("stats", {}) or {}
        batting = stats.get("batting", {}) or {}
        gs = d.get("gameStatus", {}) or {}
        return cls(
            player_id=person.get("id", 0),
            full_name=person.get("fullName", ""),
            position=pos.get("abbreviation", ""),
            batting_order=d.get("battingOrder"),
            is_current_batter=gs.get("isCurrentBatter", False),
            is_current_pitcher=gs.get("isCurrentPitcher", False),
            is_substitute=gs.get("isSubstitute", False),
            is_on_bench=gs.get("isOnBench", False),
            at_bats=batting.get("atBats", 0),
            runs=batting.get("runs", 0),
            hits=batting.get("hits", 0),
            doubles=batting.get("doubles", 0),
            triples=batting.get("triples", 0),
            home_runs=batting.get("homeRuns", 0),
            rbi=batting.get("rbi", 0),
            walks=batting.get("baseOnBalls", 0),
            strikeouts=batting.get("strikeOuts", 0),
            avg=batting.get("avg", ""),
            obp=batting.get("obp", ""),
            slg=batting.get("slg", ""),
            ops=batting.get("ops", ""),
        )


@dataclass
class PlayerGameLog:
    player_id: int
    player_name: str
    team_id: int
    opponent_id: int
    date: str
    game_pk: int
    is_home: bool
    is_win: bool
    game_type: str
    season: str

    hits: int = 0
    at_bats: int = 0
    plate_appearances: int = 0
    runs: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    rbi: int = 0
    walks: int = 0
    strikeouts: int = 0
    avg: str = ".---"
    obp: str = ".---"
    slg: str = ".---"
    position_abbr: str = ""
    summary: str = ""

    # Pitching-specific fields (populated when group='pitching')
    innings_pitched: str = ""
    earned_runs: int = 0
    era: str = ""
    whip: str = ""
    batters_faced: int = 0
    games_started: int = 0
    complete_games: int = 0
    shutouts: int = 0
    wins: int = 0
    losses: int = 0
    saves: int = 0
    holds: int = 0
    blown_saves: int = 0
    hit_batsmen: int = 0
    wild_pitches: int = 0
    balks: int = 0
    inherited_runners: int = 0
    inherited_runners_scored: int = 0

    def hits_over(self, threshold: float) -> bool:
        return self.hits > threshold

    @classmethod
    def from_split_dict(cls, split: dict[str, Any]) -> PlayerGameLog:
        stat = split.get("stat", {})
        player = split.get("player", {})
        team = split.get("team", {})
        opponent = split.get("opponent", {})
        game = split.get("game", {})
        positions = split.get("positionsPlayed", [])

        return cls(
            player_id=player.get("id", 0),
            player_name=player.get("fullName", ""),
            team_id=team.get("id", 0),
            opponent_id=opponent.get("id", 0),
            date=split.get("date", ""),
            game_pk=game.get("gamePk", 0),
            is_home=split.get("isHome", False),
            is_win=split.get("isWin", False),
            game_type=split.get("gameType", ""),
            season=split.get("season", ""),
            hits=int(stat.get("hits", 0)),
            at_bats=int(stat.get("atBats", 0)),
            plate_appearances=int(stat.get("plateAppearances", 0)),
            runs=int(stat.get("runs", 0)),
            doubles=int(stat.get("doubles", 0)),
            triples=int(stat.get("triples", 0)),
            home_runs=int(stat.get("homeRuns", 0)),
            rbi=int(stat.get("rbi", 0)),
            walks=int(stat.get("baseOnBalls", 0)),
            strikeouts=int(stat.get("strikeOuts", 0)),
            avg=stat.get("avg", ".---"),
            obp=stat.get("obp", ".---"),
            slg=stat.get("slg", ".---"),
            position_abbr=positions[0].get("abbreviation", "") if positions else "",
            summary=stat.get("summary", ""),
            innings_pitched=stat.get("inningsPitched", ""),
            earned_runs=int(stat.get("earnedRuns", 0)),
            era=stat.get("era", ""),
            whip=stat.get("whip", ""),
            batters_faced=int(stat.get("battersFaced", 0)),
            games_started=int(stat.get("gamesStarted", 0)),
            complete_games=int(stat.get("completeGames", 0)),
            shutouts=int(stat.get("shutouts", 0)),
            wins=int(stat.get("wins", 0)),
            losses=int(stat.get("losses", 0)),
            saves=int(stat.get("saves", 0)),
            holds=int(stat.get("holds", 0)),
            blown_saves=int(stat.get("blownSaves", 0)),
            hit_batsmen=int(stat.get("hitBatsmen", 0)),
            wild_pitches=int(stat.get("wildPitches", 0)),
            balks=int(stat.get("balks", 0)),
            inherited_runners=int(stat.get("inheritedRunners", 0)),
            inherited_runners_scored=int(stat.get("inheritedRunnersScored", 0)),
        )
