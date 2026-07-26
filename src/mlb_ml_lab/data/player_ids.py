"""Cross-source player ID resolver.

Maps player identities across MLBAM, FanGraphs, Baseball-Reference,
and Retrosheet ID systems using the Chadwick Bureau register.

Usage:
    from mlb_ml_lab.data.player_ids import PlayerIdResolver

    resolver = PlayerIdResolver()
    ids = resolver.lookup(mlbam=660271)  # Mike Trout
    # -> {"mlbam": 660271, "fangraphs": 10153, "bref": "troutmi01", "retrosheet": "trot001"}
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CHADWICK_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/"
    "chadwickbureau/register/master/data/people-{}.csv"
)
CHADWICK_PARTS = [f"{i:x}" for i in range(16)]  # people-0.csv .. people-f.csv
DEFAULT_CACHE_DIR = "data/player_ids"
BUNDLED_PATH = os.path.join(os.path.dirname(__file__), "register_sample.json")

ID_SOURCES = [
    ("key_mlbam", "mlbam", int),
    ("key_fangraphs", "fangraphs", int),
    ("key_bbref", "bref", str),
    ("key_retro", "retrosheet", str),
]

CHADWICK_COLS = [
    "name_last", "name_first", "key_mlbam", "key_retro", "key_bbref",
    "key_fangraphs", "key_bbref_minors", "key_cbs", "key_espn",
    "key_nfbc", "key_war_daily", "key_bpro",
]


def _safe_int(val: str) -> int | None:
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


class PlayerIdResolver:
    """Resolve player identities across MLBAM, FanGraphs, BREF, Retrosheet.

    Builds its mapping from the Chadwick Bureau register, cached locally
    at ``cache_dir``. Falls back to a bundled sample for offline use.

    Parameters
    ----------
    cache_dir : str, optional
        Directory to cache the Chadwick register CSV.
    auto_sync : bool
        Whether to auto-download the Chadwick register on init if missing.
    """

    def __init__(
        self, cache_dir: str = DEFAULT_CACHE_DIR, auto_sync: bool = False,
    ) -> None:
        self._cache_dir = cache_dir
        self._by_mlbam: dict[int, dict[str, Any]] = {}
        self._by_fangraphs: dict[int, int] = {}
        self._by_bref: dict[str, int] = {}
        self._by_retro: dict[str, int] = {}
        self._loaded = False
        if auto_sync:
            self.sync()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        mlbam: int | None = None,
        fangraphs: int | None = None,
        bref: str | None = None,
        retrosheet: str | None = None,
    ) -> dict[str, Any] | None:
        """Look up a player by any known ID.

        Returns a dict with all known IDs plus name, or None if not found.

        Examples
        --------
        >>> resolver.lookup(mlbam=660271)
        {"mlbam": 660271, "fangraphs": 10153, "bref": "troutmi01", ...}
        """
        self._ensure_loaded()
        mlbam_id = self._resolve_mlbam(mlbam, fangraphs, bref, retrosheet)
        if mlbam_id is None:
            return None
        return dict(self._by_mlbam[mlbam_id])

    def resolve(self, player_id: int, source: str = "mlbam") -> dict[str, Any] | None:
        """Resolve all known IDs for a player from a given source.

        Parameters
        ----------
        player_id : int
            The player's ID in the given source system.
        source : str
            One of ``"mlbam"``, ``"fangraphs"``, ``"bref"``, ``"retrosheet"``.

        Returns
        -------
        dict or None
        """
        kwargs = {source: player_id}
        return self.lookup(**kwargs)

    def search(self, name: str | None = None) -> list[dict[str, Any]]:
        """Search players by name fragment (case-insensitive).

        Returns a list of matching player records.
        """
        self._ensure_loaded()
        results: list[dict[str, Any]] = []
        name_lower = name.lower() if name else ""
        for record in self._by_mlbam.values():
            full = f"{record.get('name_first', '')} {record.get('name_last', '')}".lower()
            if name_lower and name_lower not in full:
                continue
            results.append(dict(record))
        return results

    def sync(self, force: bool = False) -> None:
        """Download the Chadwick register and build the mapping."""
        path = self._cache_path()
        if not force and os.path.isfile(path):
            logger.info("Loading cached Chadwick register from %s", path)
        else:
            self._download_register(path)
        self._build_from_csv(path)

    def sync_from_path(self, csv_path: str) -> None:
        """Build mapping from a local Chadwick register CSV."""
        self._build_from_csv(csv_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_path(self) -> str:
        os.makedirs(self._cache_dir, exist_ok=True)
        return os.path.join(self._cache_dir, "people.csv")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        cache_path = self._cache_path()
        if os.path.isfile(cache_path):
            self._build_from_csv(cache_path)
        elif os.path.isfile(BUNDLED_PATH):
            self._build_from_json(BUNDLED_PATH)
        else:
            logger.warning(
                "No Chadwick register found at %s and no bundled sample. "
                "Call resolver.sync() to download it.",
                cache_path,
            )
            self._by_mlbam = {}
            self._loaded = True

    def _download_register(self, path: str) -> None:
        import httpx
        total = 0
        with open(path, "w", encoding="utf-8") as out:
            for i, part in enumerate(CHADWICK_PARTS):
                url = CHADWICK_URL_TEMPLATE.format(part)
                logger.info("Downloading %s ...", url)
                resp = httpx.get(url, timeout=60.0, follow_redirects=True)
                resp.raise_for_status()
                text = resp.text
                if i == 0:
                    out.write(text)
                else:
                    lines = text.splitlines()
                    if lines:
                        out.write("\n".join(lines[1:]) + "\n")
                total += len(text)
        logger.info("Saved merged register to %s (%d bytes)", path, total)

    def _build_from_csv(self, path: str) -> None:
        self._by_mlbam = {}
        self._by_fangraphs = {}
        self._by_bref = {}
        self._by_retro = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mlbam = _safe_int(row.get("key_mlbam", ""))
                if mlbam is None:
                    continue
                fg = _safe_int(row.get("key_fangraphs", ""))
                bref = row.get("key_bbref", "") or None
                retro = row.get("key_retro", "") or None
                record: dict[str, Any] = {
                    "mlbam": mlbam,
                    "name_last": row.get("name_last", ""),
                    "name_first": row.get("name_first", ""),
                    "fangraphs": fg,
                    "bref": bref,
                    "retrosheet": retro,
                }
                self._by_mlbam[mlbam] = record
                if fg is not None:
                    self._by_fangraphs[fg] = mlbam
                if bref:
                    self._by_bref[bref] = mlbam
                if retro:
                    self._by_retro[retro] = mlbam
        self._loaded = True
        logger.info("Built mapping: %d players", len(self._by_mlbam))

    def _build_from_json(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._by_mlbam = {}
        self._by_fangraphs = {}
        self._by_bref = {}
        self._by_retro = {}
        for record in data:
            mlbam = record.get("mlbam")
            if mlbam is None:
                continue
            self._by_mlbam[mlbam] = record
            fg = record.get("fangraphs")
            if fg is not None:
                self._by_fangraphs[fg] = mlbam
            bref = record.get("bref")
            if bref:
                self._by_bref[bref] = mlbam
            retro = record.get("retrosheet")
            if retro:
                self._by_retro[retro] = mlbam
        self._loaded = True
        logger.info("Built mapping from bundled JSON: %d players", len(self._by_mlbam))

    def _resolve_mlbam(
        self, mlbam: int | None, fangraphs: int | None, bref: str | None, retrosheet: str | None,
    ) -> int | None:
        if mlbam is not None and mlbam in self._by_mlbam:
            return mlbam
        if fangraphs is not None and fangraphs in self._by_fangraphs:
            return self._by_fangraphs[fangraphs]
        if bref is not None and bref in self._by_bref:
            return self._by_bref[bref]
        if retrosheet is not None and retrosheet in self._by_retro:
            return self._by_retro[retrosheet]
        return None

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._by_mlbam)
