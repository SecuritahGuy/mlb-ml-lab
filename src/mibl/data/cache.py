from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class DiskCache:
    """Simple JSON-on-disk cache keyed by string keys.

    Each cached entry is a separate file under *cache_dir*.  Entries
    expire after *default_ttl* seconds (overridable per-set).
    """

    def __init__(self, cache_dir: str, default_ttl: int = 86400) -> None:
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        expires = entry.get("_expires", 0)
        if time.time() > expires:
            path.unlink(missing_ok=True)
            return None
        return entry["_data"]

    def set(self, key: str, data: Any, ttl: int | None = None) -> None:
        entry: dict[str, Any] = {
            "_data": data,
            "_expires": time.time() + (ttl if ttl is not None else self._default_ttl),
        }
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f)

    def clear(self) -> None:
        for child in self._root.iterdir():
            if child.is_file():
                child.unlink()

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_").replace("&", "_")
        return self._root / f"{safe}.json"
