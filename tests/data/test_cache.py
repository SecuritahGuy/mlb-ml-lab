from __future__ import annotations

import time

from mlb_ml_lab.data.cache import DiskCache


class TestDiskCache:
    def test_set_and_get(self, tmp_path):
        cache = DiskCache(str(tmp_path / "cache"))
        cache.set("test_key", {"foo": "bar"})
        assert cache.get("test_key") == {"foo": "bar"}

    def test_missing_key(self, tmp_path):
        cache = DiskCache(str(tmp_path / "cache"))
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, tmp_path):
        cache = DiskCache(str(tmp_path / "cache"), default_ttl=1)
        cache.set("soon_gone", "data")
        time.sleep(1.1)
        assert cache.get("soon_gone") is None

    def test_custom_ttl(self, tmp_path):
        cache = DiskCache(str(tmp_path / "cache"))
        cache.set("short", "data", ttl=0)
        time.sleep(0.1)
        assert cache.get("short") is None

    def test_clear(self, tmp_path):
        cache = DiskCache(str(tmp_path / "cache"))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_special_chars_in_key(self, tmp_path):
        cache = DiskCache(str(tmp_path / "cache"))
        key = "/people/545361/stats?stats=gameLog&season=2025"
        cache.set(key, "ok")
        assert cache.get(key) == "ok"
