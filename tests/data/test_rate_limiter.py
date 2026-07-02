import time

from mlb_ml_lab.data.rate_limiter import TokenBucket


class TestTokenBucket:
    def test_allows_burst(self):
        limiter = TokenBucket(capacity=5, refill_rate=10)
        start = time.monotonic()
        for _ in range(5):
            limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # burst should be near-instant

    def test_throttles_excess(self):
        limiter = TokenBucket(capacity=2, refill_rate=2)
        start = time.monotonic()
        for _ in range(4):
            limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.5  # had to wait for refill

    def test_context_manager(self):
        limiter = TokenBucket(capacity=10, refill_rate=10)
        start = time.monotonic()
        with limiter:
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.2
