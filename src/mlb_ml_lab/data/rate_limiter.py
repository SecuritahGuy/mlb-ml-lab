from __future__ import annotations

import time
import threading


class TokenBucket:
    """Simple token-bucket rate limiter.

    Usage::

        limiter = TokenBucket(capacity=10, refill_rate=10)  # 10 req/s
        with limiter:
            ...  # will block if no tokens available
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        with self._lock:
            self._refill()
            while self._tokens < tokens:
                sleep_for = (tokens - self._tokens) / self._refill_rate
                self._lock.release()
                time.sleep(sleep_for)
                self._lock.acquire()  # pylint: disable=consider-using-with
                self._refill()
            self._tokens -= tokens

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def __enter__(self) -> TokenBucket:
        return self

    def __exit__(self, *args: object) -> None:
        self.acquire()
