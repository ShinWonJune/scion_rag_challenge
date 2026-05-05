from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class ThrottleCounters:
    request_count: int = 0
    rate_limit_count: int = 0


class AimdThrottle:
    def __init__(
        self,
        max_concurrency: int = 2,
        min_interval_sec: float = 0.5,
        increase_step: float = 0.25,
        decrease_factor: float = 0.9,
        decrease_after_success: int = 20,
        interval_cap_sec: float = 10.0,
        fixed_concurrency: bool = False,
    ) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.increase_step = max(0.0, float(increase_step))
        self.decrease_factor = float(decrease_factor)
        self.decrease_after_success = max(1, int(decrease_after_success))
        self.interval_cap_sec = max(self.min_interval_sec, float(interval_cap_sec))
        self.fixed_concurrency = fixed_concurrency

        self._semaphore = threading.Semaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0
        self._success_streak = 0
        self._counters = ThrottleCounters()

    def acquire(self) -> None:
        self._semaphore.acquire()
        with self._lock:
            wait_sec = max(0.0, self._next_allowed_at - time.monotonic())
        if wait_sec > 0:
            time.sleep(wait_sec)
        with self._lock:
            self._counters.request_count += 1
            self._next_allowed_at = max(time.monotonic(), self._next_allowed_at) + self.min_interval_sec

    def release(self) -> None:
        self._semaphore.release()

    def report_success(self) -> None:
        with self._lock:
            self._success_streak += 1
            if self.fixed_concurrency:
                return
            if self._success_streak >= self.decrease_after_success:
                self.min_interval_sec = max(0.1, self.min_interval_sec * self.decrease_factor)
                self._success_streak = 0

    def report_429(self, retry_after_sec: float | None = None) -> None:
        with self._lock:
            self._counters.rate_limit_count += 1
            self._success_streak = 0
            if not self.fixed_concurrency:
                increased = self.min_interval_sec * (1.0 + self.increase_step)
                self.min_interval_sec = min(self.interval_cap_sec, max(self.min_interval_sec, increased))
            if retry_after_sec:
                self._next_allowed_at = max(
                    self._next_allowed_at,
                    time.monotonic() + max(0.0, float(retry_after_sec)),
                )

    def get_counters(self) -> dict[str, float | int | bool]:
        with self._lock:
            return {
                "request_count": self._counters.request_count,
                "rate_limit_count": self._counters.rate_limit_count,
                "min_interval_sec": self.min_interval_sec,
                "max_concurrency": self.max_concurrency,
                "fixed_concurrency": self.fixed_concurrency,
            }
