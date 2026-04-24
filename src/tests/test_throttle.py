import time

from src.search.throttle import AimdThrottle


def test_throttle_interval_increases_after_429() -> None:
    throttle = AimdThrottle(min_interval_sec=0.05, fixed_concurrency=False)
    throttle.acquire()
    start = time.perf_counter()
    throttle.report_429(0.05)
    throttle.acquire()
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.04
