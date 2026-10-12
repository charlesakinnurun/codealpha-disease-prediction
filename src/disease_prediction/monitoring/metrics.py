"""Lightweight, dependency-free monitoring hooks.

Provides in-process counters and histograms for latency, errors, validation
failures and prediction-distribution drift signals. This is deliberately not a
full metrics platform — it exposes a ``/v1/metrics`` snapshot that a real
deployment could scrape or forward to Prometheus/OpenTelemetry. See
``docs/architecture.md`` for the drift-monitoring roadmap.

No patient feature values are recorded: only counts, task names, model names and
bucketed probability ranges.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_PROBA_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class MetricsRegistry:
    """Thread-safe, in-process counter and histogram registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._latency: dict[str, int] = defaultdict(int)
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)
        self._predictions: dict[str, int] = defaultdict(int)
        self._proba_hist: dict[str, list[int]] = defaultdict(
            lambda: [0] * (len(_PROBA_BUCKETS) - 1)
        )
        self._started = time.time()

    # -- counters ---------------------------------------------------------
    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    # -- latency ----------------------------------------------------------
    def observe_latency(self, name: str, seconds: float) -> None:
        bucket = _bucket_index(_LATENCY_BUCKETS, seconds)
        with self._lock:
            self._latency[f"{name}|le_{_LATENCY_BUCKETS[bucket]}"] += 1
            self._latency_sum[name] += seconds
            self._latency_count[name] += 1

    # -- predictions ------------------------------------------------------
    def record_prediction(self, task: str, model: str, probability: float, label: int) -> None:
        bucket = _bucket_index(_PROBA_BUCKETS, probability)
        with self._lock:
            self._predictions[f"{task}|{model}|label_{label}"] += 1
            self._proba_hist[f"{task}|{model}"][bucket] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latency = {}
            for key, count in self._latency_count.items():
                latency[key] = {
                    "count": count,
                    "sum_seconds": round(self._latency_sum[key], 6),
                    "mean_seconds": round(self._latency_sum[key] / count, 6) if count else None,
                }
            return {
                "uptime_seconds": round(time.time() - self._started, 3),
                "counters": dict(self._counters),
                "latency_histogram": dict(self._latency),
                "latency_summary": latency,
                "prediction_counts": dict(self._predictions),
                "probability_histogram": {
                    key: {
                        "buckets": list(_PROBA_BUCKETS),
                        "counts": counts,
                    }
                    for key, counts in self._proba_hist.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latency.clear()
            self._latency_sum.clear()
            self._latency_count.clear()
            self._predictions.clear()
            self._proba_hist.clear()


def _bucket_index(bounds: tuple[float, ...], value: float) -> int:
    for i, upper in enumerate(bounds[1:], start=0):
        if value <= upper:
            return i
    return len(bounds) - 2


@contextmanager
def timed(registry: MetricsRegistry, name: str):
    """Context manager recording the wall-clock duration of a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        registry.observe_latency(name, time.perf_counter() - start)


#: Process-wide registry used by the API. Tests may create their own instance.
metrics = MetricsRegistry()
