"""Lightweight wall-time and process-RSS profiling for production stages.

The profiler intentionally measures *process resident memory* (RSS/working set),
not only Python allocations.  This is important for NumPy/SciPy workloads whose
large buffers may live outside Python's object allocator.

``psutil`` is optional at import time so profiling can never break scientific
computation.  When it is unavailable, timing is still recorded and memory
fields are reported as unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time

try:  # pragma: no cover - availability is environment dependent.
    import psutil
except Exception:  # pragma: no cover
    psutil = None


_MIB = 1024.0 * 1024.0


def current_rss_bytes() -> int | None:
    """Return current process RSS/working-set bytes, if available."""
    if psutil is None:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def format_mib(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / _MIB:.1f} MiB"


class _RSSSampler:
    """Sample process RSS while one compute call is running.

    A 20 ms default interval keeps overhead negligible relative to the extrema
    calls on production-sized matrices while still catching short-lived SciPy /
    NumPy buffers much better than before/after measurements alone.
    """

    def __init__(self, interval: float = 0.02):
        self.interval = max(0.005, float(interval))
        self.before = current_rss_bytes()
        self.after = self.before
        self.peak = self.before
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if psutil is None:
            return
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="wavelet-rss-profiler",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self):
        while not self._stop.wait(self.interval):
            self._sample_once()

    def _sample_once(self):
        value = current_rss_bytes()
        if value is not None and (self.peak is None or value > self.peak):
            self.peak = value

    def stop(self):
        self._sample_once()
        self.after = current_rss_bytes()
        if self.after is not None and (self.peak is None or self.after > self.peak):
            self.peak = self.after
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval * 4.0))


@dataclass
class StageProfile:
    """Aggregate compute time and RSS statistics over repeated stage calls."""

    name: str
    sample_interval: float = 0.02
    compute_seconds: float = 0.0
    computed_calls: int = 0
    cache_hits: int = 0
    rss_start: int | None = None
    rss_end: int | None = None
    rss_peak: int | None = None
    max_call_rss_increase: int | None = None

    def measure(self):
        return _StageMeasurement(self)

    def record_cache_hit(self):
        self.cache_hits += 1

    def _record(self, elapsed: float, sampler: _RSSSampler):
        self.compute_seconds += float(elapsed)
        self.computed_calls += 1
        if self.rss_start is None:
            self.rss_start = sampler.before
        self.rss_end = sampler.after
        if sampler.peak is not None:
            if self.rss_peak is None or sampler.peak > self.rss_peak:
                self.rss_peak = sampler.peak
        if sampler.before is not None and sampler.peak is not None:
            increase = max(0, sampler.peak - sampler.before)
            if self.max_call_rss_increase is None or increase > self.max_call_rss_increase:
                self.max_call_rss_increase = increase

    def summary(self) -> str:
        return (
            f"{self.name}: compute={self.compute_seconds:.3f} s, "
            f"calls={self.computed_calls}, cache_hits={self.cache_hits}, "
            f"RSS start={format_mib(self.rss_start)}, "
            f"peak={format_mib(self.rss_peak)}, "
            f"end={format_mib(self.rss_end)}, "
            f"max call +RSS={format_mib(self.max_call_rss_increase)}"
        )


class _StageMeasurement:
    def __init__(self, profile: StageProfile):
        self.profile = profile
        self.sampler = _RSSSampler(profile.sample_interval)
        self.started = None

    def __enter__(self):
        self.started = time.perf_counter()
        self.sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.started
        self.sampler.stop()
        self.profile._record(elapsed, self.sampler)
        return False
