"""Metrics collection for RecCache monitoring."""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class LatencyHistogram:
    """Histogram for latency tracking."""

    buckets: List[float] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100, 250, 500, 1000])
    counts: List[int] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float):
        """Record a latency observation."""
        self.sum_value += value
        self.count += 1

        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self.counts[i] += 1
                return
        self.counts[-1] += 1  # +Inf bucket

    def get_percentile(self, p: float) -> float:
        """Estimate percentile from histogram."""
        if self.count == 0:
            return 0.0

        target = self.count * p
        cumsum = 0

        for i, bucket in enumerate(self.buckets):
            cumsum += self.counts[i]
            if cumsum >= target:
                return bucket

        return self.buckets[-1]


@dataclass
class Counter:
    """Simple counter metric."""

    value: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def inc(self, amount: int = 1):
        self.value += amount


@dataclass
class Gauge:
    """Gauge metric for current values."""

    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def set(self, value: float):
        self.value = value

    def inc(self, amount: float = 1.0):
        self.value += amount

    def dec(self, amount: float = 1.0):
        self.value -= amount


class MetricsCollector:
    """
    Central metrics collector for RecCache.

    Tracks:
    - Cache hit/miss rates
    - Request latencies
    - Quality metrics
    - System health
    """

    def __init__(
        self,
        window_size: int = 1000,
        export_interval: int = 60,
    ):
        self.window_size = window_size
        self.export_interval = export_interval

        # Counters
        self._counters: Dict[str, Counter] = {
            "requests_total": Counter(),
            "cache_hits_local": Counter(),
            "cache_hits_redis": Counter(),
            "cache_misses": Counter(),
            "quality_skips": Counter(),
            "errors_total": Counter(),
        }

        # Gauges
        self._gauges: Dict[str, Gauge] = {
            "cache_size_local": Gauge(),
            "cache_size_redis": Gauge(),
            "cluster_count": Gauge(),
            "active_users": Gauge(),
        }

        # Histograms
        self._histograms: Dict[str, LatencyHistogram] = {
            "request_latency_ms": LatencyHistogram(),
            "cache_lookup_latency_ms": LatencyHistogram(buckets=[0.1, 0.5, 1, 2, 5, 10]),
            "recommendation_latency_ms": LatencyHistogram(),
        }

        # Rolling windows for recent stats
        self._recent_latencies = deque(maxlen=window_size)
        self._recent_hit_rates = deque(maxlen=window_size)
        self._recent_quality_scores = deque(maxlen=window_size)

        # Thread safety
        self._lock = threading.RLock()

        # Callbacks for export
        self._export_callbacks: List[Callable[[Dict], None]] = []

        # Start time
        self._start_time = time.time()

    def record_request(
        self,
        latency_ms: float,
        cache_hit: bool,
        cache_level: str = "miss",
        quality_score: Optional[float] = None,
    ):
        """
        Record a recommendation request.

        Args:
            latency_ms: Request latency in milliseconds
            cache_hit: Whether cache was hit
            cache_level: "local", "redis", or "miss"
            quality_score: Optional quality score for the request
        """
        with self._lock:
            self._counters["requests_total"].inc()

            if cache_hit:
                if cache_level == "local":
                    self._counters["cache_hits_local"].inc()
                elif cache_level == "redis":
                    self._counters["cache_hits_redis"].inc()
            else:
                self._counters["cache_misses"].inc()

            self._histograms["request_latency_ms"].observe(latency_ms)
            self._recent_latencies.append(latency_ms)
            self._recent_hit_rates.append(1.0 if cache_hit else 0.0)

            if quality_score is not None:
                self._recent_quality_scores.append(quality_score)

    def record_cache_lookup(self, latency_ms: float):
        """Record cache lookup latency."""
        with self._lock:
            self._histograms["cache_lookup_latency_ms"].observe(latency_ms)

    def record_recommendation_compute(self, latency_ms: float):
        """Record recommendation computation latency."""
        with self._lock:
            self._histograms["recommendation_latency_ms"].observe(latency_ms)

    def record_quality_skip(self):
        """Record when request skipped cache due to quality prediction."""
        with self._lock:
            self._counters["quality_skips"].inc()

    def record_error(self, error_type: str = "general"):
        """Record an error."""
        with self._lock:
            self._counters["errors_total"].inc()

    def set_cache_size(self, local_size: int, redis_size: int = 0):
        """Update cache size gauges."""
        with self._lock:
            self._gauges["cache_size_local"].set(local_size)
            self._gauges["cache_size_redis"].set(redis_size)

    def set_cluster_count(self, count: int):
        """Update cluster count gauge."""
        with self._lock:
            self._gauges["cluster_count"].set(count)

    def set_active_users(self, count: int):
        """Update active users gauge."""
        with self._lock:
            self._gauges["active_users"].set(count)

    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        with self._lock:
            total_requests = self._counters["requests_total"].value
            local_hits = self._counters["cache_hits_local"].value
            redis_hits = self._counters["cache_hits_redis"].value
            misses = self._counters["cache_misses"].value

            total_hits = local_hits + redis_hits
            hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

            latency_hist = self._histograms["request_latency_ms"]

            return {
                "uptime_seconds": time.time() - self._start_time,
                "requests": {
                    "total": total_requests,
                    "rate_per_second": total_requests / (time.time() - self._start_time) if total_requests > 0 else 0,
                },
                "cache": {
                    "hit_rate": hit_rate,
                    "local_hits": local_hits,
                    "redis_hits": redis_hits,
                    "misses": misses,
                    "quality_skips": self._counters["quality_skips"].value,
                    "local_size": int(self._gauges["cache_size_local"].value),
                    "redis_size": int(self._gauges["cache_size_redis"].value),
                },
                "latency_ms": {
                    "mean": latency_hist.sum_value / latency_hist.count if latency_hist.count > 0 else 0,
                    "p50": latency_hist.get_percentile(0.5),
                    "p95": latency_hist.get_percentile(0.95),
                    "p99": latency_hist.get_percentile(0.99),
                },
                "recent": self._get_recent_stats(),
                "errors": self._counters["errors_total"].value,
            }

    def _get_recent_stats(self) -> Dict:
        """Get statistics for recent requests."""
        if not self._recent_latencies:
            return {}

        latencies = list(self._recent_latencies)
        hit_rates = list(self._recent_hit_rates)

        return {
            "window_size": len(latencies),
            "latency_mean": float(np.mean(latencies)),
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "hit_rate": float(np.mean(hit_rates)),
            "quality_mean": float(np.mean(self._recent_quality_scores)) if self._recent_quality_scores else None,
        }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Counters
        for name, counter in self._counters.items():
            lines.append(f"# TYPE reccache_{name} counter")
            lines.append(f"reccache_{name} {counter.value}")

        # Gauges
        for name, gauge in self._gauges.items():
            lines.append(f"# TYPE reccache_{name} gauge")
            lines.append(f"reccache_{name} {gauge.value}")

        # Histograms
        for name, hist in self._histograms.items():
            lines.append(f"# TYPE reccache_{name} histogram")
            cumsum = 0
            for i, bucket in enumerate(hist.buckets):
                cumsum += hist.counts[i]
                lines.append(f'reccache_{name}_bucket{{le="{bucket}"}} {cumsum}')
            lines.append(f'reccache_{name}_bucket{{le="+Inf"}} {hist.count}')
            lines.append(f"reccache_{name}_sum {hist.sum_value}")
            lines.append(f"reccache_{name}_count {hist.count}")

        return "\n".join(lines)

    def register_export_callback(self, callback: Callable[[Dict], None]):
        """Register a callback for metric exports."""
        self._export_callbacks.append(callback)

    def export(self):
        """Export metrics to all registered callbacks."""
        summary = self.get_summary()
        for callback in self._export_callbacks:
            try:
                callback(summary)
            except Exception as e:
                print(f"Error in export callback: {e}")

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            for counter in self._counters.values():
                counter.value = 0

            for gauge in self._gauges.values():
                gauge.value = 0.0

            for hist in self._histograms.values():
                hist.counts = [0] * (len(hist.buckets) + 1)
                hist.sum_value = 0.0
                hist.count = 0

            self._recent_latencies.clear()
            self._recent_hit_rates.clear()
            self._recent_quality_scores.clear()

            self._start_time = time.time()


# Global metrics instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_metrics_collector(collector: MetricsCollector):
    """Set global metrics collector."""
    global _global_collector
    _global_collector = collector
