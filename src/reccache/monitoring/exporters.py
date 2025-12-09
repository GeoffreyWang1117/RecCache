"""Metrics exporters for various formats and destinations."""

import json
import time
import logging
from typing import Dict, Optional, Callable
from pathlib import Path
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class BaseExporter:
    """Base class for metrics exporters."""

    def export(self, metrics: Dict):
        """Export metrics. Override in subclass."""
        raise NotImplementedError


class JSONExporter(BaseExporter):
    """Export metrics to JSON file."""

    def __init__(
        self,
        output_path: str = "metrics.json",
        append: bool = True,
        pretty: bool = False,
    ):
        self.output_path = Path(output_path)
        self.append = append
        self.pretty = pretty

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, metrics: Dict):
        """Export metrics to JSON file."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }

        mode = "a" if self.append else "w"
        indent = 2 if self.pretty else None

        with open(self.output_path, mode) as f:
            json.dump(record, f, indent=indent, default=str)
            f.write("\n")

        logger.debug(f"Exported metrics to {self.output_path}")


class PrometheusExporter(BaseExporter):
    """
    Export metrics in Prometheus format.

    Can either write to file or serve via HTTP endpoint.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.output_path = Path(output_path) if output_path else None
        self.port = port

        self._latest_metrics: str = ""
        self._lock = threading.Lock()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def export(self, metrics: Dict):
        """Export metrics in Prometheus format."""
        lines = self._format_prometheus(metrics)

        with self._lock:
            self._latest_metrics = lines

        if self.output_path:
            with open(self.output_path, "w") as f:
                f.write(lines)

    def _format_prometheus(self, metrics: Dict) -> str:
        """Format metrics as Prometheus text."""
        lines = []

        # Add help and type annotations
        lines.append("# HELP reccache_requests_total Total number of recommendation requests")
        lines.append("# TYPE reccache_requests_total counter")
        lines.append(f"reccache_requests_total {metrics.get('requests', {}).get('total', 0)}")

        lines.append("# HELP reccache_cache_hit_rate Current cache hit rate")
        lines.append("# TYPE reccache_cache_hit_rate gauge")
        lines.append(f"reccache_cache_hit_rate {metrics.get('cache', {}).get('hit_rate', 0):.4f}")

        lines.append("# HELP reccache_cache_hits_total Total cache hits by level")
        lines.append("# TYPE reccache_cache_hits_total counter")
        cache = metrics.get("cache", {})
        lines.append(f'reccache_cache_hits_total{{level="local"}} {cache.get("local_hits", 0)}')
        lines.append(f'reccache_cache_hits_total{{level="redis"}} {cache.get("redis_hits", 0)}')

        lines.append("# HELP reccache_cache_misses_total Total cache misses")
        lines.append("# TYPE reccache_cache_misses_total counter")
        lines.append(f"reccache_cache_misses_total {cache.get('misses', 0)}")

        lines.append("# HELP reccache_latency_ms Request latency in milliseconds")
        lines.append("# TYPE reccache_latency_ms summary")
        latency = metrics.get("latency_ms", {})
        lines.append(f'reccache_latency_ms{{quantile="0.5"}} {latency.get("p50", 0):.2f}')
        lines.append(f'reccache_latency_ms{{quantile="0.95"}} {latency.get("p95", 0):.2f}')
        lines.append(f'reccache_latency_ms{{quantile="0.99"}} {latency.get("p99", 0):.2f}')

        lines.append("# HELP reccache_cache_size Current cache size")
        lines.append("# TYPE reccache_cache_size gauge")
        lines.append(f'reccache_cache_size{{level="local"}} {cache.get("local_size", 0)}')
        lines.append(f'reccache_cache_size{{level="redis"}} {cache.get("redis_size", 0)}')

        lines.append("# HELP reccache_errors_total Total errors")
        lines.append("# TYPE reccache_errors_total counter")
        lines.append(f"reccache_errors_total {metrics.get('errors', 0)}")

        lines.append("# HELP reccache_uptime_seconds Uptime in seconds")
        lines.append("# TYPE reccache_uptime_seconds gauge")
        lines.append(f"reccache_uptime_seconds {metrics.get('uptime_seconds', 0):.0f}")

        return "\n".join(lines)

    def get_metrics(self) -> str:
        """Get latest formatted metrics."""
        with self._lock:
            return self._latest_metrics


class ConsoleExporter(BaseExporter):
    """Export metrics to console (for debugging)."""

    def __init__(self, interval: int = 10):
        self.interval = interval
        self._last_export = 0

    def export(self, metrics: Dict):
        """Print metrics to console if enough time has passed."""
        now = time.time()
        if now - self._last_export < self.interval:
            return

        self._last_export = now

        print("\n" + "=" * 50)
        print(f"RecCache Metrics - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)

        requests = metrics.get("requests", {})
        print(f"Requests: {requests.get('total', 0):,} "
              f"({requests.get('rate_per_second', 0):.1f}/s)")

        cache = metrics.get("cache", {})
        print(f"Cache Hit Rate: {cache.get('hit_rate', 0):.1%}")
        print(f"  Local: {cache.get('local_hits', 0):,}, "
              f"Redis: {cache.get('redis_hits', 0):,}, "
              f"Miss: {cache.get('misses', 0):,}")

        latency = metrics.get("latency_ms", {})
        print(f"Latency: p50={latency.get('p50', 0):.2f}ms, "
              f"p95={latency.get('p95', 0):.2f}ms, "
              f"p99={latency.get('p99', 0):.2f}ms")

        print("=" * 50 + "\n")


class DatadogExporter(BaseExporter):
    """
    Export metrics to Datadog.

    Requires datadog package: pip install datadog
    """

    def __init__(
        self,
        api_key: str,
        app_key: Optional[str] = None,
        prefix: str = "reccache",
        tags: Optional[Dict[str, str]] = None,
    ):
        self.prefix = prefix
        self.tags = tags or {}

        try:
            from datadog import initialize, statsd
            initialize(api_key=api_key, app_key=app_key)
            self._statsd = statsd
            self._available = True
        except ImportError:
            logger.warning("datadog package not installed")
            self._available = False

    def export(self, metrics: Dict):
        """Export metrics to Datadog."""
        if not self._available:
            return

        tags = [f"{k}:{v}" for k, v in self.tags.items()]

        # Requests
        requests = metrics.get("requests", {})
        self._statsd.gauge(f"{self.prefix}.requests.total", requests.get("total", 0), tags=tags)
        self._statsd.gauge(f"{self.prefix}.requests.rate", requests.get("rate_per_second", 0), tags=tags)

        # Cache
        cache = metrics.get("cache", {})
        self._statsd.gauge(f"{self.prefix}.cache.hit_rate", cache.get("hit_rate", 0), tags=tags)
        self._statsd.gauge(f"{self.prefix}.cache.local_hits", cache.get("local_hits", 0), tags=tags)
        self._statsd.gauge(f"{self.prefix}.cache.redis_hits", cache.get("redis_hits", 0), tags=tags)
        self._statsd.gauge(f"{self.prefix}.cache.misses", cache.get("misses", 0), tags=tags)

        # Latency
        latency = metrics.get("latency_ms", {})
        self._statsd.gauge(f"{self.prefix}.latency.p50", latency.get("p50", 0), tags=tags)
        self._statsd.gauge(f"{self.prefix}.latency.p95", latency.get("p95", 0), tags=tags)
        self._statsd.gauge(f"{self.prefix}.latency.p99", latency.get("p99", 0), tags=tags)


class MultiExporter(BaseExporter):
    """Combine multiple exporters."""

    def __init__(self, exporters: list):
        self.exporters = exporters

    def export(self, metrics: Dict):
        """Export to all registered exporters."""
        for exporter in self.exporters:
            try:
                exporter.export(metrics)
            except Exception as e:
                logger.error(f"Exporter {type(exporter).__name__} failed: {e}")

    def add_exporter(self, exporter: BaseExporter):
        """Add an exporter."""
        self.exporters.append(exporter)


def setup_default_exporters(
    json_path: Optional[str] = "metrics/metrics.json",
    prometheus_path: Optional[str] = "metrics/prometheus.txt",
    console: bool = False,
) -> MultiExporter:
    """Setup default exporters."""
    exporters = []

    if json_path:
        exporters.append(JSONExporter(json_path))

    if prometheus_path:
        exporters.append(PrometheusExporter(prometheus_path))

    if console:
        exporters.append(ConsoleExporter())

    return MultiExporter(exporters)
