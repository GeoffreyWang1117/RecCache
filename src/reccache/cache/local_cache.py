"""Local in-process cache with LRU + quality-aware eviction."""

import time
import threading
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import heapq


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    quality_score: float = 1.0  # Higher = better quality
    ttl: int = 300  # seconds

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def priority_score(self) -> float:
        """
        Compute eviction priority (lower = evict first).

        Combines:
        - Recency (LRU factor)
        - Quality score
        - Access frequency
        """
        age = time.time() - self.last_accessed
        recency = 1.0 / (1.0 + age / 60.0)  # Decay over minutes
        frequency = min(1.0, self.access_count / 10.0)

        return 0.4 * recency + 0.4 * self.quality_score + 0.2 * frequency


class LocalCache:
    """
    Local in-process cache with quality-aware eviction.

    Features:
    - LRU + quality-aware eviction policy
    - TTL support
    - Thread-safe operations
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 300,
        cleanup_interval: int = 60,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Background cleanup
        self._last_cleanup = time.time()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Update access stats
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._hits += 1

            return entry.value

    def put(
        self,
        key: str,
        value: Any,
        quality_score: float = 1.0,
        ttl: Optional[int] = None,
    ):
        """
        Put value into cache.

        Args:
            key: Cache key
            value: Value to cache
            quality_score: Quality score for eviction priority
            ttl: Time to live in seconds
        """
        with self._lock:
            # Check if we need cleanup
            self._maybe_cleanup()

            # Evict if at capacity
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_one()

            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                quality_score=quality_score,
                ttl=ttl or self.default_ttl,
            )
            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    def update_quality(self, key: str, quality_score: float):
        """Update the quality score of a cached entry."""
        with self._lock:
            if key in self._cache:
                self._cache[key].quality_score = quality_score

    def _evict_one(self):
        """Evict the lowest priority entry."""
        if not self._cache:
            return

        # Find entry with lowest priority
        min_priority = float("inf")
        min_key = None

        for key, entry in self._cache.items():
            if entry.is_expired():
                min_key = key
                break
            priority = entry.priority_score
            if priority < min_priority:
                min_priority = priority
                min_key = key

        if min_key:
            del self._cache[min_key]
            self._evictions += 1

    def _evict_batch(self, n: int):
        """Evict n lowest priority entries."""
        if not self._cache or n <= 0:
            return

        # Get all priorities
        priorities = [
            (entry.priority_score, key)
            for key, entry in self._cache.items()
        ]

        # Get n smallest
        to_evict = heapq.nsmallest(n, priorities)

        for _, key in to_evict:
            if key in self._cache:
                del self._cache[key]
                self._evictions += 1

    def _maybe_cleanup(self):
        """Periodically clean up expired entries."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._last_cleanup = now

        # Remove expired entries
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
            }

    def get_keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    def get_entries_by_quality(
        self, min_quality: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Get entries filtered by minimum quality score."""
        with self._lock:
            return [
                (key, entry.quality_score)
                for key, entry in self._cache.items()
                if entry.quality_score >= min_quality and not entry.is_expired()
            ]
