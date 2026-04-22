"""Baseline caching strategies for comparison."""

import time
import random
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import heapq

import numpy as np


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses


class BaseCacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._lock = threading.RLock()
        self.stats = CacheStats()

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def put(self, key: str, value: Any, **kwargs):
        """Put value into cache."""
        pass

    @abstractmethod
    def _evict(self):
        """Evict one entry according to strategy."""
        pass

    def clear(self):
        """Clear all entries."""
        pass

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = CacheStats()


class LRUCache(BaseCacheStrategy):
    """
    Least Recently Used (LRU) cache.

    Evicts the least recently accessed item.
    Standard baseline for cache replacement.
    """

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        super().__init__(max_size, ttl)
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            # Check TTL
            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                self.stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any, **kwargs):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                self._timestamps[key] = time.time()
                return

            if len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def _evict(self):
        """Evict least recently used item."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(key, None)
            self.stats.evictions += 1

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


class LFUCache(BaseCacheStrategy):
    """
    Least Frequently Used (LFU) cache.

    Evicts the least frequently accessed item.
    """

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        super().__init__(max_size, ttl)
        self._cache: Dict[str, Any] = {}
        self._frequencies: Dict[str, int] = {}
        self._timestamps: Dict[str, float] = {}
        self._freq_to_keys: Dict[int, OrderedDict] = defaultdict(OrderedDict)
        self._min_freq: int = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            # Check TTL
            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                self._remove(key)
                self.stats.misses += 1
                return None

            # Update frequency
            self._update_frequency(key)
            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any, **kwargs):
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = time.time()
                self._update_frequency(key)
                return

            if len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = value
            self._frequencies[key] = 1
            self._timestamps[key] = time.time()
            self._freq_to_keys[1][key] = None
            self._min_freq = 1

    def _update_frequency(self, key: str):
        """Update frequency count for a key."""
        freq = self._frequencies[key]
        self._frequencies[key] = freq + 1

        # Remove from old frequency list
        del self._freq_to_keys[freq][key]
        if not self._freq_to_keys[freq]:
            del self._freq_to_keys[freq]
            if self._min_freq == freq:
                self._min_freq = freq + 1

        # Add to new frequency list
        self._freq_to_keys[freq + 1][key] = None

    def _remove(self, key: str):
        """Remove a key from cache."""
        freq = self._frequencies.pop(key, 1)
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

        if key in self._freq_to_keys.get(freq, {}):
            del self._freq_to_keys[freq][key]
            if not self._freq_to_keys[freq]:
                del self._freq_to_keys[freq]

    def _evict(self):
        """Evict least frequently used item."""
        if not self._freq_to_keys:
            return

        # Find minimum frequency
        while self._min_freq not in self._freq_to_keys or not self._freq_to_keys[self._min_freq]:
            self._min_freq += 1
            if self._min_freq > 1000000:  # Safety check
                return

        # Get first key with minimum frequency (LRU among same freq)
        key, _ = self._freq_to_keys[self._min_freq].popitem(last=False)
        if not self._freq_to_keys[self._min_freq]:
            del self._freq_to_keys[self._min_freq]

        self._cache.pop(key, None)
        self._frequencies.pop(key, None)
        self._timestamps.pop(key, None)
        self.stats.evictions += 1

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._frequencies.clear()
            self._timestamps.clear()
            self._freq_to_keys.clear()
            self._min_freq = 0


class RandomCache(BaseCacheStrategy):
    """
    Random replacement cache.

    Evicts a random item when full.
    Simple baseline.
    """

    def __init__(self, max_size: int = 10000, ttl: int = 300, seed: int = 42):
        super().__init__(max_size, ttl)
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._keys: List[str] = []
        self._rng = random.Random(seed)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            # Check TTL
            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                self._remove(key)
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any, **kwargs):
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = time.time()
                return

            if len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._keys.append(key)

    def _remove(self, key: str):
        """Remove a key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            if key in self._keys:
                self._keys.remove(key)

    def _evict(self):
        """Evict a random item."""
        if self._keys:
            key = self._rng.choice(self._keys)
            self._remove(key)
            self.stats.evictions += 1

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._keys.clear()


class PopularityCache(BaseCacheStrategy):
    """
    Popularity-based cache.

    Caches items based on global popularity.
    Evicts least popular items first.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl: int = 300,
        item_popularity: Optional[Dict[int, float]] = None,
    ):
        super().__init__(max_size, ttl)
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._popularity: Dict[str, float] = {}
        self._item_popularity = item_popularity or {}

    def set_item_popularity(self, popularity: Dict[int, float]):
        """Set item popularity scores."""
        self._item_popularity = popularity

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                self._remove(key)
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any, popularity: float = 0.0, **kwargs):
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = time.time()
                return

            if len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._popularity[key] = popularity

    def _remove(self, key: str):
        """Remove a key from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._popularity.pop(key, None)

    def _evict(self):
        """Evict least popular item."""
        if not self._cache:
            return

        # Find key with lowest popularity
        min_pop = float("inf")
        min_key = None
        for key, pop in self._popularity.items():
            if pop < min_pop:
                min_pop = pop
                min_key = key

        if min_key:
            self._remove(min_key)
            self.stats.evictions += 1

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._popularity.clear()


class FIFOCache(BaseCacheStrategy):
    """
    First-In-First-Out (FIFO) cache.

    Evicts the oldest item.
    """

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        super().__init__(max_size, ttl)
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                self.stats.misses += 1
                return None

            # Don't move to end (unlike LRU)
            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any, **kwargs):
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                # Don't update timestamp for FIFO
                return

            if len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def _evict(self):
        """Evict first (oldest) item."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(key, None)
            self.stats.evictions += 1

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


class OracleCache(BaseCacheStrategy):
    """
    Oracle (optimal) cache with perfect knowledge.

    For theoretical comparison only - knows future accesses.
    In practice, simulated by providing access sequence upfront.
    """

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        super().__init__(max_size, ttl)
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._future_accesses: Dict[str, List[int]] = defaultdict(list)
        self._current_time: int = 0

    def set_future_accesses(self, access_sequence: List[str]):
        """
        Set future access sequence for optimal eviction.

        Args:
            access_sequence: List of cache keys in order of future access
        """
        self._future_accesses.clear()
        for t, key in enumerate(access_sequence):
            self._future_accesses[key].append(t)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._current_time += 1

            if key not in self._cache:
                self.stats.misses += 1
                return None

            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any, **kwargs):
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = time.time()
                return

            if len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = value
            self._timestamps[key] = time.time()

    def _evict(self):
        """Evict item that will be used furthest in the future (Belady's algorithm)."""
        if not self._cache:
            return

        max_next_use = -1
        evict_key = None

        for key in self._cache:
            # Find next future access for this key
            future = self._future_accesses.get(key, [])

            # Filter to only future accesses
            future_access = [t for t in future if t > self._current_time]

            if not future_access:
                # Item never accessed again - evict it
                evict_key = key
                break
            else:
                next_use = future_access[0]
                if next_use > max_next_use:
                    max_next_use = next_use
                    evict_key = key

        if evict_key:
            del self._cache[evict_key]
            self._timestamps.pop(evict_key, None)
            self.stats.evictions += 1

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._current_time = 0


class NoCacheBaseline(BaseCacheStrategy):
    """
    No caching baseline.

    Always returns miss - for measuring maximum latency.
    """

    def __init__(self, **kwargs):
        super().__init__(max_size=0, ttl=0)

    def get(self, key: str) -> Optional[Any]:
        self.stats.misses += 1
        return None

    def put(self, key: str, value: Any, **kwargs):
        pass  # Don't store anything

    def _evict(self):
        pass

    def clear(self):
        pass


class ARCCache(BaseCacheStrategy):
    """
    Adaptive Replacement Cache (ARC).

    Balances between recency and frequency adaptively.
    More sophisticated than LRU or LFU alone.
    """

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        super().__init__(max_size, ttl)

        # T1: recently accessed once
        self._t1: OrderedDict = OrderedDict()
        # T2: recently accessed at least twice
        self._t2: OrderedDict = OrderedDict()
        # B1: ghost entries for T1 (evicted recently)
        self._b1: OrderedDict = OrderedDict()
        # B2: ghost entries for T2 (evicted recently)
        self._b2: OrderedDict = OrderedDict()

        self._timestamps: Dict[str, float] = {}
        self._p: float = 0  # Target size for T1

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            # Check T1
            if key in self._t1:
                if time.time() - self._timestamps.get(key, 0) > self.ttl:
                    del self._t1[key]
                    del self._timestamps[key]
                    self.stats.misses += 1
                    return None

                value = self._t1.pop(key)
                self._t2[key] = value
                self.stats.hits += 1
                return value

            # Check T2
            if key in self._t2:
                if time.time() - self._timestamps.get(key, 0) > self.ttl:
                    del self._t2[key]
                    del self._timestamps[key]
                    self.stats.misses += 1
                    return None

                self._t2.move_to_end(key)
                self.stats.hits += 1
                return self._t2[key]

            self.stats.misses += 1
            return None

    def put(self, key: str, value: Any, **kwargs):
        with self._lock:
            # Case: key in T1 or T2
            if key in self._t1:
                val = self._t1.pop(key)
                self._t2[key] = value
                self._timestamps[key] = time.time()
                return

            if key in self._t2:
                self._t2[key] = value
                self._t2.move_to_end(key)
                self._timestamps[key] = time.time()
                return

            # Case: key in B1 (was recently evicted from T1)
            if key in self._b1:
                # Increase target size of T1
                self._p = min(self.max_size, self._p + max(1, len(self._b2) / max(1, len(self._b1))))
                self._replace(key)
                del self._b1[key]
                self._t2[key] = value
                self._timestamps[key] = time.time()
                return

            # Case: key in B2 (was recently evicted from T2)
            if key in self._b2:
                # Decrease target size of T1
                self._p = max(0, self._p - max(1, len(self._b1) / max(1, len(self._b2))))
                self._replace(key)
                del self._b2[key]
                self._t2[key] = value
                self._timestamps[key] = time.time()
                return

            # Case: key not in T1, T2, B1, B2
            total = len(self._t1) + len(self._t2)
            if total >= self.max_size:
                # Cache is full
                if len(self._t1) + len(self._b1) >= self.max_size:
                    # B1 is full, remove from B1
                    if self._b1:
                        self._b1.popitem(last=False)
                    self._replace(key)
                elif total < 2 * self.max_size:
                    if len(self._t1) + len(self._t2) + len(self._b1) + len(self._b2) >= 2 * self.max_size:
                        if self._b2:
                            self._b2.popitem(last=False)
                    self._replace(key)

            # Add to T1
            self._t1[key] = value
            self._timestamps[key] = time.time()

    def _replace(self, key: str):
        """Replace a page from cache."""
        if self._t1 and (len(self._t1) > self._p or (key in self._b2 and len(self._t1) == self._p)):
            # Evict from T1
            old_key, _ = self._t1.popitem(last=False)
            self._b1[old_key] = None  # Add to ghost list
            self._timestamps.pop(old_key, None)
            self.stats.evictions += 1
        elif self._t2:
            # Evict from T2
            old_key, _ = self._t2.popitem(last=False)
            self._b2[old_key] = None  # Add to ghost list
            self._timestamps.pop(old_key, None)
            self.stats.evictions += 1

    def _evict(self):
        """Evict using ARC policy."""
        self._replace("")

    def clear(self):
        with self._lock:
            self._t1.clear()
            self._t2.clear()
            self._b1.clear()
            self._b2.clear()
            self._timestamps.clear()
            self._p = 0


class LeCaRCache(BaseCacheStrategy):
    """
    LeCaR: Learning Cache Replacement.

    Adaptively combines LRU and LFU using regret-based online learning.
    Maintains ghost queues to learn which policy is more effective.

    Reference: Vietri et al., HotStorage 2018.
    """

    def __init__(self, max_size: int = 10000, ttl: int = 300, discount: float = 0.45):
        super().__init__(max_size, ttl)
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}

        # LRU ordering: front = least recent
        self._lru_order: OrderedDict = OrderedDict()
        # LFU frequency tracking
        self._frequencies: Dict[str, int] = {}

        # Ghost queues (evicted keys, limited to max_size each)
        self._lru_ghost: OrderedDict = OrderedDict()
        self._lfu_ghost: OrderedDict = OrderedDict()

        # Adaptive weight: w = probability of choosing LRU for eviction
        self._w: float = 0.5
        self._discount: float = discount

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            # Check TTL
            if time.time() - self._timestamps.get(key, 0) > self.ttl:
                self._remove(key)
                self.stats.misses += 1
                return None

            # Update LRU order
            self._lru_order.move_to_end(key)
            # Update frequency
            self._frequencies[key] = self._frequencies.get(key, 0) + 1

            self.stats.hits += 1
            return self._cache[key]

    def put(self, key: str, value: Any, **kwargs):
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = time.time()
                self._lru_order.move_to_end(key)
                self._frequencies[key] = self._frequencies.get(key, 0) + 1
                return

            # Check ghost queues and adjust weight before insertion
            if key in self._lru_ghost:
                # LRU ghost hit: LRU policy would have kept this => increase LRU weight
                self._w = min(1.0, self._w * (1 + self._discount))
                del self._lru_ghost[key]
            elif key in self._lfu_ghost:
                # LFU ghost hit: LFU policy would have kept this => decrease LRU weight
                self._w = max(0.0, self._w * (1 - self._discount))
                del self._lfu_ghost[key]

            if len(self._cache) >= self.max_size:
                self._evict()

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._lru_order[key] = None
            self._frequencies[key] = 1

    def _remove(self, key: str):
        """Remove a key from cache without adding to ghost queues."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._lru_order.pop(key, None)
        self._frequencies.pop(key, None)

    def _evict(self):
        """Evict using adaptive LRU/LFU selection based on learned weight."""
        if not self._cache:
            return

        # Use weight w to probabilistically choose between LRU and LFU eviction
        if random.random() < self._w:
            # LRU eviction: remove least recently used
            evict_key = self._get_lru_victim()
            ghost_queue = self._lru_ghost
        else:
            # LFU eviction: remove least frequently used
            evict_key = self._get_lfu_victim()
            ghost_queue = self._lfu_ghost

        if evict_key is not None:
            # Add to ghost queue
            ghost_queue[evict_key] = None
            # Limit ghost queue size
            while len(ghost_queue) > self.max_size:
                ghost_queue.popitem(last=False)

            self._remove(evict_key)
            self.stats.evictions += 1

    def _get_lru_victim(self) -> Optional[str]:
        """Get the LRU eviction victim."""
        if self._lru_order:
            # First item is least recently used
            return next(iter(self._lru_order))
        return None

    def _get_lfu_victim(self) -> Optional[str]:
        """Get the LFU eviction victim (least frequently used, LRU tie-break)."""
        if not self._frequencies:
            return None

        min_freq = min(self._frequencies.values())
        # Among items with minimum frequency, pick LRU (first in order)
        for key in self._lru_order:
            if self._frequencies.get(key, 0) == min_freq:
                return key
        return next(iter(self._lru_order)) if self._lru_order else None

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._lru_order.clear()
            self._frequencies.clear()
            self._lru_ghost.clear()
            self._lfu_ghost.clear()
            self._w = 0.5


# Factory function
def create_cache(
    strategy: str,
    max_size: int = 10000,
    ttl: int = 300,
    **kwargs
) -> BaseCacheStrategy:
    """
    Create a cache with the specified strategy.

    Args:
        strategy: Cache strategy name (lru, lfu, random, popularity, fifo, arc, oracle, none)
        max_size: Maximum cache size
        ttl: Time to live in seconds
        **kwargs: Additional arguments for specific strategies

    Returns:
        Cache instance
    """
    strategy = strategy.lower()

    strategies = {
        "lru": LRUCache,
        "lfu": LFUCache,
        "random": RandomCache,
        "popularity": PopularityCache,
        "fifo": FIFOCache,
        "arc": ARCCache,
        "lecar": LeCaRCache,
        "oracle": OracleCache,
        "none": NoCacheBaseline,
        "nocache": NoCacheBaseline,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")

    return strategies[strategy](max_size=max_size, ttl=ttl, **kwargs)


# Helper for comparing strategies
class CacheStrategyComparator:
    """Compare multiple cache strategies on the same access pattern."""

    def __init__(self, strategies: List[str], max_size: int = 10000, ttl: int = 300):
        self.caches = {
            name: create_cache(name, max_size=max_size, ttl=ttl)
            for name in strategies
        }

    def simulate_access(
        self,
        access_sequence: List[Tuple[str, Any]],
        value_generator: Optional[callable] = None,
    ) -> Dict[str, Dict]:
        """
        Simulate cache accesses across all strategies.

        Args:
            access_sequence: List of (key, value) tuples
            value_generator: Optional function to generate values for misses

        Returns:
            Dict mapping strategy name to performance metrics
        """
        # Reset all caches
        for cache in self.caches.values():
            cache.clear()
            cache.reset_stats()

        # Set oracle future accesses if present
        if "oracle" in self.caches:
            self.caches["oracle"].set_future_accesses([key for key, _ in access_sequence])

        # Simulate accesses
        for key, value in access_sequence:
            for name, cache in self.caches.items():
                result = cache.get(key)
                if result is None:
                    # Cache miss - store the value
                    cache.put(key, value)

        # Collect results
        return {name: cache.get_stats() for name, cache in self.caches.items()}
