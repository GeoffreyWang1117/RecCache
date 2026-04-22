"""
Oracle cache implementations for theoretical upper bound analysis.

Belady's Algorithm (OPT): Evicts the item that will be used furthest in the future.
This provides the theoretical optimal hit rate for any cache replacement policy.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import heapq


class BeladyCache:
    """
    Belady's MIN/OPT algorithm implementation.

    This algorithm requires knowledge of future requests to make optimal eviction
    decisions. It serves as a theoretical upper bound for cache hit rates.

    Note: This is an offline algorithm that requires the full request sequence
    in advance. It cannot be used in production but provides a benchmark.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: Dict[Any, Any] = {}
        self.hits = 0
        self.misses = 0

    def simulate(self, request_sequence: List[Any]) -> Dict[str, float]:
        """
        Simulate Belady's algorithm on a sequence of requests.

        Args:
            request_sequence: List of cache keys in order of access

        Returns:
            Dictionary with hit_rate and other statistics
        """
        # Build future access map: for each position, when is next access?
        next_access = self._build_next_access_map(request_sequence)

        self.cache = {}
        self.hits = 0
        self.misses = 0

        for i, key in enumerate(request_sequence):
            if key in self.cache:
                self.hits += 1
            else:
                self.misses += 1

                # Need to add to cache
                if len(self.cache) >= self.max_size:
                    # Evict the item used furthest in the future
                    evict_key = self._find_eviction_victim(i, next_access)
                    if evict_key is not None:
                        del self.cache[evict_key]

                # Add new item (with its next access time for eviction decisions)
                self.cache[key] = next_access[i].get(key, float('inf'))

        total = self.hits + self.misses
        return {
            "hit_rate": self.hits / total if total > 0 else 0,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
        }

    def _build_next_access_map(self, sequence: List[Any]) -> List[Dict[Any, int]]:
        """
        Build a map of next access times for each position in the sequence.

        next_access[i][key] = position of next access to key after position i
        """
        n = len(sequence)

        # Last occurrence of each key
        last_seen: Dict[Any, int] = {}

        # For each position, store the next access time for keys currently in consideration
        next_access = [{} for _ in range(n)]

        # Scan from end to beginning
        for i in range(n - 1, -1, -1):
            key = sequence[i]

            # Copy forward the next access info
            if i < n - 1:
                for k, v in next_access[i + 1].items():
                    next_access[i][k] = v

            # Update next access for current key
            if key in last_seen:
                next_access[i][key] = last_seen[key]
            else:
                next_access[i][key] = float('inf')  # Never accessed again

            last_seen[key] = i

        return next_access

    def _find_eviction_victim(self, current_pos: int, next_access: List[Dict]) -> Optional[Any]:
        """Find the cache item that will be used furthest in the future."""
        if not self.cache:
            return None

        # Find item with maximum next access time
        max_next = -1
        victim = None

        for key in self.cache:
            next_time = next_access[current_pos].get(key, float('inf'))
            if next_time > max_next:
                max_next = next_time
                victim = key

        return victim


class ClusterAwareBeladyCache:
    """
    Belady's algorithm with cluster-aware cache keys.

    This simulates the theoretical optimal for cluster-based caching,
    where users in the same cluster share cache entries.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.belady = BeladyCache(max_size)

    def simulate(
        self,
        user_sequence: List[int],
        user_to_cluster: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Simulate optimal caching with cluster-based keys.

        Args:
            user_sequence: List of user IDs in order of access
            user_to_cluster: Mapping from user ID to cluster ID

        Returns:
            Dictionary with hit_rate and other statistics
        """
        # Convert user sequence to cluster sequence
        cluster_sequence = [user_to_cluster.get(u, u) for u in user_sequence]

        return self.belady.simulate(cluster_sequence)


def compute_oracle_bounds(
    user_sequence: List[int],
    user_to_cluster: Dict[int, int],
    cache_sizes: List[int],
) -> Dict[str, Dict[int, float]]:
    """
    Compute oracle upper bounds for different cache sizes.

    Returns both user-level and cluster-level optimal hit rates.
    """
    results = {
        "user_level_optimal": {},
        "cluster_level_optimal": {},
    }

    cluster_sequence = [user_to_cluster.get(u, u) for u in user_sequence]

    for size in cache_sizes:
        # User-level optimal (no clustering)
        user_belady = BeladyCache(size)
        user_result = user_belady.simulate(user_sequence)
        results["user_level_optimal"][size] = user_result["hit_rate"]

        # Cluster-level optimal (with clustering)
        cluster_belady = BeladyCache(size)
        cluster_result = cluster_belady.simulate(cluster_sequence)
        results["cluster_level_optimal"][size] = cluster_result["hit_rate"]

    return results
