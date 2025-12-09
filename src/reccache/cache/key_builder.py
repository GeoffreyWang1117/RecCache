"""Cache key builder for recommendation requests."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False


class CacheKeyBuilder:
    """
    Builds cache keys for recommendation requests.

    Key format: (user_cluster, context_hash) -> recommendations

    The key design balances:
    - Specificity: More specific keys = higher quality but lower hit rate
    - Generalization: More general keys = higher hit rate but lower quality
    """

    def __init__(
        self,
        context_precision: int = 3,
        include_time_bucket: bool = True,
        time_bucket_hours: int = 4,
    ):
        """
        Args:
            context_precision: Decimal places for context feature hashing
            include_time_bucket: Whether to include time in the key
            time_bucket_hours: Hours per time bucket
        """
        self.context_precision = context_precision
        self.include_time_bucket = include_time_bucket
        self.time_bucket_hours = time_bucket_hours

    def build_key(
        self,
        cluster_id: int,
        context_features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        extra_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a cache key from cluster and context.

        Args:
            cluster_id: User's cluster ID
            context_features: Numerical context features
            timestamp: Request timestamp
            extra_features: Additional features to include in key

        Returns:
            Cache key string
        """
        key_parts = [f"c{cluster_id}"]

        # Add context hash
        if context_features is not None:
            context_hash = self._hash_context(context_features)
            key_parts.append(f"ctx{context_hash}")

        # Add time bucket
        if self.include_time_bucket and timestamp is not None:
            time_bucket = self._get_time_bucket(timestamp)
            key_parts.append(f"t{time_bucket}")

        # Add extra features
        if extra_features:
            extra_hash = self._hash_dict(extra_features)
            key_parts.append(f"x{extra_hash}")

        return ":".join(key_parts)

    def build_user_specific_key(
        self,
        user_id: int,
        context_features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        """
        Build a user-specific cache key (no clustering).

        Use this for VIP users or when cluster cache isn't appropriate.
        """
        key_parts = [f"u{user_id}"]

        if context_features is not None:
            context_hash = self._hash_context(context_features)
            key_parts.append(f"ctx{context_hash}")

        if self.include_time_bucket and timestamp is not None:
            time_bucket = self._get_time_bucket(timestamp)
            key_parts.append(f"t{time_bucket}")

        return ":".join(key_parts)

    def _hash_context(self, features: np.ndarray) -> str:
        """Hash context features to a short string."""
        # Round to reduce sensitivity
        rounded = np.round(features, self.context_precision)

        # Create hash
        if XXHASH_AVAILABLE:
            h = xxhash.xxh64()
            h.update(rounded.tobytes())
            return h.hexdigest()[:8]
        else:
            h = hashlib.md5(rounded.tobytes())
            return h.hexdigest()[:8]

    def _hash_dict(self, d: Dict[str, Any]) -> str:
        """Hash a dictionary to a short string."""
        # Sort keys for consistency
        serialized = json.dumps(d, sort_keys=True)

        if XXHASH_AVAILABLE:
            h = xxhash.xxh64()
            h.update(serialized.encode())
            return h.hexdigest()[:8]
        else:
            h = hashlib.md5(serialized.encode())
            return h.hexdigest()[:8]

    def _get_time_bucket(self, timestamp: float) -> int:
        """Get time bucket index."""
        hours_since_epoch = timestamp / 3600
        return int(hours_since_epoch / self.time_bucket_hours)

    def parse_key(self, key: str) -> Dict[str, Any]:
        """
        Parse a cache key back to its components.

        Returns:
            Dictionary with parsed components
        """
        parts = key.split(":")
        result = {}

        for part in parts:
            if part.startswith("c") and part[1:].isdigit():
                result["cluster_id"] = int(part[1:])
            elif part.startswith("u") and part[1:].isdigit():
                result["user_id"] = int(part[1:])
            elif part.startswith("ctx"):
                result["context_hash"] = part[3:]
            elif part.startswith("t") and part[1:].isdigit():
                result["time_bucket"] = int(part[1:])
            elif part.startswith("x"):
                result["extra_hash"] = part[1:]

        return result

    def get_related_keys(
        self,
        cluster_id: int,
        context_features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        neighbor_clusters: Optional[List[int]] = None,
        time_range: int = 1,
    ) -> List[str]:
        """
        Get related cache keys for potential fallback.

        This is useful when the exact key isn't cached but similar
        requests might be.

        Args:
            cluster_id: Primary cluster ID
            context_features: Context features
            timestamp: Request timestamp
            neighbor_clusters: Nearby cluster IDs
            time_range: Number of adjacent time buckets to include

        Returns:
            List of related cache keys
        """
        keys = []

        # Primary key
        primary = self.build_key(cluster_id, context_features, timestamp)
        keys.append(primary)

        # Keys for neighboring clusters
        if neighbor_clusters:
            for nc in neighbor_clusters:
                key = self.build_key(nc, context_features, timestamp)
                if key != primary:
                    keys.append(key)

        # Keys for adjacent time buckets
        if timestamp and self.include_time_bucket and time_range > 0:
            base_bucket = self._get_time_bucket(timestamp)
            for offset in range(-time_range, time_range + 1):
                if offset == 0:
                    continue
                adjusted_ts = timestamp + offset * self.time_bucket_hours * 3600
                key = self.build_key(cluster_id, context_features, adjusted_ts)
                if key not in keys:
                    keys.append(key)

        return keys


def compute_key_similarity(key1: str, key2: str) -> float:
    """
    Compute similarity between two cache keys.

    Returns a score between 0 and 1.
    """
    builder = CacheKeyBuilder()
    parts1 = builder.parse_key(key1)
    parts2 = builder.parse_key(key2)

    score = 0.0
    weights = {"cluster_id": 0.5, "context_hash": 0.3, "time_bucket": 0.2}

    for key, weight in weights.items():
        if key in parts1 and key in parts2:
            if parts1[key] == parts2[key]:
                score += weight

    return score
