"""Redis-based distributed cache for recommendation results."""

import json
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RedisCacheEntry:
    """Serializable cache entry for Redis."""

    value: Any
    created_at: float
    quality_score: float
    access_count: int = 0


class RedisCache:
    """
    Redis-based distributed cache.

    Features:
    - Automatic serialization/deserialization
    - Quality score tracking via sorted sets
    - TTL support
    - Connection pooling
    - Graceful fallback when Redis unavailable
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
        max_connections: int = 10,
        key_prefix: str = "reccache:",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        self._client: Optional[Any] = None
        self._connected = False

        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0

        if REDIS_AVAILABLE:
            try:
                self._pool = redis.ConnectionPool(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    max_connections=max_connections,
                    decode_responses=False,
                )
                self._client = redis.Redis(connection_pool=self._pool)
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._connected = False

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"

    def _make_quality_key(self) -> str:
        """Key for quality scores sorted set."""
        return f"{self.key_prefix}quality_scores"

    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        if not self._connected or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.is_connected():
            self._misses += 1
            return None

        try:
            full_key = self._make_key(key)
            data = self._client.get(full_key)

            if data is None:
                self._misses += 1
                return None

            entry: RedisCacheEntry = pickle.loads(data)

            # Update access count
            entry.access_count += 1
            self._client.set(full_key, pickle.dumps(entry), keepttl=True)

            self._hits += 1
            return entry.value

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._errors += 1
            self._misses += 1
            return None

    def put(
        self,
        key: str,
        value: Any,
        quality_score: float = 1.0,
        ttl: Optional[int] = None,
    ):
        """
        Put value into Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            quality_score: Quality score for priority
            ttl: Time to live in seconds
        """
        if not self.is_connected():
            return

        try:
            full_key = self._make_key(key)
            entry = RedisCacheEntry(
                value=value,
                created_at=time.time(),
                quality_score=quality_score,
                access_count=1,
            )

            self._client.set(
                full_key,
                pickle.dumps(entry),
                ex=ttl or self.default_ttl,
            )

            # Track quality score in sorted set
            self._client.zadd(
                self._make_quality_key(),
                {key: quality_score},
            )

        except Exception as e:
            logger.error(f"Redis put error: {e}")
            self._errors += 1

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.is_connected():
            return False

        try:
            full_key = self._make_key(key)
            result = self._client.delete(full_key)

            # Remove from quality scores
            self._client.zrem(self._make_quality_key(), key)

            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self._errors += 1
            return False

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_connected():
            return False

        try:
            full_key = self._make_key(key)
            return self._client.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            self._errors += 1
            return False

    def update_quality(self, key: str, quality_score: float):
        """Update quality score for a key."""
        if not self.is_connected():
            return

        try:
            full_key = self._make_key(key)
            data = self._client.get(full_key)

            if data is not None:
                entry: RedisCacheEntry = pickle.loads(data)
                entry.quality_score = quality_score
                self._client.set(full_key, pickle.dumps(entry), keepttl=True)

                # Update sorted set
                self._client.zadd(self._make_quality_key(), {key: quality_score})

        except Exception as e:
            logger.error(f"Redis update_quality error: {e}")
            self._errors += 1

    def get_by_quality_range(
        self,
        min_quality: float,
        max_quality: float = float("inf"),
        limit: int = 100,
    ) -> List[Tuple[str, float]]:
        """Get keys within quality score range."""
        if not self.is_connected():
            return []

        try:
            results = self._client.zrangebyscore(
                self._make_quality_key(),
                min_quality,
                max_quality if max_quality != float("inf") else "+inf",
                start=0,
                num=limit,
                withscores=True,
            )
            return [(k.decode() if isinstance(k, bytes) else k, s) for k, s in results]
        except Exception as e:
            logger.error(f"Redis get_by_quality_range error: {e}")
            self._errors += 1
            return []

    def get_lowest_quality_keys(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get n keys with lowest quality scores."""
        if not self.is_connected():
            return []

        try:
            results = self._client.zrange(
                self._make_quality_key(),
                0,
                n - 1,
                withscores=True,
            )
            return [(k.decode() if isinstance(k, bytes) else k, s) for k, s in results]
        except Exception as e:
            logger.error(f"Redis get_lowest_quality_keys error: {e}")
            self._errors += 1
            return []

    def evict_lowest_quality(self, n: int = 10):
        """Evict n entries with lowest quality scores."""
        if not self.is_connected():
            return

        try:
            # Get lowest quality keys
            to_evict = self.get_lowest_quality_keys(n)

            for key, _ in to_evict:
                self.delete(key)

        except Exception as e:
            logger.error(f"Redis evict_lowest_quality error: {e}")
            self._errors += 1

    def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values at once."""
        if not self.is_connected():
            return {}

        try:
            full_keys = [self._make_key(k) for k in keys]
            results = self._client.mget(full_keys)

            output = {}
            for key, data in zip(keys, results):
                if data is not None:
                    entry: RedisCacheEntry = pickle.loads(data)
                    output[key] = entry.value
                    self._hits += 1
                else:
                    self._misses += 1

            return output

        except Exception as e:
            logger.error(f"Redis get_multi error: {e}")
            self._errors += 1
            return {}

    def put_multi(
        self,
        items: Dict[str, Any],
        quality_scores: Optional[Dict[str, float]] = None,
        ttl: Optional[int] = None,
    ):
        """Put multiple values at once."""
        if not self.is_connected():
            return

        try:
            pipe = self._client.pipeline()
            quality_key = self._make_quality_key()

            for key, value in items.items():
                full_key = self._make_key(key)
                quality = quality_scores.get(key, 1.0) if quality_scores else 1.0

                entry = RedisCacheEntry(
                    value=value,
                    created_at=time.time(),
                    quality_score=quality,
                    access_count=1,
                )

                pipe.set(full_key, pickle.dumps(entry), ex=ttl or self.default_ttl)
                pipe.zadd(quality_key, {key: quality})

            pipe.execute()

        except Exception as e:
            logger.error(f"Redis put_multi error: {e}")
            self._errors += 1

    def clear(self):
        """Clear all cache entries with this prefix."""
        if not self.is_connected():
            return

        try:
            # Delete all keys with prefix
            pattern = f"{self.key_prefix}*"
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break

            # Clear quality scores
            self._client.delete(self._make_quality_key())

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            self._errors += 1

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        stats = {
            "connected": self.is_connected(),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "errors": self._errors,
        }

        if self.is_connected():
            try:
                info = self._client.info("memory")
                stats["used_memory"] = info.get("used_memory_human", "N/A")
                stats["n_keys"] = self._client.dbsize()
            except Exception:
                pass

        return stats

    def close(self):
        """Close Redis connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._connected = False
