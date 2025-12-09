"""Async Redis cache for high-concurrency scenarios."""

import asyncio
import pickle
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import redis.asyncio as aioredis
    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AsyncRedisCacheEntry:
    """Serializable cache entry for async Redis."""

    value: Any
    created_at: float
    quality_score: float
    access_count: int = 0


class AsyncRedisCache:
    """
    Async Redis cache for high-throughput scenarios.

    Uses redis.asyncio for non-blocking operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
        max_connections: int = 50,
        key_prefix: str = "reccache:",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self.key_prefix = key_prefix

        self._pool: Optional[Any] = None
        self._connected = False

        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0

    async def connect(self):
        """Establish async connection to Redis."""
        if not ASYNC_REDIS_AVAILABLE:
            logger.warning("redis.asyncio not available")
            return

        try:
            self._pool = aioredis.ConnectionPool.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=False,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            await self._client.ping()
            self._connected = True
            logger.info(f"Async Redis connected to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect async Redis: {e}")
            self._connected = False

    async def close(self):
        """Close async connection."""
        if self._pool:
            await self._pool.disconnect()
            self._connected = False

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Async get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self._connected:
            self._misses += 1
            return None

        try:
            full_key = self._make_key(key)
            data = await self._client.get(full_key)

            if data is None:
                self._misses += 1
                return None

            entry: AsyncRedisCacheEntry = pickle.loads(data)
            self._hits += 1
            return entry.value

        except Exception as e:
            logger.error(f"Async Redis get error: {e}")
            self._errors += 1
            self._misses += 1
            return None

    async def put(
        self,
        key: str,
        value: Any,
        quality_score: float = 1.0,
        ttl: Optional[int] = None,
    ):
        """
        Async put value into cache.

        Args:
            key: Cache key
            value: Value to cache
            quality_score: Quality score for priority
            ttl: Time to live in seconds
        """
        if not self._connected:
            return

        try:
            full_key = self._make_key(key)
            entry = AsyncRedisCacheEntry(
                value=value,
                created_at=time.time(),
                quality_score=quality_score,
                access_count=1,
            )

            await self._client.set(
                full_key,
                pickle.dumps(entry),
                ex=ttl or self.default_ttl,
            )

        except Exception as e:
            logger.error(f"Async Redis put error: {e}")
            self._errors += 1

    async def delete(self, key: str) -> bool:
        """Async delete a key from cache."""
        if not self._connected:
            return False

        try:
            full_key = self._make_key(key)
            result = await self._client.delete(full_key)
            return result > 0
        except Exception as e:
            logger.error(f"Async Redis delete error: {e}")
            self._errors += 1
            return False

    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Async get multiple values at once."""
        if not self._connected:
            return {}

        try:
            full_keys = [self._make_key(k) for k in keys]
            results = await self._client.mget(full_keys)

            output = {}
            for key, data in zip(keys, results):
                if data is not None:
                    entry: AsyncRedisCacheEntry = pickle.loads(data)
                    output[key] = entry.value
                    self._hits += 1
                else:
                    self._misses += 1

            return output

        except Exception as e:
            logger.error(f"Async Redis get_multi error: {e}")
            self._errors += 1
            return {}

    async def put_multi(
        self,
        items: Dict[str, Any],
        quality_scores: Optional[Dict[str, float]] = None,
        ttl: Optional[int] = None,
    ):
        """Async put multiple values at once."""
        if not self._connected:
            return

        try:
            pipe = self._client.pipeline()

            for key, value in items.items():
                full_key = self._make_key(key)
                quality = quality_scores.get(key, 1.0) if quality_scores else 1.0

                entry = AsyncRedisCacheEntry(
                    value=value,
                    created_at=time.time(),
                    quality_score=quality,
                    access_count=1,
                )

                pipe.set(full_key, pickle.dumps(entry), ex=ttl or self.default_ttl)

            await pipe.execute()

        except Exception as e:
            logger.error(f"Async Redis put_multi error: {e}")
            self._errors += 1

    async def clear(self):
        """Clear all cache entries with this prefix."""
        if not self._connected:
            return

        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._client.delete(*keys)
                if cursor == 0:
                    break

        except Exception as e:
            logger.error(f"Async Redis clear error: {e}")
            self._errors += 1

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "connected": self._connected,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "errors": self._errors,
        }


class AsyncCacheManager:
    """
    Async cache manager for high-concurrency scenarios.

    Wraps AsyncRedisCache with additional logic.
    """

    def __init__(
        self,
        redis_cache: AsyncRedisCache,
        local_cache: Optional[Any] = None,
    ):
        self.redis_cache = redis_cache
        self.local_cache = local_cache

    async def get(self, key: str) -> Optional[Any]:
        """
        Get from cache with fallback.

        Checks local cache first, then Redis.
        """
        # Try local cache first
        if self.local_cache:
            value = self.local_cache.get(key)
            if value is not None:
                return value

        # Try Redis
        value = await self.redis_cache.get(key)

        # Promote to local cache
        if value is not None and self.local_cache:
            self.local_cache.put(key, value)

        return value

    async def put(self, key: str, value: Any, **kwargs):
        """Put to both local and Redis cache."""
        if self.local_cache:
            self.local_cache.put(key, value, **kwargs)

        await self.redis_cache.put(key, value, **kwargs)

    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get for efficiency."""
        results = {}

        # Check local cache
        if self.local_cache:
            for key in keys:
                value = self.local_cache.get(key)
                if value is not None:
                    results[key] = value

        # Get remaining from Redis
        missing_keys = [k for k in keys if k not in results]
        if missing_keys:
            redis_results = await self.redis_cache.get_multi(missing_keys)
            results.update(redis_results)

            # Promote to local
            if self.local_cache:
                for key, value in redis_results.items():
                    self.local_cache.put(key, value)

        return results


async def example_usage():
    """Example of async cache usage."""
    cache = AsyncRedisCache(host="localhost", port=6379)
    await cache.connect()

    # Put some data
    await cache.put("user:1:recs", [1, 2, 3, 4, 5])

    # Get data
    recs = await cache.get("user:1:recs")
    print(f"Recommendations: {recs}")

    # Batch operations
    await cache.put_multi({
        "user:2:recs": [6, 7, 8],
        "user:3:recs": [9, 10, 11],
    })

    results = await cache.get_multi(["user:2:recs", "user:3:recs"])
    print(f"Batch results: {results}")

    await cache.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
