"""Cache module for RecCache."""

from reccache.cache.local_cache import LocalCache
from reccache.cache.redis_cache import RedisCache
from reccache.cache.manager import CacheManager, CacheAwareRecommender
from reccache.cache.key_builder import CacheKeyBuilder
from reccache.cache.warming import CacheWarmer, IncrementalWarmer
from reccache.cache.retrieval_pool import EmbeddingPool, PoolManager, RetrievalResult
from reccache.cache.baselines import (
    BaseCacheStrategy,
    LRUCache,
    LFUCache,
    RandomCache,
    PopularityCache,
    FIFOCache,
    ARCCache,
    LeCaRCache,
    OracleCache,
    NoCacheBaseline,
    create_cache,
    CacheStrategyComparator,
)

__all__ = [
    "LocalCache",
    "RedisCache",
    "CacheManager",
    "CacheAwareRecommender",
    "CacheKeyBuilder",
    "CacheWarmer",
    "IncrementalWarmer",
    # Retrieval pool
    "EmbeddingPool",
    "PoolManager",
    "RetrievalResult",
    # Baselines
    "BaseCacheStrategy",
    "LRUCache",
    "LFUCache",
    "RandomCache",
    "PopularityCache",
    "FIFOCache",
    "ARCCache",
    "LeCaRCache",
    "OracleCache",
    "NoCacheBaseline",
    "create_cache",
    "CacheStrategyComparator",
]
