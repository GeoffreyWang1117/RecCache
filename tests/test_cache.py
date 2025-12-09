"""Tests for cache components."""

import pytest
import numpy as np
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reccache.cache.local_cache import LocalCache
from reccache.cache.key_builder import CacheKeyBuilder, compute_key_similarity
from reccache.cache.manager import CacheManager, RecommendationRequest
from reccache.utils.config import CacheConfig


class TestLocalCache:
    """Tests for LocalCache."""

    def test_basic_operations(self):
        """Test basic get/put/delete."""
        cache = LocalCache(max_size=100, default_ttl=60)

        # Put and get
        cache.put("key1", [1, 2, 3])
        assert cache.get("key1") == [1, 2, 3]

        # Miss
        assert cache.get("nonexistent") is None

        # Delete
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LocalCache(max_size=100, default_ttl=1)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """Test LRU-style eviction when cache is full."""
        cache = LocalCache(max_size=3, default_ttl=300)

        cache.put("key1", "val1")
        cache.put("key2", "val2")
        cache.put("key3", "val3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new item, should evict key2 (least recently used)
        cache.put("key4", "val4")

        assert cache.get("key1") == "val1"  # Still there
        assert cache.get("key4") == "val4"  # New item

    def test_quality_aware_eviction(self):
        """Test quality score affects eviction."""
        cache = LocalCache(max_size=3, default_ttl=300)

        cache.put("high_quality", "val1", quality_score=0.9)
        cache.put("low_quality", "val2", quality_score=0.1)
        cache.put("medium_quality", "val3", quality_score=0.5)

        # Force eviction by adding more
        cache.put("new_item", "val4", quality_score=0.8)

        # Low quality should be evicted first
        assert cache.get("high_quality") == "val1"
        assert cache.get("low_quality") is None  # Evicted

    def test_statistics(self):
        """Test cache statistics tracking."""
        cache = LocalCache(max_size=100)

        cache.put("key1", "val1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3


class TestCacheKeyBuilder:
    """Tests for CacheKeyBuilder."""

    def test_basic_key_building(self):
        """Test basic cache key construction."""
        builder = CacheKeyBuilder()

        key = builder.build_key(cluster_id=42)
        assert "c42" in key

        key = builder.build_key(
            cluster_id=42,
            context_features=np.array([0.1, 0.2, 0.3]),
        )
        assert "c42" in key
        assert "ctx" in key

    def test_key_with_timestamp(self):
        """Test time bucket in cache key."""
        builder = CacheKeyBuilder(include_time_bucket=True, time_bucket_hours=4)

        key1 = builder.build_key(cluster_id=1, timestamp=1000000000)
        key2 = builder.build_key(cluster_id=1, timestamp=1000000000 + 3600)  # +1 hour

        # Same time bucket
        assert key1 == key2

        key3 = builder.build_key(cluster_id=1, timestamp=1000000000 + 5 * 3600)  # +5 hours
        assert key1 != key3  # Different time bucket

    def test_user_specific_key(self):
        """Test user-specific cache keys."""
        builder = CacheKeyBuilder()

        key = builder.build_user_specific_key(user_id=123)
        assert "u123" in key
        assert "c" not in key  # No cluster

    def test_key_parsing(self):
        """Test parsing cache keys."""
        builder = CacheKeyBuilder()

        key = builder.build_key(
            cluster_id=42,
            context_features=np.array([0.1, 0.2]),
            timestamp=1000000000,
        )

        parsed = builder.parse_key(key)
        assert parsed["cluster_id"] == 42
        assert "context_hash" in parsed
        assert "time_bucket" in parsed

    def test_key_similarity(self):
        """Test cache key similarity computation."""
        builder = CacheKeyBuilder()

        key1 = builder.build_key(cluster_id=1, timestamp=1000000000)
        key2 = builder.build_key(cluster_id=1, timestamp=1000000000)
        key3 = builder.build_key(cluster_id=2, timestamp=1000000000)

        # Identical keys have high similarity (cluster_id + time_bucket match)
        sim_identical = compute_key_similarity(key1, key2)
        assert sim_identical > 0.6  # cluster (0.5) + time (0.2) = 0.7

        # Different cluster has lower similarity
        sim_different = compute_key_similarity(key1, key3)
        assert sim_different < sim_identical  # Different cluster


class TestCacheManager:
    """Tests for CacheManager."""

    def test_cache_manager_basic(self):
        """Test basic cache manager operations."""
        config = CacheConfig(
            local_cache_size=100,
            use_redis_cache=False,
        )
        manager = CacheManager(cache_config=config)

        request = RecommendationRequest(
            user_id=1,
            n_recommendations=20,
        )

        # Cache miss initially
        result = manager.get(request)
        assert not result.hit

        # Put something
        manager.put(request, [1, 2, 3, 4, 5])

        # Cache hit now
        result = manager.get(request)
        assert result.hit
        assert result.value == [1, 2, 3, 4, 5]
        assert result.cache_level == "local"

    def test_cache_stats(self):
        """Test cache statistics."""
        config = CacheConfig(
            local_cache_size=100,
            use_redis_cache=False,
        )
        manager = CacheManager(cache_config=config)

        # Generate some traffic
        for i in range(10):
            request = RecommendationRequest(user_id=i % 3)
            manager.get(request)
            manager.put(request, [i])

        stats = manager.get_stats()
        assert "total_requests" in stats
        assert "local_cache" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
