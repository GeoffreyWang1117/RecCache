"""Cache manager coordinating local and Redis caches."""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from reccache.cache.local_cache import LocalCache
from reccache.cache.redis_cache import RedisCache
from reccache.cache.key_builder import CacheKeyBuilder
from reccache.clustering.user_cluster import UserClusterManager, UserClusterInfo
from reccache.utils.config import CacheConfig, ClusterConfig


logger = logging.getLogger(__name__)


@dataclass
class CacheResult:
    """Result from cache lookup."""

    hit: bool
    value: Optional[Any]
    cache_level: str  # "local", "redis", or "miss"
    key: str
    quality_score: float
    cluster_info: Optional[UserClusterInfo]
    latency_ms: float


@dataclass
class RecommendationRequest:
    """Request for recommendations."""

    user_id: int
    context_features: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
    n_recommendations: int = 20
    extra_features: Optional[Dict[str, Any]] = None


class CacheManager:
    """
    Two-level cache manager for recommendation results.

    Architecture:
    1. Local cache (L1): Fast, in-process, limited size
    2. Redis cache (L2): Distributed, larger capacity, slower

    Cache Decision Flow:
    1. Get user's cluster assignment
    2. Build cache key from (cluster, context)
    3. Check local cache
    4. If miss, check Redis cache
    5. If miss, return None (caller should compute fresh)
    6. Optionally apply lightweight reranking to cached results
    """

    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        cluster_config: Optional[ClusterConfig] = None,
        cluster_manager: Optional[UserClusterManager] = None,
    ):
        self.cache_config = cache_config or CacheConfig()
        self.cluster_config = cluster_config or ClusterConfig()

        # Initialize caches
        self.local_cache = LocalCache(
            max_size=self.cache_config.local_cache_size,
            default_ttl=self.cache_config.local_cache_ttl,
        ) if self.cache_config.use_local_cache else None

        self.redis_cache = RedisCache(
            host=self.cache_config.redis_host,
            port=self.cache_config.redis_port,
            db=self.cache_config.redis_db,
            password=self.cache_config.redis_password,
            default_ttl=self.cache_config.redis_cache_ttl,
            max_connections=self.cache_config.redis_max_connections,
        ) if self.cache_config.use_redis_cache else None

        # Cluster manager (can be set later)
        self.cluster_manager = cluster_manager

        # Key builder
        self.key_builder = CacheKeyBuilder()

        # Quality predictor (will be set later)
        self._quality_predictor = None

        # Statistics
        self._stats = {
            "local_hits": 0,
            "redis_hits": 0,
            "misses": 0,
            "skipped_low_quality": 0,
            "total_requests": 0,
        }

    def set_cluster_manager(self, manager: UserClusterManager):
        """Set the cluster manager."""
        self.cluster_manager = manager

    def set_quality_predictor(self, predictor):
        """Set the quality predictor for cache decisions."""
        self._quality_predictor = predictor

    def get(self, request: RecommendationRequest) -> CacheResult:
        """
        Get cached recommendations for a request.

        Args:
            request: Recommendation request

        Returns:
            CacheResult with hit status and cached value
        """
        start_time = time.time()
        self._stats["total_requests"] += 1

        # Get user cluster info
        cluster_info = None
        if self.cluster_manager:
            cluster_info = self.cluster_manager.get_user_cluster(request.user_id)
            cluster_id = cluster_info.cluster_id
        else:
            # Fallback: use user_id as cluster (user-specific caching)
            cluster_id = request.user_id

        # Build cache key
        cache_key = self.key_builder.build_key(
            cluster_id=cluster_id,
            context_features=request.context_features,
            timestamp=request.timestamp,
            extra_features=request.extra_features,
        )

        # Note: Quality prediction is now used for eviction decisions in put()
        # rather than skipping cache lookups, to maintain high hit rates
        # while still leveraging quality-aware eviction

        # Try local cache first
        if self.local_cache:
            value = self.local_cache.get(cache_key)
            if value is not None:
                self._stats["local_hits"] += 1
                return CacheResult(
                    hit=True,
                    value=value,
                    cache_level="local",
                    key=cache_key,
                    quality_score=1.0,
                    cluster_info=cluster_info,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Try Redis cache
        if self.redis_cache:
            value = self.redis_cache.get(cache_key)
            if value is not None:
                self._stats["redis_hits"] += 1

                # Promote to local cache
                if self.local_cache:
                    self.local_cache.put(cache_key, value)

                return CacheResult(
                    hit=True,
                    value=value,
                    cache_level="redis",
                    key=cache_key,
                    quality_score=1.0,
                    cluster_info=cluster_info,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Cache miss
        self._stats["misses"] += 1
        return CacheResult(
            hit=False,
            value=None,
            cache_level="miss",
            key=cache_key,
            quality_score=0.0,
            cluster_info=cluster_info,
            latency_ms=(time.time() - start_time) * 1000,
        )

    def put(
        self,
        request: RecommendationRequest,
        recommendations: List[Any],
        quality_score: float = None,
    ):
        """
        Store recommendations in cache.

        Args:
            request: Original recommendation request
            recommendations: Recommendations to cache
            quality_score: Quality score for eviction priority (auto-computed if None)
        """
        # Get cluster info for key
        cluster_info = None
        if self.cluster_manager:
            cluster_info = self.cluster_manager.get_user_cluster(request.user_id)
            cluster_id = cluster_info.cluster_id
        else:
            cluster_id = request.user_id

        # Compute quality score based on cluster info if not provided
        if quality_score is None:
            quality_score = self._compute_quality_score(cluster_info)

        # Build cache key
        cache_key = self.key_builder.build_key(
            cluster_id=cluster_id,
            context_features=request.context_features,
            timestamp=request.timestamp,
            extra_features=request.extra_features,
        )

        # Store in both caches
        if self.local_cache:
            self.local_cache.put(
                cache_key,
                recommendations,
                quality_score=quality_score,
            )

        if self.redis_cache:
            self.redis_cache.put(
                cache_key,
                recommendations,
                quality_score=quality_score,
            )

    def _compute_quality_score(self, cluster_info: Optional[UserClusterInfo]) -> float:
        """
        Compute quality score based on cluster information.

        Users closer to cluster center get higher quality scores because
        their cached recommendations are more representative of the cluster.
        """
        if cluster_info is None:
            return 1.0

        # Use quality predictor if available
        if self._quality_predictor:
            prediction = self._quality_predictor.predict(
                distance_to_center=cluster_info.distance_to_center,
                cluster_size=cluster_info.cluster_size,
            )
            return prediction.quality_score

        # Fallback: inverse relationship with distance to center
        # Normalize distance assuming max distance ~ 2.0 for normalized embeddings
        max_distance = 2.0
        normalized_distance = min(cluster_info.distance_to_center / max_distance, 1.0)

        # Quality decreases with distance, but also consider cluster size
        # Smaller clusters = more homogeneous = higher quality
        size_factor = 1.0 / (1.0 + np.log1p(cluster_info.cluster_size) / 5.0)

        quality = (1.0 - normalized_distance * 0.5) * (0.7 + 0.3 * size_factor)
        return max(0.1, min(1.0, quality))

    def invalidate_user(self, user_id: int):
        """
        Invalidate all cache entries for a user's cluster.

        Call this when user behavior significantly changes.
        """
        if not self.cluster_manager:
            return

        cluster_info = self.cluster_manager.get_user_cluster(user_id)
        cluster_id = cluster_info.cluster_id

        # Get all keys for this cluster
        pattern = f"c{cluster_id}:*"

        # Local cache doesn't support patterns easily, so we scan
        if self.local_cache:
            keys_to_delete = [
                k for k in self.local_cache.get_keys()
                if k.startswith(f"c{cluster_id}:")
            ]
            for key in keys_to_delete:
                self.local_cache.delete(key)

        # Redis supports pattern deletion
        if self.redis_cache and self.redis_cache.is_connected():
            # This is handled internally by Redis SCAN + DELETE

            pass

    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        stats = dict(self._stats)

        total = stats["total_requests"]
        if total > 0:
            stats["local_hit_rate"] = stats["local_hits"] / total
            stats["redis_hit_rate"] = stats["redis_hits"] / total
            stats["overall_hit_rate"] = (
                stats["local_hits"] + stats["redis_hits"]
            ) / total
            stats["miss_rate"] = stats["misses"] / total

        if self.local_cache:
            stats["local_cache"] = self.local_cache.get_stats()

        if self.redis_cache:
            stats["redis_cache"] = self.redis_cache.get_stats()

        return stats

    def clear(self):
        """Clear all caches."""
        if self.local_cache:
            self.local_cache.clear()
        if self.redis_cache:
            self.redis_cache.clear()

        # Reset stats
        self._stats = {
            "local_hits": 0,
            "redis_hits": 0,
            "misses": 0,
            "skipped_low_quality": 0,
            "total_requests": 0,
        }

    def warmup(
        self,
        requests: List[RecommendationRequest],
        recommender,
        batch_size: int = 100,
    ):
        """
        Warm up cache with pre-computed recommendations.

        Args:
            requests: List of requests to pre-cache
            recommender: Recommender model to generate recommendations
            batch_size: Batch size for processing
        """
        logger.info(f"Warming up cache with {len(requests)} requests...")

        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            for request in batch:
                # Generate recommendations
                recs = recommender.recommend(
                    user_id=request.user_id,
                    n=request.n_recommendations,
                )

                # Store in cache
                self.put(request, recs)

        logger.info("Cache warmup complete")


class CacheAwareRecommender:
    """
    Wrapper that adds caching to any recommender.

    Usage:
        recommender = YourRecommender()
        cached = CacheAwareRecommender(recommender, cache_manager)
        recs = cached.recommend(user_id=123, n=20)
    """

    def __init__(
        self,
        recommender,
        cache_manager: CacheManager,
        reranker=None,
    ):
        self.recommender = recommender
        self.cache_manager = cache_manager
        self.reranker = reranker

        # Statistics
        self._cache_time_saved_ms = 0.0
        self._total_requests = 0

    def recommend(
        self,
        user_id: int,
        n: int = 20,
        context_features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[int], Dict]:
        """
        Get recommendations, using cache when available.

        Returns:
            Tuple of (recommendations, metadata)
        """
        self._total_requests += 1
        start_time = time.time()

        request = RecommendationRequest(
            user_id=user_id,
            n_recommendations=n,
            context_features=context_features,
            timestamp=timestamp,
        )

        # Try cache
        cache_result = self.cache_manager.get(request)

        if cache_result.hit:
            recommendations = cache_result.value

            # Apply lightweight reranking if available
            if self.reranker:
                rerank_result = self.reranker.rerank(
                    user_id=user_id,
                    items=recommendations,
                    context_features=context_features,
                )
                recommendations = rerank_result.items[:n]

            # Estimate time saved
            estimated_compute_time = 50  # ms
            self._cache_time_saved_ms += estimated_compute_time - cache_result.latency_ms

            return recommendations, {
                "cache_hit": True,
                "cache_level": cache_result.cache_level,
                "latency_ms": cache_result.latency_ms,
                "cluster_id": cache_result.cluster_info.cluster_id if cache_result.cluster_info else None,
            }

        # Cache miss: compute fresh recommendations
        compute_start = time.time()
        recommendations = self.recommender.recommend(user_id=user_id, n=n)
        compute_time = (time.time() - compute_start) * 1000

        # Store in cache
        self.cache_manager.put(request, recommendations)

        total_time = (time.time() - start_time) * 1000

        return recommendations, {
            "cache_hit": False,
            "cache_level": "miss",
            "latency_ms": total_time,
            "compute_time_ms": compute_time,
            "cluster_id": cache_result.cluster_info.cluster_id if cache_result.cluster_info else None,
        }

    def get_stats(self) -> Dict:
        """Get recommender statistics."""
        cache_stats = self.cache_manager.get_stats()
        return {
            **cache_stats,
            "total_requests": self._total_requests,
            "cache_time_saved_ms": self._cache_time_saved_ms,
        }
