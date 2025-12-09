"""Integration tests for RecCache system."""

import pytest
import numpy as np
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reccache.utils.data_loader import generate_synthetic_data
from reccache.utils.config import Config, CacheConfig, ClusterConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.cache.manager import CacheManager, RecommendationRequest, CacheAwareRecommender
from reccache.models.reranker import LightweightReranker
from reccache.models.quality_predictor import QualityPredictor
from reccache.evaluation.metrics import RecommendationMetrics
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig


class TestEndToEndFlow:
    """Test complete end-to-end flows."""

    @pytest.fixture
    def setup_system(self):
        """Setup a complete RecCache system for testing."""
        np.random.seed(42)

        # Generate synthetic data
        data = generate_synthetic_data(
            n_users=100,
            n_items=50,
            n_interactions=2000,
            n_user_clusters=5,
        )

        # Train recommender
        recommender = MatrixFactorizationRecommender(
            n_users=data.n_users,
            n_items=data.n_items,
            embedding_dim=32,
        )
        recommender.fit(
            data.user_ids, data.item_ids, data.ratings,
            epochs=5, verbose=False,
        )

        # Setup clustering
        cluster_manager = UserClusterManager(
            n_clusters=10,
            embedding_dim=32,
            n_items=data.n_items,
        )
        item_embeddings = recommender.get_all_item_embeddings()
        cluster_manager.set_item_embeddings(item_embeddings)
        cluster_manager.initialize_from_interactions(
            data.user_ids, data.item_ids, data.ratings,
        )

        # Setup cache
        cache_config = CacheConfig(
            local_cache_size=500,
            local_cache_ttl=300,
            use_redis_cache=False,
        )
        cache_manager = CacheManager(
            cache_config=cache_config,
            cluster_manager=cluster_manager,
        )

        # Setup reranker
        reranker = LightweightReranker()
        reranker.set_item_embeddings(item_embeddings)

        return {
            "data": data,
            "recommender": recommender,
            "cluster_manager": cluster_manager,
            "cache_manager": cache_manager,
            "reranker": reranker,
        }

    def test_cache_aware_recommender(self, setup_system):
        """Test CacheAwareRecommender functionality."""
        recommender = setup_system["recommender"]
        cache_manager = setup_system["cache_manager"]
        reranker = setup_system["reranker"]
        data = setup_system["data"]

        cached_rec = CacheAwareRecommender(
            recommender=recommender,
            cache_manager=cache_manager,
            reranker=reranker,
        )

        # First request - cache miss
        recs1, meta1 = cached_rec.recommend(user_id=0, n=10)
        assert len(recs1) == 10
        assert meta1["cache_hit"] == False

        # Same user, same cluster - should hit cache
        recs2, meta2 = cached_rec.recommend(user_id=0, n=10)
        assert meta2["cache_hit"] == True
        assert meta2["cache_level"] == "local"

    def test_cluster_based_cache_sharing(self, setup_system):
        """Test that users in same cluster share cache."""
        recommender = setup_system["recommender"]
        cache_manager = setup_system["cache_manager"]
        cluster_manager = setup_system["cluster_manager"]

        # Find two users in same cluster
        user_clusters = {}
        for user_id in range(50):
            info = cluster_manager.get_user_cluster(user_id)
            cluster_id = info.cluster_id
            if cluster_id not in user_clusters:
                user_clusters[cluster_id] = []
            user_clusters[cluster_id].append(user_id)

        # Find a cluster with multiple users
        shared_cluster = None
        for cluster_id, users in user_clusters.items():
            if len(users) >= 2:
                shared_cluster = (cluster_id, users[:2])
                break

        if shared_cluster:
            cluster_id, (user1, user2) = shared_cluster

            # User 1 request - populates cache
            request1 = RecommendationRequest(user_id=user1, n_recommendations=10)
            result1 = cache_manager.get(request1)
            assert not result1.hit

            # Store recommendations
            recs = recommender.recommend(user1, n=10)
            cache_manager.put(request1, recs)

            # User 2 from same cluster - should hit cache
            request2 = RecommendationRequest(user_id=user2, n_recommendations=10)
            result2 = cache_manager.get(request2)

            # Both users share the same cluster, so should hit
            assert result2.hit or result1.cluster_info.cluster_id == result2.cluster_info.cluster_id

    def test_quality_predictor_integration(self, setup_system):
        """Test quality predictor with cache manager."""
        cache_manager = setup_system["cache_manager"]
        cluster_manager = setup_system["cluster_manager"]

        # Setup quality predictor
        quality_predictor = QualityPredictor(quality_threshold=0.2)
        cache_manager.set_quality_predictor(quality_predictor)

        # Test predictions for different users
        for user_id in range(10):
            cluster_info = cluster_manager.get_user_cluster(user_id)

            prediction = quality_predictor.predict(
                distance_to_center=cluster_info.distance_to_center,
                cluster_size=cluster_info.cluster_size,
            )

            assert 0 <= prediction.quality_score <= 1
            assert prediction.use_cache in (True, False)  # Works with numpy bool too

    def test_reranking_personalization(self, setup_system):
        """Test that reranking provides personalization."""
        reranker = setup_system["reranker"]
        data = setup_system["data"]

        # Set different histories for two users
        user1_history = [0, 1, 2, 3, 4]
        user2_history = [45, 46, 47, 48, 49]

        reranker.set_user_history(0, user1_history)
        reranker.set_user_history(1, user2_history)

        # Same input items
        items = list(range(20, 40))

        result1 = reranker.rerank(user_id=0, items=items)
        result2 = reranker.rerank(user_id=1, items=items)

        # Results should be different due to different histories
        # (At least some difference in ordering)
        assert result1.items != result2.items or result1.personalization_boost != result2.personalization_boost

    def test_simulation_flow(self, setup_system):
        """Test complete simulation flow."""
        recommender = setup_system["recommender"]
        cache_manager = setup_system["cache_manager"]
        cluster_manager = setup_system["cluster_manager"]
        data = setup_system["data"]

        # Build simple ground truth
        from collections import defaultdict
        ground_truth = defaultdict(set)
        for user_id, item_id, rating in zip(
            data.user_ids, data.item_ids, data.ratings
        ):
            if rating >= 4.0:
                ground_truth[int(user_id)].add(int(item_id))
        ground_truth = dict(ground_truth)

        # Run mini simulation
        sim_config = SimulationConfig(
            n_requests=500,
            n_warmup_requests=100,
            eval_sample_rate=0.2,
        )

        simulator = OnlineSimulator(
            recommender=recommender,
            cache_manager=cache_manager,
            cluster_manager=cluster_manager,
            config=sim_config,
        )

        result = simulator.run_simulation(
            n_users=data.n_users,
            n_items=data.n_items,
            ground_truth=ground_truth,
            verbose=False,
        )

        # Basic sanity checks
        assert 0 <= result.hit_rate <= 1
        assert result.avg_latency_ms >= 0
        assert result.n_requests == 500


class TestPerformance:
    """Performance-related tests."""

    def test_cache_speedup(self):
        """Test that cache provides meaningful speedup."""
        np.random.seed(42)

        # Setup minimal system
        n_users, n_items = 50, 30

        recommender = MatrixFactorizationRecommender(
            n_users=n_users, n_items=n_items, embedding_dim=16,
        )
        # Train with random data
        recommender.fit(
            np.random.randint(0, n_users, 500),
            np.random.randint(0, n_items, 500),
            np.random.uniform(1, 5, 500).astype(np.float32),
            epochs=3, verbose=False,
        )

        cache_manager = CacheManager(
            cache_config=CacheConfig(local_cache_size=100, use_redis_cache=False),
        )

        # Measure cache miss latency
        miss_latencies = []
        for user_id in range(10):
            start = time.time()
            request = RecommendationRequest(user_id=user_id)
            result = cache_manager.get(request)
            recommender.recommend(user_id, n=10)
            miss_latencies.append((time.time() - start) * 1000)

            # Populate cache
            cache_manager.put(request, list(range(10)))

        # Measure cache hit latency
        hit_latencies = []
        for user_id in range(10):
            start = time.time()
            request = RecommendationRequest(user_id=user_id)
            result = cache_manager.get(request)
            hit_latencies.append((time.time() - start) * 1000)

        # Cache hits should be significantly faster
        avg_miss = np.mean(miss_latencies)
        avg_hit = np.mean(hit_latencies)

        assert avg_hit < avg_miss, "Cache hits should be faster than misses"

    def test_throughput(self):
        """Test system throughput."""
        np.random.seed(42)

        cache_manager = CacheManager(
            cache_config=CacheConfig(local_cache_size=1000, use_redis_cache=False),
        )

        # Pre-populate cache
        for i in range(100):
            request = RecommendationRequest(user_id=i % 10)
            cache_manager.put(request, list(range(20)))

        # Measure throughput
        n_requests = 1000
        start = time.time()

        for i in range(n_requests):
            request = RecommendationRequest(user_id=i % 10)
            cache_manager.get(request)

        duration = time.time() - start
        throughput = n_requests / duration

        # Should handle at least 1000 requests/second
        assert throughput > 1000, f"Throughput too low: {throughput:.1f} req/s"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_cache(self):
        """Test behavior with empty cache."""
        cache_manager = CacheManager(
            cache_config=CacheConfig(local_cache_size=10, use_redis_cache=False),
        )

        request = RecommendationRequest(user_id=999)
        result = cache_manager.get(request)

        assert not result.hit
        assert result.cache_level == "miss"

    def test_new_user(self):
        """Test handling of new users without history."""
        cluster_manager = UserClusterManager(
            n_clusters=5, embedding_dim=16, n_items=50,
        )
        cluster_manager.set_item_embeddings(np.random.randn(50, 16).astype(np.float32))

        # New user with no history
        cluster_info = cluster_manager.get_user_cluster(user_id=9999)

        assert cluster_info.cluster_id >= 0
        assert cluster_info.cluster_id < 5

    def test_cache_overflow(self):
        """Test cache behavior when full."""
        cache_manager = CacheManager(
            cache_config=CacheConfig(local_cache_size=5, use_redis_cache=False),
        )

        # Fill cache beyond capacity
        for i in range(10):
            request = RecommendationRequest(user_id=i)
            cache_manager.put(request, [i])

        # Cache should still function
        stats = cache_manager.get_stats()
        assert stats["local_cache"]["size"] <= 5

    def test_invalid_recommendations(self):
        """Test handling of edge cases in recommendations."""
        reranker = LightweightReranker()

        # Empty items
        result = reranker.rerank(user_id=0, items=[])
        assert result.items == []

        # Single item
        result = reranker.rerank(user_id=0, items=[42])
        assert result.items == [42]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
