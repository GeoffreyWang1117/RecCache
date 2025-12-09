#!/usr/bin/env python3
"""
RecCache Demo: MovieLens 100K

This script demonstrates the complete RecCache system:
1. Train a recommendation model on MovieLens
2. Build user clusters from embeddings
3. Run traffic simulation with caching
4. Evaluate cache hit rate and quality tradeoffs

Usage:
    python scripts/demo_movielens.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from collections import defaultdict

from reccache.utils.data_loader import DataLoader
from reccache.utils.config import Config, CacheConfig, ClusterConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.cache.manager import CacheManager, RecommendationRequest, CacheAwareRecommender
from reccache.models.reranker import LightweightReranker
from reccache.models.quality_predictor import QualityPredictor
from reccache.evaluation.metrics import RecommendationMetrics
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig


def main():
    print("=" * 60)
    print("RecCache Demo: MovieLens 100K")
    print("=" * 60)

    # Configuration
    config = Config.default()
    config.cache.use_redis_cache = False  # Disable Redis for demo
    config.cluster.n_clusters = 50
    config.model.recommender_epochs = 15

    np.random.seed(config.random_seed)

    # Step 1: Load Data
    print("\n[1/6] Loading MovieLens 100K dataset...")
    loader = DataLoader(data_dir="data")
    train_data, val_data, test_data = loader.load_movielens_100k()

    print(f"  - Training samples: {len(train_data.user_ids):,}")
    print(f"  - Validation samples: {len(val_data.user_ids):,}")
    print(f"  - Test samples: {len(test_data.user_ids):,}")
    print(f"  - Users: {train_data.n_users}")
    print(f"  - Items: {train_data.n_items}")

    # Step 2: Train Recommender
    print("\n[2/6] Training Matrix Factorization recommender...")
    recommender = MatrixFactorizationRecommender(
        n_users=train_data.n_users,
        n_items=train_data.n_items,
        embedding_dim=config.model.recommender_embedding_dim,
        device=config.device,
    )

    train_stats = recommender.fit(
        user_ids=train_data.user_ids,
        item_ids=train_data.item_ids,
        ratings=train_data.ratings,
        epochs=config.model.recommender_epochs,
        batch_size=config.model.recommender_batch_size,
        lr=config.model.recommender_lr,
        verbose=True,
    )
    print(f"  - Final training loss: {train_stats['final_loss']:.4f}")

    # Evaluate on test set
    print("\n  Evaluating model quality...")
    test_ground_truth = build_ground_truth(test_data, min_rating=4.0)
    sample_users = list(test_ground_truth.keys())[:500]

    recommendations = {}
    for user_id in sample_users:
        recs = recommender.recommend(user_id, n=20)
        recommendations[user_id] = recs

    metrics = RecommendationMetrics.evaluate_recommendations(
        recommendations, test_ground_truth, k=10
    )
    print(f"  - NDCG@10: {metrics['ndcg@k']:.4f}")
    print(f"  - Hit Rate@10: {metrics['hit_rate']:.4f}")
    print(f"  - Precision@10: {metrics['precision@k']:.4f}")

    # Step 3: Build User Clusters
    print("\n[3/6] Building user clusters...")
    cluster_manager = UserClusterManager(
        n_clusters=config.cluster.n_clusters,
        embedding_dim=config.model.recommender_embedding_dim,
        n_items=train_data.n_items,
    )

    # Set item embeddings from trained model
    item_embeddings = recommender.get_all_item_embeddings()
    cluster_manager.set_item_embeddings(item_embeddings)

    # Initialize clusters from training interactions
    cluster_manager.initialize_from_interactions(
        user_ids=train_data.user_ids,
        item_ids=train_data.item_ids,
        ratings=train_data.ratings,
    )

    stats = cluster_manager.get_statistics()
    print(f"  - Number of clusters: {stats['n_clusters']}")
    print(f"  - Users clustered: {stats['n_users']}")
    cluster_sizes = np.array(stats['cluster_sizes'])
    print(f"  - Cluster size stats: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")

    # Step 4: Setup Cache System
    print("\n[4/6] Setting up cache system...")
    cache_manager = CacheManager(
        cache_config=config.cache,
        cluster_config=config.cluster,
        cluster_manager=cluster_manager,
    )

    # Setup reranker
    reranker = LightweightReranker()
    reranker.set_item_embeddings(item_embeddings)

    # Setup quality predictor
    quality_predictor = QualityPredictor(
        quality_threshold=config.cache.quality_threshold,
    )
    cache_manager.set_quality_predictor(quality_predictor)

    print(f"  - Local cache size: {config.cache.local_cache_size}")
    print(f"  - Cache TTL: {config.cache.local_cache_ttl}s")
    print(f"  - Quality threshold: {config.cache.quality_threshold}")

    # Step 5: Run Simulation
    print("\n[5/6] Running traffic simulation...")
    sim_config = SimulationConfig(
        n_requests=5000,
        n_warmup_requests=1000,
        user_distribution="zipf",
        zipf_alpha=1.2,
        eval_sample_rate=0.2,
        k=10,
    )

    simulator = OnlineSimulator(
        recommender=recommender,
        cache_manager=cache_manager,
        cluster_manager=cluster_manager,
        reranker=reranker,
        config=sim_config,
    )

    result = simulator.run_simulation(
        n_users=train_data.n_users,
        n_items=train_data.n_items,
        ground_truth=test_ground_truth,
        verbose=True,
    )

    # Step 6: Results
    print("\n[6/6] Simulation Results")
    print("=" * 60)

    print("\n📊 Cache Performance:")
    print(f"  - Overall Hit Rate: {result.hit_rate:.1%}")
    print(f"  - Local Cache Hits: {result.local_hit_rate:.1%}")
    print(f"  - Redis Cache Hits: {result.redis_hit_rate:.1%}")
    print(f"  - Cache Misses: {result.miss_rate:.1%}")

    print("\n⚡ Latency Metrics:")
    print(f"  - Average Latency: {result.avg_latency_ms:.2f}ms")
    print(f"  - P50 Latency: {result.p50_latency_ms:.2f}ms")
    print(f"  - P95 Latency: {result.p95_latency_ms:.2f}ms")
    print(f"  - P99 Latency: {result.p99_latency_ms:.2f}ms")

    print("\n📈 Quality Metrics:")
    print(f"  - Average NDCG@10: {result.avg_ndcg:.4f}")
    print(f"  - Quality Degradation: {result.ndcg_degradation:.2%}")
    print(f"  - List Similarity: {result.avg_list_similarity:.1%}")

    print("\n💰 Cost Savings:")
    print(f"  - Compute Saved: {result.compute_saved_pct:.1f}%")
    print(f"  - Est. Cost Reduction: {result.estimated_cost_reduction:.1%}")

    print("\n📊 Throughput:")
    print(f"  - Total Requests: {result.n_requests:,}")
    print(f"  - Duration: {result.duration_seconds:.1f}s")
    print(f"  - Requests/sec: {result.requests_per_second:.1f}")

    # Additional analysis
    print("\n" + "=" * 60)
    print("Analysis: Cache Quality vs. Hit Rate Tradeoff")
    print("=" * 60)

    analyze_quality_tradeoff(
        recommender, cache_manager, cluster_manager,
        test_ground_truth, sample_users[:100]
    )

    print("\n✅ Demo complete!")


def build_ground_truth(data, min_rating: float = 4.0) -> dict:
    """Build ground truth from test data."""
    ground_truth = defaultdict(set)

    for user_id, item_id, rating in zip(
        data.user_ids, data.item_ids, data.ratings
    ):
        if rating >= min_rating:
            ground_truth[int(user_id)].add(int(item_id))

    return dict(ground_truth)


def analyze_quality_tradeoff(
    recommender, cache_manager, cluster_manager,
    ground_truth, sample_users
):
    """Analyze the quality vs. hit rate tradeoff."""

    # Group users by distance to cluster center
    distance_buckets = {"close": [], "medium": [], "far": []}

    for user_id in sample_users:
        if user_id not in ground_truth:
            continue

        cluster_info = cluster_manager.get_user_cluster(user_id)
        dist = cluster_info.distance_to_center

        if dist < 0.5:
            distance_buckets["close"].append((user_id, dist))
        elif dist < 1.0:
            distance_buckets["medium"].append((user_id, dist))
        else:
            distance_buckets["far"].append((user_id, dist))

    print("\nQuality by Distance to Cluster Center:")
    print("-" * 50)

    for bucket_name, users in distance_buckets.items():
        if not users:
            continue

        ndcg_scores = []
        for user_id, _ in users:
            recs = recommender.recommend(user_id, n=10)
            relevant = ground_truth.get(user_id, set())
            if relevant:
                ndcg = RecommendationMetrics.ndcg_at_k(recs, relevant, 10)
                ndcg_scores.append(ndcg)

        if ndcg_scores:
            avg_dist = np.mean([d for _, d in users])
            avg_ndcg = np.mean(ndcg_scores)
            print(f"  {bucket_name.capitalize():8s}: {len(users):3d} users, "
                  f"avg_dist={avg_dist:.3f}, avg_NDCG={avg_ndcg:.4f}")

    print("\n💡 Insight: Users closer to cluster centers tend to")
    print("   have better cache quality (lower quality degradation)")
    print("   because they share more similar preferences.")


if __name__ == "__main__":
    main()
