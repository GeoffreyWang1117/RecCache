#!/usr/bin/env python3
"""
RecCache Benchmark Script

Comprehensive benchmark measuring:
1. Cache hit rates at different cluster granularities
2. Quality degradation under various cache configurations
3. Latency comparisons
4. Scalability analysis

Usage:
    python scripts/benchmark.py [--n-users 1000] [--n-items 500]
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from reccache.utils.data_loader import DataLoader, generate_synthetic_data
from reccache.utils.config import Config, CacheConfig, ClusterConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.cache.manager import CacheManager, CacheAwareRecommender
from reccache.models.reranker import LightweightReranker
from reccache.evaluation.metrics import RecommendationMetrics
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="RecCache Benchmark")
    parser.add_argument("--dataset", type=str, default="movielens",
                        choices=["movielens", "synthetic"])
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--n-items", type=int, default=500)
    parser.add_argument("--n-requests", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_cluster_granularity_experiment(
    recommender,
    train_data,
    test_ground_truth,
    n_clusters_list: List[int] = [10, 25, 50, 100, 200],
    n_requests: int = 5000,
) -> Dict:
    """
    Experiment: How does cluster granularity affect hit rate and quality?
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Cluster Granularity Analysis")
    print("=" * 60)

    results = []
    item_embeddings = recommender.get_all_item_embeddings()

    for n_clusters in n_clusters_list:
        print(f"\nTesting with {n_clusters} clusters...")

        # Setup cluster manager
        cluster_manager = UserClusterManager(
            n_clusters=n_clusters,
            embedding_dim=item_embeddings.shape[1],
            n_items=len(item_embeddings),
        )
        cluster_manager.set_item_embeddings(item_embeddings)
        cluster_manager.initialize_from_interactions(
            user_ids=train_data.user_ids,
            item_ids=train_data.item_ids,
            ratings=train_data.ratings,
        )

        # Setup cache
        cache_config = CacheConfig(
            local_cache_size=5000,
            use_redis_cache=False,
        )
        cache_manager = CacheManager(
            cache_config=cache_config,
            cluster_manager=cluster_manager,
        )

        # Run simulation
        sim_config = SimulationConfig(
            n_requests=n_requests,
            n_warmup_requests=500,
            eval_sample_rate=0.1,
        )

        simulator = OnlineSimulator(
            recommender=recommender,
            cache_manager=cache_manager,
            cluster_manager=cluster_manager,
            config=sim_config,
        )

        result = simulator.run_simulation(
            n_users=train_data.n_users,
            n_items=train_data.n_items,
            ground_truth=test_ground_truth,
            verbose=False,
        )

        results.append({
            "n_clusters": n_clusters,
            "hit_rate": result.hit_rate,
            "avg_ndcg": result.avg_ndcg,
            "ndcg_degradation": result.ndcg_degradation,
            "avg_latency_ms": result.avg_latency_ms,
        })

        print(f"  Hit Rate: {result.hit_rate:.1%}, "
              f"NDCG: {result.avg_ndcg:.4f}, "
              f"Degradation: {result.ndcg_degradation:.2%}")

    return {"cluster_granularity": results}


def run_cache_size_experiment(
    recommender,
    train_data,
    test_ground_truth,
    cluster_manager,
    cache_sizes: List[int] = [100, 500, 1000, 5000, 10000],
    n_requests: int = 5000,
) -> Dict:
    """
    Experiment: How does cache size affect performance?
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Cache Size Analysis")
    print("=" * 60)

    results = []

    for cache_size in cache_sizes:
        print(f"\nTesting with cache size {cache_size}...")

        cache_config = CacheConfig(
            local_cache_size=cache_size,
            use_redis_cache=False,
        )
        cache_manager = CacheManager(
            cache_config=cache_config,
            cluster_manager=cluster_manager,
        )

        sim_config = SimulationConfig(
            n_requests=n_requests,
            n_warmup_requests=500,
            eval_sample_rate=0.1,
        )

        simulator = OnlineSimulator(
            recommender=recommender,
            cache_manager=cache_manager,
            cluster_manager=cluster_manager,
            config=sim_config,
        )

        result = simulator.run_simulation(
            n_users=train_data.n_users,
            n_items=train_data.n_items,
            ground_truth=test_ground_truth,
            verbose=False,
        )

        results.append({
            "cache_size": cache_size,
            "hit_rate": result.hit_rate,
            "avg_latency_ms": result.avg_latency_ms,
        })

        print(f"  Hit Rate: {result.hit_rate:.1%}, "
              f"Avg Latency: {result.avg_latency_ms:.2f}ms")

    return {"cache_size": results}


def run_traffic_pattern_experiment(
    recommender,
    train_data,
    test_ground_truth,
    cluster_manager,
    n_requests: int = 5000,
) -> Dict:
    """
    Experiment: How do different traffic patterns affect cache performance?
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Traffic Pattern Analysis")
    print("=" * 60)

    patterns = ["uniform", "zipf", "clustered"]
    results = []

    for pattern in patterns:
        print(f"\nTesting with {pattern} traffic distribution...")

        cache_config = CacheConfig(
            local_cache_size=5000,
            use_redis_cache=False,
        )
        cache_manager = CacheManager(
            cache_config=cache_config,
            cluster_manager=cluster_manager,
        )

        sim_config = SimulationConfig(
            n_requests=n_requests,
            n_warmup_requests=500,
            user_distribution=pattern,
            eval_sample_rate=0.1,
        )

        simulator = OnlineSimulator(
            recommender=recommender,
            cache_manager=cache_manager,
            cluster_manager=cluster_manager,
            config=sim_config,
        )

        result = simulator.run_simulation(
            n_users=train_data.n_users,
            n_items=train_data.n_items,
            ground_truth=test_ground_truth,
            verbose=False,
        )

        results.append({
            "pattern": pattern,
            "hit_rate": result.hit_rate,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
        })

        print(f"  Hit Rate: {result.hit_rate:.1%}, "
              f"Avg Latency: {result.avg_latency_ms:.2f}ms, "
              f"P95 Latency: {result.p95_latency_ms:.2f}ms")

    return {"traffic_pattern": results}


def run_latency_breakdown(
    recommender,
    train_data,
    cluster_manager,
    n_samples: int = 1000,
) -> Dict:
    """
    Analyze latency breakdown for cache hit vs. miss.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Latency Breakdown")
    print("=" * 60)

    cache_config = CacheConfig(
        local_cache_size=5000,
        use_redis_cache=False,
    )
    cache_manager = CacheManager(
        cache_config=cache_config,
        cluster_manager=cluster_manager,
    )

    cached_recommender = CacheAwareRecommender(
        recommender=recommender,
        cache_manager=cache_manager,
    )

    hit_latencies = []
    miss_latencies = []
    fresh_latencies = []

    users = list(range(train_data.n_users))

    # Warm up cache
    for user_id in np.random.choice(users, size=min(500, len(users)), replace=False):
        cached_recommender.recommend(user_id, n=20)

    # Measure latencies
    for _ in range(n_samples):
        user_id = np.random.choice(users)

        # Cached path
        start = time.time()
        _, metadata = cached_recommender.recommend(user_id, n=20)
        latency = (time.time() - start) * 1000

        if metadata["cache_hit"]:
            hit_latencies.append(latency)
        else:
            miss_latencies.append(latency)

        # Fresh path
        start = time.time()
        recommender.recommend(user_id, n=20)
        fresh_latencies.append((time.time() - start) * 1000)

    results = {
        "cache_hit": {
            "count": len(hit_latencies),
            "mean": np.mean(hit_latencies) if hit_latencies else 0,
            "p50": np.percentile(hit_latencies, 50) if hit_latencies else 0,
            "p95": np.percentile(hit_latencies, 95) if hit_latencies else 0,
            "p99": np.percentile(hit_latencies, 99) if hit_latencies else 0,
        },
        "cache_miss": {
            "count": len(miss_latencies),
            "mean": np.mean(miss_latencies) if miss_latencies else 0,
            "p50": np.percentile(miss_latencies, 50) if miss_latencies else 0,
            "p95": np.percentile(miss_latencies, 95) if miss_latencies else 0,
            "p99": np.percentile(miss_latencies, 99) if miss_latencies else 0,
        },
        "fresh": {
            "count": len(fresh_latencies),
            "mean": np.mean(fresh_latencies),
            "p50": np.percentile(fresh_latencies, 50),
            "p95": np.percentile(fresh_latencies, 95),
            "p99": np.percentile(fresh_latencies, 99),
        },
    }

    print("\nLatency Breakdown (ms):")
    print("-" * 50)
    for path, stats in results.items():
        print(f"  {path.replace('_', ' ').title():12s}: "
              f"mean={stats['mean']:.2f}, p50={stats['p50']:.2f}, "
              f"p95={stats['p95']:.2f}, p99={stats['p99']:.2f}")

    if hit_latencies and fresh_latencies:
        speedup = np.mean(fresh_latencies) / np.mean(hit_latencies)
        print(f"\n  Speedup (cache hit vs fresh): {speedup:.1f}x")

    return {"latency_breakdown": results}


def save_results(results: Dict, output_dir: str):
    """Save benchmark results."""
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


def print_summary(all_results: Dict):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if "cluster_granularity" in all_results:
        print("\n📊 Cluster Granularity:")
        best = max(all_results["cluster_granularity"],
                   key=lambda x: x["hit_rate"] - x["ndcg_degradation"])
        print(f"  Best config: {best['n_clusters']} clusters")
        print(f"    - Hit Rate: {best['hit_rate']:.1%}")
        print(f"    - Quality Loss: {best['ndcg_degradation']:.2%}")

    if "cache_size" in all_results:
        print("\n📦 Cache Size:")
        best = max(all_results["cache_size"], key=lambda x: x["hit_rate"])
        print(f"  Best hit rate at size {best['cache_size']}: {best['hit_rate']:.1%}")

    if "traffic_pattern" in all_results:
        print("\n🚦 Traffic Patterns:")
        for r in all_results["traffic_pattern"]:
            print(f"  {r['pattern']:10s}: {r['hit_rate']:.1%} hit rate")

    if "latency_breakdown" in all_results:
        lb = all_results["latency_breakdown"]
        print("\n⚡ Latency (P50):")
        print(f"  Cache Hit:  {lb['cache_hit']['p50']:.2f}ms")
        print(f"  Cache Miss: {lb['cache_miss']['p50']:.2f}ms")
        print(f"  Fresh:      {lb['fresh']['p50']:.2f}ms")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("RecCache Benchmark Suite")
    print("=" * 60)

    # Load or generate data
    if args.dataset == "movielens":
        print("\nLoading MovieLens 100K...")
        loader = DataLoader(data_dir="data")
        train_data, val_data, test_data = loader.load_movielens_100k()
    else:
        print(f"\nGenerating synthetic data ({args.n_users} users, {args.n_items} items)...")
        train_data = generate_synthetic_data(
            n_users=args.n_users,
            n_items=args.n_items,
            n_interactions=args.n_users * 50,
        )
        test_data = generate_synthetic_data(
            n_users=args.n_users,
            n_items=args.n_items,
            n_interactions=args.n_users * 10,
        )

    print(f"  Users: {train_data.n_users}, Items: {train_data.n_items}")

    # Train recommender
    print("\nTraining recommender...")
    recommender = MatrixFactorizationRecommender(
        n_users=train_data.n_users,
        n_items=train_data.n_items,
        embedding_dim=64,
    )
    recommender.fit(
        user_ids=train_data.user_ids,
        item_ids=train_data.item_ids,
        ratings=train_data.ratings,
        epochs=10,
        verbose=True,
    )

    # Build ground truth
    test_ground_truth = defaultdict(set)
    for user_id, item_id, rating in zip(
        test_data.user_ids, test_data.item_ids, test_data.ratings
    ):
        if rating >= 4.0:
            test_ground_truth[int(user_id)].add(int(item_id))
    test_ground_truth = dict(test_ground_truth)

    # Default cluster manager for some experiments
    item_embeddings = recommender.get_all_item_embeddings()
    default_cluster_manager = UserClusterManager(
        n_clusters=50,
        embedding_dim=item_embeddings.shape[1],
        n_items=len(item_embeddings),
    )
    default_cluster_manager.set_item_embeddings(item_embeddings)
    default_cluster_manager.initialize_from_interactions(
        user_ids=train_data.user_ids,
        item_ids=train_data.item_ids,
        ratings=train_data.ratings,
    )

    # Run experiments
    all_results = {}

    all_results.update(run_cluster_granularity_experiment(
        recommender, train_data, test_ground_truth,
        n_requests=args.n_requests,
    ))

    all_results.update(run_cache_size_experiment(
        recommender, train_data, test_ground_truth,
        default_cluster_manager,
        n_requests=args.n_requests,
    ))

    all_results.update(run_traffic_pattern_experiment(
        recommender, train_data, test_ground_truth,
        default_cluster_manager,
        n_requests=args.n_requests,
    ))

    all_results.update(run_latency_breakdown(
        recommender, train_data, default_cluster_manager,
    ))

    # Save and summarize
    save_results(all_results, args.output_dir)
    print_summary(all_results)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
