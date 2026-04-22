#!/usr/bin/env python3
"""
Final experiment script for RecCache validation.
Runs key experiments on ML-100K and ML-1M datasets.
"""

import sys
from pathlib import Path
from collections import defaultdict
import json
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats

from reccache.utils.data_loader import DataLoader
from reccache.utils.config import CacheConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.cache.baselines import create_cache
from reccache.cache.manager import CacheManager, RecommendationRequest
from reccache.clustering.user_cluster import UserClusterManager
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig
from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker
import time


def set_seed(seed):
    np.random.seed(seed)


def build_ground_truth(test_data, min_rating=4.0):
    ground_truth = defaultdict(set)
    for user_id, item_id, rating in zip(test_data.user_ids, test_data.item_ids, test_data.ratings):
        if rating >= min_rating:
            ground_truth[int(user_id)].add(int(item_id))
    return dict(ground_truth)


def run_experiments(dataset_name, n_runs=3, n_requests=3000):
    """Run experiments on a single dataset."""
    print(f"\n{'#'*70}")
    print(f"# Dataset: {dataset_name.upper()}")
    print('#'*70, flush=True)

    # Load data
    loader = DataLoader("data")
    try:
        train, val, test = loader.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    print(f"Users: {train.n_users}, Items: {train.n_items}, Interactions: {len(train.user_ids)}", flush=True)
    ground_truth = build_ground_truth(test)

    # Train recommender
    print("\nTraining MF recommender...", flush=True)
    recommender = MatrixFactorizationRecommender(
        n_users=train.n_users,
        n_items=train.n_items,
        embedding_dim=64,
    )
    recommender.fit(train.user_ids, train.item_ids, train.ratings, epochs=10, verbose=True)

    # Setup clustering
    item_embeddings = recommender.get_all_item_embeddings()
    cluster_manager = UserClusterManager(
        n_clusters=50,
        embedding_dim=item_embeddings.shape[1],
        n_items=len(item_embeddings),
    )
    cluster_manager.set_item_embeddings(item_embeddings)
    cluster_manager.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # Prepare user history
    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))

    results = {}

    # Test configurations
    configs = {
        "LRU (no clustering)": {"use_clustering": False, "use_quality": False, "use_reranker": False},
        "LRU + Clustering": {"use_clustering": True, "use_quality": False, "use_reranker": False},
        "RecCache (Full)": {"use_clustering": True, "use_quality": True, "use_reranker": True},
    }

    for config_name, config in configs.items():
        print(f"\n  Testing: {config_name}", flush=True)
        run_results = {"hit_rates": [], "ndcgs": [], "latencies": []}

        for run in range(n_runs):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False, quality_threshold=0.15)
            cm = cluster_manager if config["use_clustering"] else None
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

            if not config["use_quality"]:
                cache_manager.local_cache = create_cache("lru", max_size=5000)
            else:
                quality_predictor = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
                cache_manager.set_quality_predictor(quality_predictor)

            reranker = None
            if config["use_reranker"]:
                reranker = LightweightReranker(history_weight=0.3, recency_weight=0.3, diversity_weight=0.2)
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])

            sim_config = SimulationConfig(n_requests=n_requests, n_warmup_requests=300, eval_sample_rate=0.1)
            simulator = OnlineSimulator(
                recommender=recommender,
                cache_manager=cache_manager,
                cluster_manager=cm,
                reranker=reranker,
                config=sim_config,
            )

            result = simulator.run_simulation(
                n_users=train.n_users,
                n_items=train.n_items,
                ground_truth=ground_truth,
                verbose=False,
            )

            run_results["hit_rates"].append(result.hit_rate)
            run_results["ndcgs"].append(result.avg_ndcg)
            run_results["latencies"].append(result.avg_latency_ms)

        results[config_name] = run_results
        hr_mean = np.mean(run_results["hit_rates"])
        hr_std = np.std(run_results["hit_rates"])
        ndcg_mean = np.mean(run_results["ndcgs"])
        print(f"    Hit Rate: {hr_mean:.4f}±{hr_std:.4f}, NDCG: {ndcg_mean:.4f}", flush=True)

    return results


def run_efficiency_test(dataset_name="ml-100k"):
    """Run efficiency benchmark."""
    print(f"\n{'='*60}")
    print("EFFICIENCY ANALYSIS")
    print('='*60, flush=True)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(dataset_name)

    recommender = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=64
    )
    recommender.fit(train.user_ids, train.item_ids, train.ratings, epochs=10, verbose=False)

    item_embeddings = recommender.get_all_item_embeddings()
    cluster_manager = UserClusterManager(n_clusters=50, embedding_dim=item_embeddings.shape[1], n_items=len(item_embeddings))
    cluster_manager.set_item_embeddings(item_embeddings)
    cluster_manager.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
    cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cluster_manager)

    test_users = list(range(min(500, train.n_users)))

    # Warm up cache
    for user_id in test_users[:100]:
        recs = recommender.recommend(user_id, n=20)
        request = RecommendationRequest(user_id=user_id, n_recommendations=20)
        cache_manager.put(request, recs)

    # Throughput test
    for name, use_cache in [("Cached", True), ("Fresh", False)]:
        start = time.time()
        n_requests = 1000

        for i in range(n_requests):
            user_id = test_users[i % len(test_users)]
            if use_cache:
                request = RecommendationRequest(user_id=user_id, n_recommendations=20)
                cache_result = cache_manager.get(request)
                if not cache_result.hit:
                    recs = recommender.recommend(user_id, n=20)
                    cache_manager.put(request, recs)
            else:
                _ = recommender.recommend(user_id, n=20)

        elapsed = time.time() - start
        print(f"  {name}: {n_requests / elapsed:.0f} req/s", flush=True)


def main():
    print("="*70)
    print("RecCache Final Experiment Suite")
    print("="*70, flush=True)

    all_results = {}

    # Run on both datasets
    for dataset in ["ml-100k", "ml-1m"]:
        results = run_experiments(dataset, n_runs=3, n_requests=3000)
        if results:
            all_results[dataset] = results

    # Run efficiency test
    run_efficiency_test()

    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70, flush=True)

    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:", flush=True)

        baseline = np.mean(results["LRU (no clustering)"]["hit_rates"])
        reccache = np.mean(results["RecCache (Full)"]["hit_rates"])

        baseline_ndcg = np.mean(results["LRU (no clustering)"]["ndcgs"])
        reccache_ndcg = np.mean(results["RecCache (Full)"]["ndcgs"])

        print(f"  LRU (no clustering): Hit Rate={baseline:.4f}, NDCG={baseline_ndcg:.4f}")
        print(f"  RecCache (Full):     Hit Rate={reccache:.4f}, NDCG={reccache_ndcg:.4f}")
        print(f"  Improvement:         Hit Rate +{(reccache-baseline)/baseline*100:.1f}%, NDCG {(reccache_ndcg-baseline_ndcg)/baseline_ndcg*100:+.1f}%")

        # Statistical test
        _, p_hr = stats.ttest_rel(
            results["RecCache (Full)"]["hit_rates"],
            results["LRU (no clustering)"]["hit_rates"]
        )
        sig = "***" if p_hr < 0.001 else "**" if p_hr < 0.01 else "*" if p_hr < 0.05 else ""
        print(f"  Significance (Hit Rate): p={p_hr:.4f} {sig}")

    # Save results
    with open("results/final_results.json", "w") as f:
        # Convert numpy to python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32, np.int_)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32, np.float_)):
                return float(obj)
            return obj

        json.dump(all_results, f, indent=2, default=convert)

    print("\nResults saved to results/final_results.json")
    print("Experiment complete!")


if __name__ == "__main__":
    main()
