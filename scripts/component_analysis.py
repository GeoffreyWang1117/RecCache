#!/usr/bin/env python3
"""Analyze contribution of each RecCache component."""

import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats

from reccache.utils.data_loader import DataLoader
from reccache.utils.config import CacheConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.cache.baselines import create_cache
from reccache.cache.manager import CacheManager
from reccache.clustering.user_cluster import UserClusterManager
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig
from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker


def set_seed(seed):
    np.random.seed(seed)


def build_ground_truth(test_data, min_rating=4.0):
    ground_truth = defaultdict(set)
    for user_id, item_id, rating in zip(test_data.user_ids, test_data.item_ids, test_data.ratings):
        if rating >= min_rating:
            ground_truth[int(user_id)].add(int(item_id))
    return dict(ground_truth)


def main():
    print("="*70)
    print("Component Analysis: Understanding RecCache Performance")
    print("="*70)

    # Load data
    loader = DataLoader("data")
    train, val, test = loader.load_dataset("ml-100k")
    print(f"Dataset: ML-100K, Users: {train.n_users}, Items: {train.n_items}")

    ground_truth = build_ground_truth(test)

    # Train recommender
    print("\nTraining MF recommender...")
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

    # Configurations to test
    configs = {
        "LRU (no clustering)": {
            "use_clustering": False,
            "use_quality_aware_eviction": False,
            "use_reranker": False,
        },
        "LRU + Clustering": {
            "use_clustering": True,
            "use_quality_aware_eviction": False,
            "use_reranker": False,
        },
        "LRU + Clustering + Quality Eviction": {
            "use_clustering": True,
            "use_quality_aware_eviction": True,
            "use_reranker": False,
        },
        "RecCache Full (+ Reranker)": {
            "use_clustering": True,
            "use_quality_aware_eviction": True,
            "use_reranker": True,
        },
    }

    n_runs = 5
    results = {}

    for config_name, config in configs.items():
        print(f"\nTesting: {config_name}")
        run_results = {"hit_rates": [], "ndcgs": [], "latencies": []}

        for run in range(n_runs):
            set_seed(42 + run)

            cache_config = CacheConfig(
                local_cache_size=5000,
                use_redis_cache=False,
                quality_threshold=0.15,
            )

            cm = cluster_manager if config["use_clustering"] else None

            cache_manager = CacheManager(
                cache_config=cache_config,
                cluster_manager=cm,
            )

            # If not using quality-aware eviction, use simple LRU cache
            if not config["use_quality_aware_eviction"]:
                cache_manager.local_cache = create_cache("lru", max_size=5000)
            else:
                # Set quality predictor for quality-aware storage
                quality_predictor = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
                cache_manager.set_quality_predictor(quality_predictor)

            # Setup reranker if enabled
            reranker = None
            if config["use_reranker"]:
                reranker = LightweightReranker(
                    history_weight=0.3,
                    recency_weight=0.3,
                    diversity_weight=0.2,
                )
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])

            sim_config = SimulationConfig(
                n_requests=3000,
                n_warmup_requests=500,
                eval_sample_rate=0.1,
            )

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
        ndcg_std = np.std(run_results["ndcgs"])
        lat_mean = np.mean(run_results["latencies"])

        print(f"  Hit Rate: {hr_mean:.4f}±{hr_std:.4f}")
        print(f"  NDCG: {ndcg_mean:.4f}±{ndcg_std:.4f}")
        print(f"  Latency: {lat_mean:.2f}ms")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Configuration':<40}{'Hit Rate':>12}{'NDCG':>12}{'Latency':>12}")
    print("-"*70)

    baseline_hr = np.mean(results["LRU (no clustering)"]["hit_rates"])
    baseline_ndcg = np.mean(results["LRU (no clustering)"]["ndcgs"])

    for config_name, run_results in results.items():
        hr_mean = np.mean(run_results["hit_rates"])
        ndcg_mean = np.mean(run_results["ndcgs"])
        lat_mean = np.mean(run_results["latencies"])

        hr_change = (hr_mean - baseline_hr) / baseline_hr * 100
        ndcg_change = (ndcg_mean - baseline_ndcg) / baseline_ndcg * 100

        print(f"{config_name:<40}{hr_mean:.4f} ({hr_change:+.1f}%)  {ndcg_mean:.4f} ({ndcg_change:+.1f}%)  {lat_mean:.2f}ms")

    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS (vs LRU no clustering)")
    print("="*70)

    baseline = results["LRU (no clustering)"]

    for config_name, run_results in results.items():
        if config_name == "LRU (no clustering)":
            continue

        _, p_hr = stats.ttest_rel(run_results["hit_rates"], baseline["hit_rates"])
        _, p_ndcg = stats.ttest_rel(run_results["ndcgs"], baseline["ndcgs"])

        sig_hr = "***" if p_hr < 0.001 else "**" if p_hr < 0.01 else "*" if p_hr < 0.05 else ""
        sig_ndcg = "***" if p_ndcg < 0.001 else "**" if p_ndcg < 0.01 else "*" if p_ndcg < 0.05 else ""

        print(f"{config_name}")
        print(f"  Hit Rate: p={p_hr:.4f} {sig_hr}")
        print(f"  NDCG: p={p_ndcg:.4f} {sig_ndcg}")

    # Component contribution analysis
    print("\n" + "="*70)
    print("COMPONENT CONTRIBUTIONS")
    print("="*70)

    base = np.mean(results["LRU (no clustering)"]["hit_rates"])
    with_cluster = np.mean(results["LRU + Clustering"]["hit_rates"])
    with_quality = np.mean(results["LRU + Clustering + Quality Eviction"]["hit_rates"])
    full = np.mean(results["RecCache Full (+ Reranker)"]["hit_rates"])

    print(f"Clustering contribution: +{(with_cluster - base) / base * 100:.1f}% hit rate")
    print(f"Quality-aware eviction: {(with_quality - with_cluster) / base * 100:+.1f}% hit rate")
    print(f"Reranker contribution: {(full - with_quality) / base * 100:+.1f}% hit rate")


if __name__ == "__main__":
    main()
