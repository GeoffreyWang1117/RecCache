#!/usr/bin/env python3
"""Quick validation script for RecCache improvements."""

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
    print("="*60)
    print("Quick Validation: RecCache vs Baselines")
    print("="*60)

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

    # Test strategies
    strategies = ["LRU", "LFU", "RecCache"]
    n_runs = 5
    results = {s: {"hit_rates": [], "ndcgs": []} for s in strategies}

    for strategy in strategies:
        print(f"\nTesting {strategy}...")

        for run in range(n_runs):
            set_seed(42 + run)

            cache_config = CacheConfig(
                local_cache_size=5000,
                use_redis_cache=False,
                quality_threshold=0.15,
            )

            cache_manager = CacheManager(
                cache_config=cache_config,
                cluster_manager=cluster_manager,
            )

            reranker = None
            if strategy == "RecCache":
                # Setup Quality Predictor
                quality_predictor = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
                cache_manager.set_quality_predictor(quality_predictor)

                # Setup Reranker
                reranker = LightweightReranker(
                    history_weight=0.3,
                    recency_weight=0.3,
                    diversity_weight=0.2,
                )
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])
            else:
                cache_manager.local_cache = create_cache(strategy.lower(), max_size=5000)

            sim_config = SimulationConfig(
                n_requests=3000,
                n_warmup_requests=500,
                eval_sample_rate=0.1,
            )

            simulator = OnlineSimulator(
                recommender=recommender,
                cache_manager=cache_manager,
                cluster_manager=cluster_manager,
                reranker=reranker,
                config=sim_config,
            )

            result = simulator.run_simulation(
                n_users=train.n_users,
                n_items=train.n_items,
                ground_truth=ground_truth,
                verbose=False,
            )

            results[strategy]["hit_rates"].append(result.hit_rate)
            results[strategy]["ndcgs"].append(result.avg_ndcg)

        hr_mean = np.mean(results[strategy]["hit_rates"])
        hr_std = np.std(results[strategy]["hit_rates"])
        ndcg_mean = np.mean(results[strategy]["ndcgs"])
        ndcg_std = np.std(results[strategy]["ndcgs"])
        print(f"  Hit Rate: {hr_mean:.4f}±{hr_std:.4f}")
        print(f"  NDCG: {ndcg_mean:.4f}±{ndcg_std:.4f}")

    # Statistical test: RecCache vs LRU
    print("\n" + "="*60)
    print("Statistical Significance (RecCache vs LRU)")
    print("="*60)

    rc_hr = results["RecCache"]["hit_rates"]
    lru_hr = results["LRU"]["hit_rates"]
    rc_ndcg = results["RecCache"]["ndcgs"]
    lru_ndcg = results["LRU"]["ndcgs"]

    t_hr, p_hr = stats.ttest_rel(rc_hr, lru_hr)
    t_ndcg, p_ndcg = stats.ttest_rel(rc_ndcg, lru_ndcg)

    print(f"Hit Rate: RecCache {np.mean(rc_hr):.4f} vs LRU {np.mean(lru_hr):.4f}")
    print(f"  Difference: {np.mean(rc_hr) - np.mean(lru_hr):.4f}")
    print(f"  p-value: {p_hr:.4f} {'***' if p_hr < 0.001 else '**' if p_hr < 0.01 else '*' if p_hr < 0.05 else ''}")

    print(f"\nNDCG: RecCache {np.mean(rc_ndcg):.4f} vs LRU {np.mean(lru_ndcg):.4f}")
    print(f"  Difference: {np.mean(rc_ndcg) - np.mean(lru_ndcg):.4f}")
    print(f"  p-value: {p_ndcg:.4f} {'***' if p_ndcg < 0.001 else '**' if p_ndcg < 0.01 else '*' if p_ndcg < 0.05 else ''}")

    print("\n" + "="*60)
    if np.mean(rc_hr) >= np.mean(lru_hr):
        print("✓ RecCache achieves equal or higher Hit Rate than LRU")
    else:
        print("✗ RecCache has lower Hit Rate than LRU")

    if np.mean(rc_ndcg) >= np.mean(lru_ndcg):
        print("✓ RecCache achieves equal or higher NDCG than LRU")
    else:
        print("✗ RecCache has lower NDCG than LRU")


if __name__ == "__main__":
    main()
