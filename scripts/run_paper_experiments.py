#!/usr/bin/env python3
"""
Complete experiment suite for ICBINB paper.
Includes Oracle bounds, multiple baselines, and visualization.
"""

import sys
from pathlib import Path
from collections import defaultdict
import json
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from reccache.utils.data_loader import DataLoader
from reccache.utils.config import CacheConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.cache.baselines import create_cache
from reccache.cache.manager import CacheManager, RecommendationRequest
from reccache.cache.oracle import BeladyCache, compute_oracle_bounds
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


def generate_request_sequence(n_users: int, n_requests: int, distribution: str = "zipf") -> list:
    """Generate a sequence of user requests following specified distribution."""
    if distribution == "zipf":
        # Zipf distribution (realistic: some users much more active)
        weights = 1.0 / np.arange(1, n_users + 1) ** 1.2
        weights /= weights.sum()
        return list(np.random.choice(n_users, size=n_requests, p=weights))
    elif distribution == "uniform":
        return list(np.random.randint(0, n_users, size=n_requests))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def run_oracle_experiment(dataset_name: str, n_requests: int = 5000):
    """Compute Oracle (Belady) upper bounds."""
    print(f"\n{'='*60}")
    print(f"ORACLE BOUNDS - {dataset_name}")
    print('='*60, flush=True)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(dataset_name)

    # Train recommender for clustering
    recommender = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=64
    )
    recommender.fit(train.user_ids, train.item_ids, train.ratings, epochs=10, verbose=False)

    # Setup clustering
    item_embeddings = recommender.get_all_item_embeddings()
    cluster_manager = UserClusterManager(
        n_clusters=50, embedding_dim=item_embeddings.shape[1], n_items=len(item_embeddings)
    )
    cluster_manager.set_item_embeddings(item_embeddings)
    cluster_manager.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # Build user to cluster mapping
    user_to_cluster = {}
    for uid in range(train.n_users):
        info = cluster_manager.get_user_cluster(uid)
        user_to_cluster[uid] = info.cluster_id

    # Generate request sequence
    set_seed(42)
    user_sequence = generate_request_sequence(train.n_users, n_requests, "zipf")

    # Compute Oracle bounds
    cache_sizes = [500, 1000, 2500, 5000, 10000]
    oracle_results = compute_oracle_bounds(user_sequence, user_to_cluster, cache_sizes)

    print("\n  Cache Size | User-Level OPT | Cluster-Level OPT | Improvement")
    print("-" * 65)
    for size in cache_sizes:
        user_opt = oracle_results["user_level_optimal"][size]
        cluster_opt = oracle_results["cluster_level_optimal"][size]
        improvement = (cluster_opt - user_opt) / user_opt * 100 if user_opt > 0 else 0
        print(f"  {size:>10} | {user_opt:>14.4f} | {cluster_opt:>17.4f} | {improvement:>+10.1f}%")

    return oracle_results


def run_baseline_comparison(dataset_name: str, n_runs: int = 3, n_requests: int = 3000):
    """Compare all cache strategies including Oracle bounds."""
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON - {dataset_name}")
    print('='*60, flush=True)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(dataset_name)
    ground_truth = build_ground_truth(test)

    # Train recommender
    print("\n  Training recommender...", flush=True)
    recommender = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=64
    )
    recommender.fit(train.user_ids, train.item_ids, train.ratings, epochs=10, verbose=False)

    item_embeddings = recommender.get_all_item_embeddings()
    cluster_manager = UserClusterManager(
        n_clusters=50, embedding_dim=item_embeddings.shape[1], n_items=len(item_embeddings)
    )
    cluster_manager.set_item_embeddings(item_embeddings)
    cluster_manager.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    user_history = defaultdict(list)
    for uid, iid in zip(train_data.user_ids, train_data.item_ids):
        user_history[int(uid)].append(int(iid))

    # Strategies to test
    strategies = {
        "LRU (no clustering)": {"clustering": False, "eviction": "lru"},
        "LFU (no clustering)": {"clustering": False, "eviction": "lfu"},
        "FIFO (no clustering)": {"clustering": False, "eviction": "fifo"},
        "LRU + Clustering": {"clustering": True, "eviction": "lru"},
        "RecCache (Full)": {"clustering": True, "eviction": "quality", "reranker": True},
    }

    results = {}

    for name, config in strategies.items():
        print(f"\n  Testing: {name}", flush=True)
        run_results = {"hit_rates": [], "ndcgs": []}

        for run in range(n_runs):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
            cm = cluster_manager if config["clustering"] else None
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

            if config["eviction"] != "quality":
                cache_manager.local_cache = create_cache(config["eviction"], max_size=5000)
            else:
                qp = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
                cache_manager.set_quality_predictor(qp)

            reranker = None
            if config.get("reranker"):
                reranker = LightweightReranker(history_weight=0.3, recency_weight=0.3, diversity_weight=0.2)
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])

            sim_config = SimulationConfig(n_requests=n_requests, n_warmup_requests=300, eval_sample_rate=0.1)
            simulator = OnlineSimulator(
                recommender=recommender, cache_manager=cache_manager,
                cluster_manager=cm, reranker=reranker, config=sim_config
            )

            result = simulator.run_simulation(
                n_users=train.n_users, n_items=train.n_items,
                ground_truth=ground_truth, verbose=False
            )

            run_results["hit_rates"].append(result.hit_rate)
            run_results["ndcgs"].append(result.avg_ndcg)

        results[name] = {
            "hit_rate_mean": np.mean(run_results["hit_rates"]),
            "hit_rate_std": np.std(run_results["hit_rates"]),
            "ndcg_mean": np.mean(run_results["ndcgs"]),
            "ndcg_std": np.std(run_results["ndcgs"]),
            "raw": run_results,
        }

        print(f"    Hit Rate: {results[name]['hit_rate_mean']:.4f}±{results[name]['hit_rate_std']:.4f}")
        print(f"    NDCG: {results[name]['ndcg_mean']:.4f}±{results[name]['ndcg_std']:.4f}")

    return results


def run_cluster_sensitivity(dataset_name: str):
    """Analyze sensitivity to number of clusters."""
    print(f"\n{'='*60}")
    print(f"CLUSTER SENSITIVITY - {dataset_name}")
    print('='*60, flush=True)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(dataset_name)
    ground_truth = build_ground_truth(test)

    recommender = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=64
    )
    recommender.fit(train.user_ids, train.item_ids, train.ratings, epochs=10, verbose=False)
    item_embeddings = recommender.get_all_item_embeddings()

    n_clusters_list = [10, 25, 50, 100, 200, 500]
    results = {}

    for n_clusters in n_clusters_list:
        print(f"\n  K={n_clusters}...", flush=True)

        cluster_manager = UserClusterManager(
            n_clusters=n_clusters, embedding_dim=item_embeddings.shape[1], n_items=len(item_embeddings)
        )
        cluster_manager.set_item_embeddings(item_embeddings)
        cluster_manager.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

        run_results = []
        for run in range(3):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cluster_manager)

            sim_config = SimulationConfig(n_requests=3000, n_warmup_requests=300, eval_sample_rate=0.1)
            simulator = OnlineSimulator(
                recommender=recommender, cache_manager=cache_manager,
                cluster_manager=cluster_manager, config=sim_config
            )

            result = simulator.run_simulation(
                n_users=train.n_users, n_items=train.n_items,
                ground_truth=ground_truth, verbose=False
            )
            run_results.append({"hit_rate": result.hit_rate, "ndcg": result.avg_ndcg})

        results[n_clusters] = {
            "hit_rate": np.mean([r["hit_rate"] for r in run_results]),
            "ndcg": np.mean([r["ndcg"] for r in run_results]),
        }
        print(f"    Hit Rate: {results[n_clusters]['hit_rate']:.4f}, NDCG: {results[n_clusters]['ndcg']:.4f}")

    return results


def create_paper_figures(all_results: dict):
    """Generate figures for the paper."""
    print("\n  Generating figures...", flush=True)

    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Hit Rate comparison bar chart
    if "baselines" in all_results:
        fig, ax = plt.subplots(figsize=(8, 4))

        baselines = all_results["baselines"]
        methods = list(baselines.keys())
        hit_rates = [baselines[m]["hit_rate_mean"] for m in methods]
        errors = [baselines[m]["hit_rate_std"] for m in methods]

        colors = ['#1f77b4'] * (len(methods) - 1) + ['#2ca02c']  # Green for RecCache
        bars = ax.bar(range(len(methods)), hit_rates, yerr=errors, capsize=3, color=colors)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Cache Hit Rate')
        ax.set_title('Cache Strategy Comparison')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(fig_dir / "hit_rate_comparison.pdf", bbox_inches='tight')
        plt.savefig(fig_dir / "hit_rate_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Figure 2: Cluster sensitivity
    if "cluster_sensitivity" in all_results:
        fig, ax = plt.subplots(figsize=(6, 4))

        sensitivity = all_results["cluster_sensitivity"]
        k_values = sorted(sensitivity.keys())
        hit_rates = [sensitivity[k]["hit_rate"] for k in k_values]

        ax.plot(k_values, hit_rates, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Cache Hit Rate')
        ax.set_title('Sensitivity to Cluster Count')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "cluster_sensitivity.pdf", bbox_inches='tight')
        plt.savefig(fig_dir / "cluster_sensitivity.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Figures saved to {fig_dir}/")


def main():
    print("="*70)
    print("RecCache Paper Experiments")
    print("="*70, flush=True)

    all_results = {}

    # Run experiments on ML-100K
    dataset = "ml-100k"

    # 1. Oracle bounds
    oracle_results = run_oracle_experiment(dataset)
    all_results["oracle"] = oracle_results

    # 2. Baseline comparison
    loader = DataLoader("data")
    train_data, _, test_data = loader.load_dataset(dataset)

    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON - {dataset}")
    print('='*60, flush=True)

    ground_truth = build_ground_truth(test_data)

    recommender = MatrixFactorizationRecommender(
        n_users=train_data.n_users, n_items=train_data.n_items, embedding_dim=64
    )
    recommender.fit(train_data.user_ids, train_data.item_ids, train_data.ratings, epochs=10, verbose=True)

    item_embeddings = recommender.get_all_item_embeddings()
    cluster_manager = UserClusterManager(
        n_clusters=50, embedding_dim=item_embeddings.shape[1], n_items=len(item_embeddings)
    )
    cluster_manager.set_item_embeddings(item_embeddings)
    cluster_manager.initialize_from_interactions(train_data.user_ids, train_data.item_ids, train_data.ratings)

    user_history = defaultdict(list)
    for uid, iid in zip(train_data.user_ids, train_data.item_ids):
        user_history[int(uid)].append(int(iid))

    strategies = {
        "LRU (no cluster)": {"clustering": False, "eviction": "lru"},
        "LFU (no cluster)": {"clustering": False, "eviction": "lfu"},
        "LRU + Cluster": {"clustering": True, "eviction": "lru"},
        "RecCache": {"clustering": True, "eviction": "quality", "reranker": True},
    }

    baseline_results = {}
    n_runs = 3

    for name, config in strategies.items():
        print(f"\n  Testing: {name}", flush=True)
        run_results = {"hit_rates": [], "ndcgs": []}

        for run in range(n_runs):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
            cm = cluster_manager if config["clustering"] else None
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

            if config["eviction"] != "quality":
                cache_manager.local_cache = create_cache(config["eviction"], max_size=5000)
            else:
                qp = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
                cache_manager.set_quality_predictor(qp)

            reranker = None
            if config.get("reranker"):
                reranker = LightweightReranker(history_weight=0.3, recency_weight=0.3, diversity_weight=0.2)
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])

            sim_config = SimulationConfig(n_requests=3000, n_warmup_requests=300, eval_sample_rate=0.1)
            simulator = OnlineSimulator(
                recommender=recommender, cache_manager=cache_manager,
                cluster_manager=cm, reranker=reranker, config=sim_config
            )

            result = simulator.run_simulation(
                n_users=train_data.n_users, n_items=train_data.n_items,
                ground_truth=ground_truth, verbose=False
            )

            run_results["hit_rates"].append(result.hit_rate)
            run_results["ndcgs"].append(result.avg_ndcg)

        baseline_results[name] = {
            "hit_rate_mean": np.mean(run_results["hit_rates"]),
            "hit_rate_std": np.std(run_results["hit_rates"]),
            "ndcg_mean": np.mean(run_results["ndcgs"]),
            "ndcg_std": np.std(run_results["ndcgs"]),
        }

        print(f"    Hit Rate: {baseline_results[name]['hit_rate_mean']:.4f}±{baseline_results[name]['hit_rate_std']:.4f}")

    all_results["baselines"] = baseline_results

    # 3. Cluster sensitivity
    sensitivity_results = run_cluster_sensitivity(dataset)
    all_results["cluster_sensitivity"] = sensitivity_results

    # 4. Generate figures
    create_paper_figures(all_results)

    # 5. Save results
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70, flush=True)

    print("\nOracle Bounds (cache size=5000):")
    print(f"  User-level OPT: {oracle_results['user_level_optimal'][5000]:.4f}")
    print(f"  Cluster-level OPT: {oracle_results['cluster_level_optimal'][5000]:.4f}")

    print("\nMethod Comparison:")
    for name, res in baseline_results.items():
        print(f"  {name}: {res['hit_rate_mean']:.4f}±{res['hit_rate_std']:.4f}")

    # Gap to oracle
    reccache_hr = baseline_results["RecCache"]["hit_rate_mean"]
    cluster_opt = oracle_results["cluster_level_optimal"][5000]
    gap = (cluster_opt - reccache_hr) / cluster_opt * 100
    print(f"\nRecCache gap to cluster-level OPT: {gap:.1f}%")

    # Save to JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float_)):
            return float(obj)
        return obj

    with open("results/paper_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print("\nResults saved to results/paper_results.json")
    print("Figures saved to paper/figures/")


if __name__ == "__main__":
    main()
