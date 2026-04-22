#!/usr/bin/env python3
"""
Complete experiment suite for RecCache paper.
Runs all experiments on ML-100K, ML-1M, Amazon-Movies, and MIND-Small.
"""

import sys
from pathlib import Path
from collections import defaultdict
import json
import time
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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


# Dataset configurations
DATASET_CONFIGS = {
    "ml-100k": {
        "max_samples": None,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15, "n_requests": 5000,
    },
    "ml-1m": {
        "max_samples": None,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15, "n_requests": 5000,
    },
    "amazon-movies": {
        "max_samples": 1000000,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15, "n_requests": 5000,
    },
    "mind-small": {
        "max_samples": 500000,
        "min_user": 5, "min_item": 5,
        "implicit": True, "min_rating_gt": 0.5,
        "n_clusters": 100, "embedding_dim": 64,
        "epochs": 10, "n_requests": 5000,
    },
}


def set_seed(seed):
    np.random.seed(seed)


def build_ground_truth(test_data, min_rating=4.0):
    """Build ground truth from test data. For implicit data, use min_rating=0.5."""
    ground_truth = defaultdict(set)
    for user_id, item_id, rating in zip(test_data.user_ids, test_data.item_ids, test_data.ratings):
        if rating >= min_rating:
            ground_truth[int(user_id)].add(int(item_id))
    return dict(ground_truth)


def generate_request_sequence(n_users: int, n_requests: int, distribution: str = "zipf") -> list:
    """Generate a sequence of user requests following specified distribution."""
    if distribution == "zipf":
        weights = 1.0 / np.arange(1, n_users + 1) ** 1.2
        weights /= weights.sum()
        return list(np.random.choice(n_users, size=n_requests, p=weights))
    else:
        return list(np.random.randint(0, n_users, size=n_requests))


def run_single_dataset_experiments(dataset_name: str, n_runs: int = 5):
    """Run all experiments on a single dataset."""
    cfg = DATASET_CONFIGS[dataset_name]

    print(f"\n{'#'*70}")
    print(f"# DATASET: {dataset_name.upper()}")
    print(f"{'#'*70}\n", flush=True)

    # Load data
    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        dataset_name,
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
        max_samples=cfg["max_samples"],
    )
    ground_truth = build_ground_truth(test, min_rating=cfg["min_rating_gt"])

    print(f"Stats: {train.n_users} users, {train.n_items} items, "
          f"{len(train.user_ids)} interactions")
    print(f"Ground truth users: {len(ground_truth)}\n", flush=True)

    n_requests = cfg["n_requests"]

    # Train recommender
    print("Training MF recommender...", flush=True)
    recommender = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items,
        embedding_dim=cfg["embedding_dim"],
    )
    recommender.fit(
        train.user_ids, train.item_ids, train.ratings,
        epochs=cfg["epochs"], verbose=True,
    )

    item_embeddings = recommender.get_all_item_embeddings()

    # Setup clustering
    print("\nSetting up clustering...", flush=True)
    n_clusters = min(cfg["n_clusters"], train.n_users // 2)
    cluster_manager = UserClusterManager(
        n_clusters=n_clusters,
        embedding_dim=item_embeddings.shape[1],
        n_items=len(item_embeddings),
    )
    cluster_manager.set_item_embeddings(item_embeddings)
    cluster_manager.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # Build user to cluster mapping
    user_to_cluster = {}
    for uid in range(train.n_users):
        info = cluster_manager.get_user_cluster(uid)
        user_to_cluster[uid] = info.cluster_id

    # Prepare user history
    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))

    results = {"dataset": dataset_name, "stats": {
        "n_users": train.n_users, "n_items": train.n_items,
        "n_interactions": len(train.user_ids),
        "n_clusters": n_clusters,
        "implicit": cfg["implicit"],
    }}

    # =========================================================================
    # Experiment 1: Oracle Bounds
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Oracle Bounds (Belady's Algorithm)")
    print('='*60, flush=True)

    set_seed(42)
    user_sequence = generate_request_sequence(train.n_users, n_requests, "zipf")
    cache_sizes = [500, 1000, 2500, 5000]
    oracle_results = compute_oracle_bounds(user_sequence, user_to_cluster, cache_sizes)

    print("\n  Cache Size | User-Level OPT | Cluster-Level OPT | Improvement")
    print("-" * 65)
    for size in cache_sizes:
        user_opt = oracle_results["user_level_optimal"][size]
        cluster_opt = oracle_results["cluster_level_optimal"][size]
        improvement = (cluster_opt - user_opt) / user_opt * 100 if user_opt > 0 else 0
        print(f"  {size:>10} | {user_opt:>14.4f} | {cluster_opt:>17.4f} | {improvement:>+10.1f}%")

    results["oracle"] = oracle_results

    # =========================================================================
    # Experiment 2: Main Baseline Comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Cache Strategy Comparison")
    print('='*60, flush=True)

    strategies = {
        "LRU (no clustering)": {"clustering": False, "quality": False, "reranker": False},
        "LFU (no clustering)": {"clustering": False, "quality": False, "reranker": False, "eviction": "lfu"},
        "FIFO (no clustering)": {"clustering": False, "quality": False, "reranker": False, "eviction": "fifo"},
        "LRU + Clustering": {"clustering": True, "quality": False, "reranker": False},
        "RecCache (Full)": {"clustering": True, "quality": True, "reranker": True},
    }

    baseline_results = {}

    for name, config in strategies.items():
        print(f"\n  Testing: {name}", flush=True)
        run_results = {"hit_rates": [], "ndcgs": [], "latencies": []}

        for run in range(n_runs):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False, quality_threshold=0.15)
            cm = cluster_manager if config["clustering"] else None
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

            eviction = config.get("eviction", "lru")
            if not config["quality"]:
                cache_manager.local_cache = create_cache(eviction, max_size=5000)
            else:
                qp = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
                cache_manager.set_quality_predictor(qp)

            reranker = None
            if config.get("reranker"):
                reranker = LightweightReranker(history_weight=0.3, recency_weight=0.3, diversity_weight=0.2)
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])

            sim_config = SimulationConfig(n_requests=n_requests, n_warmup_requests=500, eval_sample_rate=0.1)
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
            run_results["latencies"].append(result.avg_latency_ms)

        baseline_results[name] = {
            "hit_rate_mean": float(np.mean(run_results["hit_rates"])),
            "hit_rate_std": float(np.std(run_results["hit_rates"])),
            "ndcg_mean": float(np.mean(run_results["ndcgs"])),
            "ndcg_std": float(np.std(run_results["ndcgs"])),
            "latency_mean": float(np.mean(run_results["latencies"])),
            "raw_hit_rates": run_results["hit_rates"],
            "raw_ndcgs": run_results["ndcgs"],
        }

        print(f"    Hit Rate: {baseline_results[name]['hit_rate_mean']:.4f}"
              f"±{baseline_results[name]['hit_rate_std']:.4f}")
        print(f"    NDCG@20:  {baseline_results[name]['ndcg_mean']:.4f}"
              f"±{baseline_results[name]['ndcg_std']:.4f}")

    results["baselines"] = baseline_results

    # =========================================================================
    # Experiment 3: Ablation Study
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Ablation Study")
    print('='*60, flush=True)

    ablations = {
        "Plain LRU": {"clustering": False, "quality": False, "reranker": False},
        "+ Clustering": {"clustering": True, "quality": False, "reranker": False},
        "+ Quality-Aware Eviction": {"clustering": True, "quality": True, "reranker": False},
        "+ Reranker (Full)": {"clustering": True, "quality": True, "reranker": True},
    }

    ablation_results = {}

    for name, config in ablations.items():
        print(f"\n  {name}", flush=True)
        run_results = {"hit_rates": [], "ndcgs": []}

        for run in range(n_runs):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False, quality_threshold=0.15)
            cm = cluster_manager if config["clustering"] else None
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

            if not config["quality"]:
                cache_manager.local_cache = create_cache("lru", max_size=5000)
            else:
                qp = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
                cache_manager.set_quality_predictor(qp)

            reranker = None
            if config.get("reranker"):
                reranker = LightweightReranker(history_weight=0.3, recency_weight=0.3, diversity_weight=0.2)
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])

            sim_config = SimulationConfig(n_requests=n_requests, n_warmup_requests=500, eval_sample_rate=0.1)
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

        ablation_results[name] = {
            "hit_rate_mean": float(np.mean(run_results["hit_rates"])),
            "hit_rate_std": float(np.std(run_results["hit_rates"])),
            "ndcg_mean": float(np.mean(run_results["ndcgs"])),
            "ndcg_std": float(np.std(run_results["ndcgs"])),
        }

        print(f"    Hit Rate: {ablation_results[name]['hit_rate_mean']:.4f}, "
              f"NDCG: {ablation_results[name]['ndcg_mean']:.4f}")

    results["ablation"] = ablation_results

    # =========================================================================
    # Experiment 4: Cluster Sensitivity
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: Cluster Count Sensitivity")
    print('='*60, flush=True)

    max_k = min(500, train.n_users // 3)
    k_values = sorted(set([
        k for k in [10, 25, 50, 100, 200, 500]
        if k <= max_k
    ]))

    sensitivity_results = {}

    for k in k_values:
        print(f"\n  K={k}", flush=True)

        cm_k = UserClusterManager(
            n_clusters=k, embedding_dim=item_embeddings.shape[1],
            n_items=len(item_embeddings),
        )
        cm_k.set_item_embeddings(item_embeddings)
        cm_k.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

        run_results = {"hit_rates": [], "ndcgs": []}

        for run in range(3):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm_k)

            sim_config = SimulationConfig(n_requests=3000, n_warmup_requests=300, eval_sample_rate=0.1)
            simulator = OnlineSimulator(
                recommender=recommender, cache_manager=cache_manager,
                cluster_manager=cm_k, config=sim_config
            )

            result = simulator.run_simulation(
                n_users=train.n_users, n_items=train.n_items,
                ground_truth=ground_truth, verbose=False
            )

            run_results["hit_rates"].append(result.hit_rate)
            run_results["ndcgs"].append(result.avg_ndcg)

        sensitivity_results[k] = {
            "hit_rate": float(np.mean(run_results["hit_rates"])),
            "ndcg": float(np.mean(run_results["ndcgs"])),
        }
        print(f"    Hit Rate: {sensitivity_results[k]['hit_rate']:.4f}, "
              f"NDCG: {sensitivity_results[k]['ndcg']:.4f}")

    results["sensitivity"] = sensitivity_results

    # =========================================================================
    # Experiment 5: Efficiency Analysis
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXPERIMENT 5: Efficiency Analysis")
    print('='*60, flush=True)

    test_users = list(range(min(500, train.n_users)))
    cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
    cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cluster_manager)

    # Warm up
    for uid in test_users[:100]:
        recs = recommender.recommend(uid, n=20)
        request = RecommendationRequest(user_id=uid, n_recommendations=20)
        cache_manager.put(request, recs)

    efficiency_results = {}

    for name, use_cache in [("Cached", True), ("Fresh", False)]:
        start = time.time()
        n_reqs = 1000

        for i in range(n_reqs):
            uid = test_users[i % len(test_users)]
            if use_cache:
                request = RecommendationRequest(user_id=uid, n_recommendations=20)
                cache_result = cache_manager.get(request)
                if not cache_result.hit:
                    recs = recommender.recommend(uid, n=20)
                    cache_manager.put(request, recs)
            else:
                _ = recommender.recommend(uid, n=20)

        elapsed = time.time() - start
        efficiency_results[name] = {
            "requests": n_reqs,
            "duration_s": elapsed,
            "throughput": n_reqs / elapsed,
        }
        print(f"  {name}: {efficiency_results[name]['throughput']:.0f} req/s")

    results["efficiency"] = efficiency_results

    # Statistical significance
    print(f"\n{'='*60}")
    print("STATISTICAL TESTS")
    print('='*60, flush=True)

    lru_hr = baseline_results["LRU (no clustering)"]["raw_hit_rates"]
    rc_hr = baseline_results["RecCache (Full)"]["raw_hit_rates"]
    t_stat, p_value = stats.ttest_rel(rc_hr, lru_hr)

    print(f"  RecCache vs LRU (paired t-test):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.6f}")
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"    Significance: {sig if sig else 'not significant'}")

    results["statistics"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }

    return results


def create_figures(all_results: dict):
    """Generate publication-quality figures."""
    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print('='*60, flush=True)

    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    datasets_with_results = [d for d in all_results if "baselines" in all_results[d]]

    # Figure 1: Hit Rate Comparison across all datasets
    n_datasets = len(datasets_with_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 4))
    if n_datasets == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets_with_results):
        ax = axes[idx]
        baselines = all_results[dataset]["baselines"]

        methods = ["LRU (no clustering)", "LFU (no clustering)", "LRU + Clustering", "RecCache (Full)"]
        methods = [m for m in methods if m in baselines]
        hit_rates = [baselines[m]["hit_rate_mean"] for m in methods]
        errors = [baselines[m]["hit_rate_std"] for m in methods]

        short_labels = []
        for m in methods:
            if "LRU" in m and "Cluster" not in m:
                short_labels.append("LRU")
            elif "LFU" in m:
                short_labels.append("LFU")
            elif "Clustering" in m:
                short_labels.append("LRU+Cluster")
            else:
                short_labels.append("RecCache")

        colors = ['#7f7f7f', '#7f7f7f', '#1f77b4', '#2ca02c'][:len(methods)]
        x = np.arange(len(methods))

        ax.bar(x, hit_rates, yerr=errors, capsize=4, color=colors,
               edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8, rotation=15)
        ax.set_ylabel('Cache Hit Rate', fontsize=10)
        ax.set_title(dataset.upper(), fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)

        if "oracle" in all_results[dataset]:
            oracle_hr = all_results[dataset]["oracle"]["cluster_level_optimal"].get(5000, 0)
            if oracle_hr > 0:
                ax.axhline(y=oracle_hr, color='red', linestyle='--', linewidth=1.5,
                          label=f'Oracle ({oracle_hr:.2f})')
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(fig_dir / "hit_rate_comparison.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(fig_dir / "hit_rate_comparison.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: hit_rate_comparison.pdf")

    # Figure 2: Ablation Study (use first dataset with ablation data)
    for dataset in datasets_with_results:
        if "ablation" in all_results[dataset]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ablation = all_results[dataset]["ablation"]
            components = list(ablation.keys())
            hit_rates = [ablation[c]["hit_rate_mean"] for c in components]

            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(components)))
            bars = ax.barh(range(len(components)), hit_rates, color=colors,
                          edgecolor='black', linewidth=0.5)

            ax.set_yticks(range(len(components)))
            ax.set_yticklabels(components, fontsize=10)
            ax.set_xlabel('Cache Hit Rate', fontsize=11)
            ax.set_title(f'Ablation Study ({dataset.upper()})', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.0)

            for bar, hr in zip(bars, hit_rates):
                ax.text(hr + 0.01, bar.get_y() + bar.get_height() / 2, f'{hr:.3f}',
                       va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(fig_dir / "ablation_study.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(fig_dir / "ablation_study.png", bbox_inches='tight', dpi=150)
            plt.close()
            print("  Saved: ablation_study.pdf")
            break

    # Figure 3: Cluster Sensitivity (overlay all datasets)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = ['o-', 's--', '^-.', 'D:']
    colors_ds = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    has_data = False

    for i, dataset in enumerate(datasets_with_results):
        if "sensitivity" in all_results[dataset]:
            sensitivity = all_results[dataset]["sensitivity"]
            k_values = sorted([int(k) for k in sensitivity.keys()])
            hit_rates = [sensitivity[k]["hit_rate"] for k in k_values]

            ax.plot(k_values, hit_rates, markers[i % len(markers)],
                   linewidth=2, markersize=7, color=colors_ds[i % len(colors_ds)],
                   label=dataset.upper())
            has_data = True

    if has_data:
        ax.set_xlabel('Number of Clusters (K)', fontsize=11)
        ax.set_ylabel('Cache Hit Rate', fontsize=11)
        ax.set_xscale('log')
        ax.set_title('Cluster Count Sensitivity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "cluster_sensitivity.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(fig_dir / "cluster_sensitivity.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: cluster_sensitivity.pdf")

    print(f"\nAll figures saved to {fig_dir}/")


def print_summary(all_results: dict):
    """Print final summary for paper."""
    print(f"\n{'='*70}")
    print("FINAL SUMMARY FOR PAPER")
    print('='*70, flush=True)

    # Summary table
    print(f"\n{'Dataset':<18} {'LRU HR':>10} {'RecCache HR':>12} {'Improv.':>10} "
          f"{'p-value':>10} {'Speedup':>10}")
    print("-" * 72)

    for dataset, results in all_results.items():
        baselines = results.get("baselines", {})
        if not baselines:
            continue

        lru_hr = baselines.get("LRU (no clustering)", {}).get("hit_rate_mean", 0)
        rc_hr = baselines.get("RecCache (Full)", {}).get("hit_rate_mean", 0)
        improvement = (rc_hr - lru_hr) / lru_hr * 100 if lru_hr > 0 else 0

        p_val = results.get("statistics", {}).get("p_value", 1.0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        cached_tp = results.get("efficiency", {}).get("Cached", {}).get("throughput", 0)
        fresh_tp = results.get("efficiency", {}).get("Fresh", {}).get("throughput", 1)
        speedup = cached_tp / fresh_tp if fresh_tp > 0 else 0

        print(f"{dataset:<18} {lru_hr:>10.4f} {rc_hr:>12.4f} {improvement:>+9.1f}% "
              f"{p_val:>9.4f}{sig:>2} {speedup:>9.1f}x")

    # Detailed per-dataset
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        print("-" * 50)

        stats_info = results.get("stats", {})
        print(f"  Users: {stats_info.get('n_users', '?')}, "
              f"Items: {stats_info.get('n_items', '?')}, "
              f"Interactions: {stats_info.get('n_interactions', '?')}, "
              f"Clusters: {stats_info.get('n_clusters', '?')}")

        baselines = results.get("baselines", {})
        if baselines:
            for name in ["LRU (no clustering)", "LFU (no clustering)", "FIFO (no clustering)",
                         "LRU + Clustering", "RecCache (Full)"]:
                if name in baselines:
                    b = baselines[name]
                    print(f"  {name:<25} HR={b['hit_rate_mean']:.4f}±{b['hit_rate_std']:.4f}  "
                          f"NDCG={b['ndcg_mean']:.4f}")

        if "oracle" in results:
            oracle = results["oracle"]["cluster_level_optimal"].get(5000, 0)
            if oracle > 0:
                rc_hr = baselines.get("RecCache (Full)", {}).get("hit_rate_mean", 0)
                print(f"  Oracle OPT (cache=5000): {oracle:.4f}  "
                      f"(%opt={rc_hr/oracle*100:.1f}%)")


def main():
    print("=" * 70)
    print("RecCache Complete Experiment Suite (4 Datasets)")
    print("=" * 70, flush=True)

    all_results = {}

    for dataset in ["ml-100k", "ml-1m", "amazon-movies", "mind-small"]:
        try:
            results = run_single_dataset_experiments(dataset, n_runs=5)
            all_results[dataset] = results
        except Exception as e:
            print(f"\nERROR on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Generate figures
    if all_results:
        create_figures(all_results)

    # Print summary
    print_summary(all_results)

    # Save results
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_path = Path("results/complete_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("Experiments complete!")


if __name__ == "__main__":
    main()
