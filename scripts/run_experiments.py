#!/usr/bin/env python3
"""
RecCache Complete Experiment Suite

Runs all experiments for top-tier venue submission:
1. Multi-dataset evaluation (ML-100K, ML-1M, Amazon-like, Yelp-like)
2. Recommender baselines comparison
3. Cache strategy baselines comparison
4. Ablation study
5. Parameter sensitivity analysis
6. User group analysis
7. Statistical significance testing

Output: JSON results + formatted tables
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats
from tqdm import tqdm

from reccache.utils.data_loader import DataLoader, InteractionData
from reccache.utils.config import CacheConfig, ClusterConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.models.baselines import (
    create_recommender,
    MostPopularRecommender,
    BPRMF,
    ItemKNNRecommender,
)
from reccache.cache.baselines import create_cache
from reccache.cache.manager import CacheManager, CacheAwareRecommender, RecommendationRequest
from reccache.clustering.user_cluster import UserClusterManager
from reccache.evaluation.metrics import RecommendationMetrics
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig
from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker


# ============================================================================
# Configuration
# ============================================================================

DATASETS = ["ml-100k", "ml-1m", "amazon-beauty", "mind-small"]
N_RUNS = 5
N_REQUESTS = 5000
EMBEDDING_DIM = 64
K = 20  # Top-K for evaluation


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass


def build_ground_truth(test_data: InteractionData, min_rating: float = 4.0) -> Dict[int, set]:
    """Build ground truth from test data."""
    ground_truth = defaultdict(set)
    for user_id, item_id, rating in zip(test_data.user_ids, test_data.item_ids, test_data.ratings):
        if rating >= min_rating:
            ground_truth[int(user_id)].add(int(item_id))
    return dict(ground_truth)


def paired_ttest(a: List[float], b: List[float], alpha: float = 0.05) -> Tuple[float, bool, str]:
    """Paired t-test with significance marker."""
    if len(a) < 2 or len(b) < 2:
        return 1.0, False, ""
    t_stat, p = stats.ttest_rel(a, b)
    sig = p < alpha
    marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return p, sig, marker


def format_mean_std(values: List[float]) -> str:
    """Format mean ± std."""
    return f"{np.mean(values):.4f}±{np.std(values):.4f}"


# ============================================================================
# Experiment 1: Recommender Baselines
# ============================================================================

def run_recommender_baselines(
    train_data: InteractionData,
    test_data: InteractionData,
    dataset_name: str,
) -> Dict[str, Dict]:
    """Compare recommender models."""
    print(f"\n{'='*60}")
    print(f"RECOMMENDER BASELINES - {dataset_name}")
    print('='*60)

    ground_truth = build_ground_truth(test_data)
    test_users = list(ground_truth.keys())[:500]  # Sample for speed

    models = {
        "Pop": "pop",
        "ItemKNN": "itemknn",
        "BPR-MF": "bpr",
        "MF": "mf",
    }

    results = {}

    for name, model_type in models.items():
        print(f"\n  Training {name}...")
        run_results = []

        for run in range(N_RUNS):
            set_seed(42 + run)

            recommender = create_recommender(
                model_type,
                n_users=train_data.n_users,
                n_items=train_data.n_items,
                embedding_dim=EMBEDDING_DIM,
            )

            epochs = 10 if model_type in ["mf", "bpr", "ncf"] else 1
            recommender.fit(
                train_data.user_ids,
                train_data.item_ids,
                train_data.ratings,
                epochs=epochs,
                verbose=False,
            )

            # Evaluate
            recommendations = {}
            for user_id in test_users:
                recommendations[user_id] = recommender.recommend(user_id, n=K)

            metrics = RecommendationMetrics.evaluate_recommendations(
                recommendations, ground_truth, k=K
            )

            run_results.append({
                "ndcg": metrics.get("ndcg@k", 0),
                "recall": metrics.get("recall@k", 0),
                "hit_rate": metrics.get("hit_rate", 0),
            })

        results[name] = {
            "runs": run_results,
            "ndcg_mean": np.mean([r["ndcg"] for r in run_results]),
            "ndcg_std": np.std([r["ndcg"] for r in run_results]),
            "recall_mean": np.mean([r["recall"] for r in run_results]),
            "hit_rate_mean": np.mean([r["hit_rate"] for r in run_results]),
        }

        print(f"    NDCG@{K}: {format_mean_std([r['ndcg'] for r in run_results])}")

    return results


# ============================================================================
# Experiment 2: Cache Baselines
# ============================================================================

def run_cache_baselines(
    recommender: MatrixFactorizationRecommender,
    train_data: InteractionData,
    test_data: InteractionData,
    cluster_manager: UserClusterManager,
    dataset_name: str,
    item_embeddings: np.ndarray = None,
) -> Dict[str, Dict]:
    """Compare cache strategies."""
    print(f"\n{'='*60}")
    print(f"CACHE STRATEGY BASELINES - {dataset_name}")
    print('='*60)

    ground_truth = build_ground_truth(test_data)

    # Note: LRU/LFU/FIFO/Random all use clustering (same as RecCache)
    # to ensure fair comparison of cache eviction strategies
    # For no-clustering baseline, see ablation study
    strategies = {
        "LRU": "lru",
        "LFU": "lfu",
        "FIFO": "fifo",
        "Random": "random",
        "RecCache": "reccache",  # Our method (quality-aware eviction + reranker)
    }

    results = {}

    for name, strategy in strategies.items():
        print(f"\n  Testing {name}...")
        run_results = []

        for run in range(N_RUNS):
            set_seed(42 + run)

            cache_config = CacheConfig(
                local_cache_size=5000,
                use_redis_cache=False,
                quality_threshold=0.15,  # Quality threshold for cache decisions
            )

            cache_manager = CacheManager(
                cache_config=cache_config,
                cluster_manager=cluster_manager,
            )

            # Configure RecCache with full quality-aware components
            reranker = None
            if strategy == "reccache":
                # 1. Setup Quality Predictor
                quality_predictor = QualityPredictor(
                    hidden_dim=32,
                    quality_threshold=0.15,
                )
                cache_manager.set_quality_predictor(quality_predictor)

                # 2. Setup Reranker for personalization
                if item_embeddings is not None:
                    reranker = LightweightReranker(
                        history_weight=0.3,
                        recency_weight=0.3,
                        diversity_weight=0.2,
                    )
                    reranker.set_item_embeddings(item_embeddings)

                    # Initialize user history from training data
                    user_history = defaultdict(list)
                    for uid, iid in zip(train_data.user_ids, train_data.item_ids):
                        user_history[int(uid)].append(int(iid))
                    for uid, history in user_history.items():
                        reranker.set_user_history(uid, history[-20:])
            else:
                # Override cache for baselines (non-RecCache)
                cache_manager.local_cache = create_cache(strategy, max_size=5000)

            sim_config = SimulationConfig(
                n_requests=N_REQUESTS,
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
                n_users=train_data.n_users,
                n_items=train_data.n_items,
                ground_truth=ground_truth,
                verbose=False,
            )

            run_results.append({
                "hit_rate": result.hit_rate,
                "ndcg": result.avg_ndcg,
                "latency_ms": result.avg_latency_ms,
                "ndcg_loss": result.ndcg_degradation,
            })

        results[name] = {
            "runs": run_results,
            "hit_rate_mean": np.mean([r["hit_rate"] for r in run_results]),
            "hit_rate_std": np.std([r["hit_rate"] for r in run_results]),
            "ndcg_mean": np.mean([r["ndcg"] for r in run_results]),
            "latency_mean": np.mean([r["latency_ms"] for r in run_results]),
            "ndcg_loss_mean": np.mean([r["ndcg_loss"] for r in run_results]),
        }

        print(f"    Hit Rate: {format_mean_std([r['hit_rate'] for r in run_results])}")
        print(f"    NDCG: {format_mean_std([r['ndcg'] for r in run_results])}")

    return results


# ============================================================================
# Experiment 3: Ablation Study
# ============================================================================

def run_ablation_study(
    recommender: MatrixFactorizationRecommender,
    train_data: InteractionData,
    test_data: InteractionData,
    cluster_manager: UserClusterManager,
    dataset_name: str,
    item_embeddings: np.ndarray = None,
) -> Dict[str, Dict]:
    """Ablation study on RecCache components."""
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY - {dataset_name}")
    print('='*60)

    ground_truth = build_ground_truth(test_data)

    # Prepare user history for reranker
    user_history = defaultdict(list)
    for uid, iid in zip(train_data.user_ids, train_data.item_ids):
        user_history[int(uid)].append(int(iid))

    # Ablation configurations - each removes a component
    ablations = {
        "Full Model": {
            "use_clustering": True,
            "use_quality_predictor": True,
            "use_reranker": True,
        },
        "w/o Quality Predictor": {
            "use_clustering": True,
            "use_quality_predictor": False,
            "use_reranker": True,
        },
        "w/o Reranker": {
            "use_clustering": True,
            "use_quality_predictor": True,
            "use_reranker": False,
        },
        "w/o Clustering": {
            "use_clustering": False,
            "use_quality_predictor": True,
            "use_reranker": True,
        },
        "w/o Quality Predictor & Reranker": {
            "use_clustering": True,
            "use_quality_predictor": False,
            "use_reranker": False,
        },
        "Plain LRU": {
            "use_clustering": False,
            "use_quality_predictor": False,
            "use_reranker": False,
        },
    }

    results = {}

    for name, config in ablations.items():
        print(f"\n  Testing {name}...")
        run_results = []

        for run in range(N_RUNS):
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

            # Setup Quality Predictor if enabled
            if config["use_quality_predictor"]:
                quality_predictor = QualityPredictor(
                    hidden_dim=32,
                    quality_threshold=0.15,
                )
                cache_manager.set_quality_predictor(quality_predictor)

            # Setup Reranker if enabled
            reranker = None
            if config["use_reranker"] and item_embeddings is not None:
                reranker = LightweightReranker(
                    history_weight=0.3,
                    recency_weight=0.3,
                    diversity_weight=0.2,
                )
                reranker.set_item_embeddings(item_embeddings)
                for uid, history in user_history.items():
                    reranker.set_user_history(uid, history[-20:])

            sim_config = SimulationConfig(
                n_requests=N_REQUESTS,
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
                n_users=train_data.n_users,
                n_items=train_data.n_items,
                ground_truth=ground_truth,
                verbose=False,
            )

            run_results.append({
                "hit_rate": result.hit_rate,
                "ndcg": result.avg_ndcg,
                "ndcg_loss": result.ndcg_degradation,
            })

        results[name] = {
            "runs": run_results,
            "hit_rate_mean": np.mean([r["hit_rate"] for r in run_results]),
            "ndcg_mean": np.mean([r["ndcg"] for r in run_results]),
            "ndcg_loss_mean": np.mean([r["ndcg_loss"] for r in run_results]),
        }

        print(f"    Hit Rate: {results[name]['hit_rate_mean']:.4f}, NDCG: {results[name]['ndcg_mean']:.4f}")

    return results


# ============================================================================
# Experiment 4: Parameter Sensitivity
# ============================================================================

def run_parameter_sensitivity(
    recommender: MatrixFactorizationRecommender,
    train_data: InteractionData,
    test_data: InteractionData,
    item_embeddings: np.ndarray,
    dataset_name: str,
) -> Dict[str, Dict]:
    """Parameter sensitivity analysis."""
    print(f"\n{'='*60}")
    print(f"PARAMETER SENSITIVITY - {dataset_name}")
    print('='*60)

    ground_truth = build_ground_truth(test_data)
    results = {}

    # 1. Number of clusters
    print("\n  Testing n_clusters...")
    n_clusters_values = [10, 25, 50, 100, 200]
    cluster_results = {}

    for n_clusters in n_clusters_values:
        run_results = []

        for run in range(3):  # Fewer runs for speed
            set_seed(42 + run)

            cm = UserClusterManager(
                n_clusters=n_clusters,
                embedding_dim=item_embeddings.shape[1],
                n_items=len(item_embeddings),
            )
            cm.set_item_embeddings(item_embeddings)
            cm.initialize_from_interactions(
                train_data.user_ids, train_data.item_ids, train_data.ratings
            )

            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

            sim_config = SimulationConfig(n_requests=3000, n_warmup_requests=300, eval_sample_rate=0.1)
            simulator = OnlineSimulator(
                recommender=recommender,
                cache_manager=cache_manager,
                cluster_manager=cm,
                config=sim_config,
            )

            result = simulator.run_simulation(
                n_users=train_data.n_users, n_items=train_data.n_items,
                ground_truth=ground_truth, verbose=False
            )

            run_results.append({"hit_rate": result.hit_rate, "ndcg": result.avg_ndcg})

        cluster_results[n_clusters] = {
            "hit_rate": np.mean([r["hit_rate"] for r in run_results]),
            "ndcg": np.mean([r["ndcg"] for r in run_results]),
        }
        print(f"    K={n_clusters}: hit_rate={cluster_results[n_clusters]['hit_rate']:.4f}")

    results["n_clusters"] = cluster_results

    # 2. Cache size
    print("\n  Testing cache_size...")
    cache_sizes = [500, 1000, 2500, 5000, 10000]
    size_results = {}

    cm = UserClusterManager(n_clusters=50, embedding_dim=item_embeddings.shape[1], n_items=len(item_embeddings))
    cm.set_item_embeddings(item_embeddings)
    cm.initialize_from_interactions(train_data.user_ids, train_data.item_ids, train_data.ratings)

    for cache_size in cache_sizes:
        run_results = []

        for run in range(3):
            set_seed(42 + run)

            cache_config = CacheConfig(local_cache_size=cache_size, use_redis_cache=False)
            cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

            sim_config = SimulationConfig(n_requests=3000, n_warmup_requests=300, eval_sample_rate=0.1)
            simulator = OnlineSimulator(
                recommender=recommender, cache_manager=cache_manager, cluster_manager=cm, config=sim_config
            )

            result = simulator.run_simulation(
                n_users=train_data.n_users, n_items=train_data.n_items,
                ground_truth=ground_truth, verbose=False
            )

            run_results.append({"hit_rate": result.hit_rate, "ndcg": result.avg_ndcg})

        size_results[cache_size] = {
            "hit_rate": np.mean([r["hit_rate"] for r in run_results]),
            "ndcg": np.mean([r["ndcg"] for r in run_results]),
        }
        print(f"    Size={cache_size}: hit_rate={size_results[cache_size]['hit_rate']:.4f}")

    results["cache_size"] = size_results

    return results


# ============================================================================
# Experiment 5: User Group Analysis
# ============================================================================

def run_user_group_analysis(
    recommender: MatrixFactorizationRecommender,
    train_data: InteractionData,
    test_data: InteractionData,
    cache_manager: CacheManager,
    dataset_name: str,
) -> Dict[str, Dict]:
    """Analyze performance across user activity groups."""
    print(f"\n{'='*60}")
    print(f"USER GROUP ANALYSIS - {dataset_name}")
    print('='*60)

    ground_truth = build_ground_truth(test_data)

    # Count user interactions
    user_counts = defaultdict(int)
    for user_id in train_data.user_ids:
        user_counts[int(user_id)] += 1

    # Group users
    groups = {
        "Cold (<10)": [],
        "Sparse (10-50)": [],
        "Normal (50-200)": [],
        "Active (>200)": [],
    }

    for user_id, count in user_counts.items():
        if user_id not in ground_truth:
            continue
        if count < 10:
            groups["Cold (<10)"].append(user_id)
        elif count < 50:
            groups["Sparse (10-50)"].append(user_id)
        elif count < 200:
            groups["Normal (50-200)"].append(user_id)
        else:
            groups["Active (>200)"].append(user_id)

    # Evaluate per group
    cached_rec = CacheAwareRecommender(recommender=recommender, cache_manager=cache_manager)

    results = {}
    for group_name, user_ids in groups.items():
        if not user_ids:
            continue

        sample_users = user_ids[:min(100, len(user_ids))]

        recommendations = {}
        for user_id in sample_users:
            if user_id in ground_truth:
                recs, _ = cached_rec.recommend(user_id, n=K)
                recommendations[user_id] = recs

        if recommendations:
            group_gt = {u: ground_truth[u] for u in recommendations.keys()}
            metrics = RecommendationMetrics.evaluate_recommendations(recommendations, group_gt, k=K)

            results[group_name] = {
                "n_users": len(sample_users),
                "ndcg": metrics.get("ndcg@k", 0),
                "recall": metrics.get("recall@k", 0),
                "hit_rate": metrics.get("hit_rate", 0),
            }

            print(f"  {group_name}: {len(sample_users)} users, NDCG={results[group_name]['ndcg']:.4f}")

    return results


# ============================================================================
# Experiment 6: Efficiency Analysis
# ============================================================================

def run_efficiency_analysis(
    recommender: MatrixFactorizationRecommender,
    train_data: InteractionData,
    cluster_manager: UserClusterManager,
    item_embeddings: np.ndarray,
    dataset_name: str,
) -> Dict[str, Any]:
    """Analyze latency breakdown and throughput."""
    print(f"\n{'='*60}")
    print(f"EFFICIENCY ANALYSIS - {dataset_name}")
    print('='*60)

    results = {}

    # Prepare user history for reranker
    user_history = defaultdict(list)
    for uid, iid in zip(train_data.user_ids, train_data.item_ids):
        user_history[int(uid)].append(int(iid))

    # Test users
    test_users = list(range(min(500, train_data.n_users)))

    # 1. Latency breakdown for each component
    print("\n  Measuring component latencies...")
    latency_breakdown = {
        "embedding_lookup": [],
        "cluster_assignment": [],
        "cache_lookup": [],
        "quality_prediction": [],
        "reranking": [],
        "fresh_recommendation": [],
        "total_cached": [],
        "total_fresh": [],
    }

    # Setup components
    cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False, quality_threshold=0.15)
    cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cluster_manager)

    quality_predictor = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
    cache_manager.set_quality_predictor(quality_predictor)

    reranker = LightweightReranker(history_weight=0.3, recency_weight=0.3, diversity_weight=0.2)
    reranker.set_item_embeddings(item_embeddings)
    for uid, history in user_history.items():
        reranker.set_user_history(uid, history[-20:])

    # Warm up cache
    for user_id in test_users[:100]:
        recs = recommender.recommend(user_id, n=20)
        request = RecommendationRequest(user_id=user_id, n_recommendations=20)
        cache_manager.put(request, recs)

    # Measure each component
    for user_id in test_users:
        # Cluster assignment time
        start = time.time()
        cluster_info = cluster_manager.get_user_cluster(user_id)
        latency_breakdown["cluster_assignment"].append((time.time() - start) * 1000)

        # Cache lookup time
        request = RecommendationRequest(user_id=user_id, n_recommendations=20)
        start = time.time()
        cache_result = cache_manager.get(request)
        latency_breakdown["cache_lookup"].append((time.time() - start) * 1000)

        # Quality prediction time (if we had cluster info)
        start = time.time()
        _ = quality_predictor.predict(
            distance_to_center=cluster_info.distance_to_center,
            cluster_size=cluster_info.cluster_size,
        )
        latency_breakdown["quality_prediction"].append((time.time() - start) * 1000)

        # Reranking time
        if cache_result.hit and cache_result.value:
            start = time.time()
            _ = reranker.rerank(user_id, cache_result.value)
            latency_breakdown["reranking"].append((time.time() - start) * 1000)
        else:
            latency_breakdown["reranking"].append(0.0)

        # Fresh recommendation time
        start = time.time()
        _ = recommender.recommend(user_id, n=20)
        latency_breakdown["fresh_recommendation"].append((time.time() - start) * 1000)

    # Compute aggregated latencies
    results["latency_breakdown"] = {
        component: {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }
        for component, times in latency_breakdown.items()
        if times
    }

    # Total latency comparison
    cached_total = (
        np.mean(latency_breakdown["cluster_assignment"]) +
        np.mean(latency_breakdown["cache_lookup"]) +
        np.mean(latency_breakdown["quality_prediction"]) +
        np.mean(latency_breakdown["reranking"])
    )
    fresh_total = np.mean(latency_breakdown["fresh_recommendation"])

    results["latency_comparison"] = {
        "cached_mean_ms": cached_total,
        "fresh_mean_ms": fresh_total,
        "speedup": fresh_total / cached_total if cached_total > 0 else 0,
    }

    print(f"    Cached path: {cached_total:.2f}ms")
    print(f"    Fresh path: {fresh_total:.2f}ms")
    print(f"    Speedup: {fresh_total / cached_total:.1f}x" if cached_total > 0 else "    Speedup: N/A")

    # 2. Throughput test
    print("\n  Measuring throughput...")
    throughput_results = {}

    for strategy_name, use_cache in [("cached", True), ("fresh", False)]:
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
        throughput_results[strategy_name] = {
            "requests": n_requests,
            "duration_s": elapsed,
            "qps": n_requests / elapsed,
        }
        print(f"    {strategy_name}: {n_requests / elapsed:.1f} req/s")

    results["throughput"] = throughput_results

    # 3. Memory usage estimate
    import sys
    cache_memory = sys.getsizeof(cache_manager.local_cache._cache) if hasattr(cache_manager.local_cache, '_cache') else 0
    cluster_memory = sys.getsizeof(cluster_manager.cluster_centers) * cluster_manager.n_clusters if hasattr(cluster_manager, 'cluster_centers') else 0

    results["memory"] = {
        "cache_bytes_estimate": cache_memory,
        "cluster_bytes_estimate": cluster_memory,
    }

    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_all_experiments():
    """Run complete experiment suite."""
    print("="*70)
    print("RecCache Complete Experiment Suite")
    print("="*70)
    print(f"Datasets: {DATASETS}")
    print(f"N_RUNS: {N_RUNS}")
    print(f"N_REQUESTS: {N_REQUESTS}")

    loader = DataLoader("data")
    all_results = {}

    for dataset_name in DATASETS:
        print(f"\n\n{'#'*70}")
        print(f"# Dataset: {dataset_name.upper()}")
        print('#'*70)

        # Load data
        print(f"\nLoading {dataset_name}...")
        try:
            train, val, test = loader.load_dataset(dataset_name)
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            continue

        stats = train.get_statistics()
        print(f"  Users: {stats['n_users']}, Items: {stats['n_items']}, Interactions: {stats['n_interactions']}")

        dataset_results = {"stats": stats}

        # Train base recommender
        print("\nTraining base MF recommender...")
        base_rec = MatrixFactorizationRecommender(
            n_users=train.n_users,
            n_items=train.n_items,
            embedding_dim=EMBEDDING_DIM,
        )
        base_rec.fit(train.user_ids, train.item_ids, train.ratings, epochs=15, verbose=True)

        # Setup clustering
        print("\nSetting up user clustering...")
        item_embeddings = base_rec.get_all_item_embeddings()
        cluster_manager = UserClusterManager(
            n_clusters=50,
            embedding_dim=item_embeddings.shape[1],
            n_items=len(item_embeddings),
        )
        cluster_manager.set_item_embeddings(item_embeddings)
        cluster_manager.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

        # Run experiments
        dataset_results["recommender_baselines"] = run_recommender_baselines(train, test, dataset_name)
        dataset_results["cache_baselines"] = run_cache_baselines(
            base_rec, train, test, cluster_manager, dataset_name, item_embeddings
        )
        dataset_results["ablation"] = run_ablation_study(
            base_rec, train, test, cluster_manager, dataset_name, item_embeddings
        )
        dataset_results["sensitivity"] = run_parameter_sensitivity(base_rec, train, test, item_embeddings, dataset_name)

        # User group analysis
        cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
        cache_mgr = CacheManager(cache_config=cache_config, cluster_manager=cluster_manager)
        dataset_results["user_groups"] = run_user_group_analysis(base_rec, train, test, cache_mgr, dataset_name)

        # Efficiency analysis
        dataset_results["efficiency"] = run_efficiency_analysis(
            base_rec, train, cluster_manager, item_embeddings, dataset_name
        )

        all_results[dataset_name] = dataset_results

    return all_results


def print_final_tables(results: Dict):
    """Print publication-ready tables."""
    print("\n\n" + "="*80)
    print("FINAL RESULTS TABLES")
    print("="*80)

    # Table 1: Recommender Baselines
    print("\n\nTable 1: Recommender Model Comparison (NDCG@20)")
    print("-"*80)
    print(f"{'Dataset':<15}", end="")
    models = ["Pop", "ItemKNN", "BPR-MF", "MF"]
    for m in models:
        print(f"{m:>15}", end="")
    print()
    print("-"*80)

    for dataset, data in results.items():
        if "recommender_baselines" not in data:
            continue
        print(f"{dataset:<15}", end="")
        for model in models:
            if model in data["recommender_baselines"]:
                val = data["recommender_baselines"][model]["ndcg_mean"]
                std = data["recommender_baselines"][model]["ndcg_std"]
                print(f"{val:.4f}±{std:.4f}", end=" ")
            else:
                print(f"{'N/A':>14}", end=" ")
        print()

    # Table 2: Cache Strategies
    print("\n\nTable 2: Cache Strategy Comparison")
    print("-"*80)
    print(f"{'Dataset':<12}{'Strategy':<12}{'HitRate':>12}{'NDCG':>12}{'Latency(ms)':>14}{'NDCG Loss':>12}")
    print("-"*80)

    for dataset, data in results.items():
        if "cache_baselines" not in data:
            continue
        for strategy, metrics in data["cache_baselines"].items():
            print(f"{dataset:<12}{strategy:<12}"
                  f"{metrics['hit_rate_mean']:>12.4f}"
                  f"{metrics['ndcg_mean']:>12.4f}"
                  f"{metrics['latency_mean']:>14.2f}"
                  f"{metrics['ndcg_loss_mean']:>12.4f}")
        print("-"*80)

    # Table 3: Ablation Study
    print("\n\nTable 3: Ablation Study")
    print("-"*70)
    print(f"{'Dataset':<12}{'Variant':<25}{'HitRate':>12}{'NDCG':>12}{'Change':>10}")
    print("-"*70)

    for dataset, data in results.items():
        if "ablation" not in data:
            continue
        full_ndcg = data["ablation"].get("Full Model", {}).get("ndcg_mean", 0)
        for variant, metrics in data["ablation"].items():
            change = ((metrics["ndcg_mean"] - full_ndcg) / full_ndcg * 100) if full_ndcg > 0 else 0
            change_str = f"{change:+.2f}%" if variant != "Full Model" else "-"
            print(f"{dataset:<12}{variant:<25}"
                  f"{metrics['hit_rate_mean']:>12.4f}"
                  f"{metrics['ndcg_mean']:>12.4f}"
                  f"{change_str:>10}")
        print("-"*70)

    # Statistical Significance
    print("\n\nStatistical Significance (RecCache vs LRU)")
    print("-"*60)

    for dataset, data in results.items():
        if "cache_baselines" not in data:
            continue
        if "RecCache" in data["cache_baselines"] and "LRU" in data["cache_baselines"]:
            rc_runs = data["cache_baselines"]["RecCache"]["runs"]
            lru_runs = data["cache_baselines"]["LRU"]["runs"]

            hr_p, hr_sig, hr_mark = paired_ttest(
                [r["hit_rate"] for r in rc_runs],
                [r["hit_rate"] for r in lru_runs]
            )
            ndcg_p, ndcg_sig, ndcg_mark = paired_ttest(
                [r["ndcg"] for r in rc_runs],
                [r["ndcg"] for r in lru_runs]
            )

            print(f"{dataset}: HitRate p={hr_p:.4f}{hr_mark}, NDCG p={ndcg_p:.4f}{ndcg_mark}")

    print("\n" + "="*80)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")


def save_results(results: Dict, output_dir: str = "results"):
    """Save results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float_)):
            return float(obj)
        return obj

    with open(output_path / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path / 'experiment_results.json'}")


if __name__ == "__main__":
    start_time = time.time()

    results = run_all_experiments()
    print_final_tables(results)
    save_results(results)

    elapsed = time.time() - start_time
    print(f"\n\nTotal experiment time: {elapsed/60:.1f} minutes")
    print("Experiment complete!")
