#!/usr/bin/env python3
"""
RecCache Full Experiment Suite

Comprehensive experiments for top-tier venue submission:
1. Multi-dataset evaluation
2. Recommender model baselines comparison
3. Cache strategy baselines comparison
4. Statistical significance testing
5. Ablation study
6. Parameter sensitivity analysis
7. User group analysis

Usage:
    python scripts/run_full_experiments.py --datasets ml-100k ml-1m --n-runs 5
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from tqdm import tqdm

from reccache.utils.data_loader import DataLoader, InteractionData
from reccache.utils.config import Config, CacheConfig, ClusterConfig
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.models.baselines import (
    create_recommender,
    MostPopularRecommender,
    BPRMF,
    LightGCNRecommender,
    ItemKNNRecommender,
)
from reccache.cache.baselines import (
    create_cache,
    LRUCache,
    LFUCache,
    RandomCache,
    PopularityCache,
    CacheStrategyComparator,
)
from reccache.cache.manager import CacheManager, CacheAwareRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.evaluation.metrics import RecommendationMetrics, CacheEvaluator
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig
from reccache.evaluation.experiment import (
    ExperimentRunner,
    StatisticalTester,
    AblationStudy,
    ParameterSensitivityAnalysis,
    UserGroupAnalysis,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RecCache Full Experiments")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ml-100k", "ml-1m", "amazon-beauty", "mind-small"],
        help="Datasets to evaluate on",
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Number of runs per experiment")
    parser.add_argument("--n-requests", type=int, default=10000, help="Simulation requests")
    parser.add_argument("--output-dir", type=str, default="results/full_experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-recommender-baselines", action="store_true")
    parser.add_argument("--skip-cache-baselines", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-user-groups", action="store_true")
    return parser.parse_args()


def load_all_datasets(
    dataset_names: List[str],
    data_dir: str = "data"
) -> Dict[str, Tuple[InteractionData, InteractionData, InteractionData]]:
    """Load multiple datasets."""
    loader = DataLoader(data_dir=data_dir)
    datasets = {}

    for name in dataset_names:
        print(f"\nLoading {name}...")
        try:
            train, val, test = loader.load_dataset(name)
            datasets[name] = (train, val, test)
            stats = train.get_statistics()
            print(f"  Users: {stats['n_users']}, Items: {stats['n_items']}, "
                  f"Interactions: {stats['n_interactions']}, Sparsity: {stats['sparsity']:.4f}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    return datasets


def build_ground_truth(test_data: InteractionData, min_rating: float = 4.0) -> Dict[int, set]:
    """Build ground truth from test data."""
    ground_truth = defaultdict(set)
    for user_id, item_id, rating in zip(test_data.user_ids, test_data.item_ids, test_data.ratings):
        if rating >= min_rating:
            ground_truth[int(user_id)].add(int(item_id))
    return dict(ground_truth)


def run_recommender_baselines(
    train_data: InteractionData,
    test_data: InteractionData,
    n_runs: int = 5,
    verbose: bool = True,
) -> Dict:
    """Compare recommender model baselines."""
    ground_truth = build_ground_truth(test_data)
    test_users = list(ground_truth.keys())

    models = ["pop", "itemknn", "bpr", "mf", "lightgcn"]
    results = {}

    runner = ExperimentRunner(n_runs=n_runs)

    for model_name in models:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name.upper()}")
            print('='*50)

        def experiment(seed, model=model_name):
            np.random.seed(seed)

            recommender = create_recommender(
                model,
                n_users=train_data.n_users,
                n_items=train_data.n_items,
                embedding_dim=64,
            )

            recommender.fit(
                train_data.user_ids,
                train_data.item_ids,
                train_data.ratings,
                epochs=10 if model in ["mf", "bpr", "lightgcn", "ncf"] else 1,
                verbose=False,
            )

            # Generate recommendations
            recommendations = {}
            for user_id in test_users[:500]:  # Sample for speed
                recommendations[user_id] = recommender.recommend(user_id, n=20)

            metrics = RecommendationMetrics.evaluate_recommendations(
                recommendations, ground_truth, k=20
            )

            return {
                "ndcg@20": metrics.get("ndcg@k", 0),
                "recall@20": metrics.get("recall@k", 0),
                "hit_rate@20": metrics.get("hit_rate", 0),
            }

        results[model_name] = runner.run_experiment(experiment, model_name, verbose)

    return results


def run_cache_baselines(
    recommender,
    train_data: InteractionData,
    test_data: InteractionData,
    cluster_manager: UserClusterManager,
    n_requests: int = 5000,
    n_runs: int = 5,
    verbose: bool = True,
) -> Dict:
    """Compare cache strategy baselines."""
    ground_truth = build_ground_truth(test_data)

    strategies = ["lru", "lfu", "random", "fifo", "popularity"]
    results = {}

    runner = ExperimentRunner(n_runs=n_runs)

    for strategy in strategies:
        if verbose:
            print(f"\nEvaluating cache strategy: {strategy.upper()}")

        def experiment(seed, strat=strategy):
            np.random.seed(seed)

            cache_config = CacheConfig(
                local_cache_size=5000,
                use_redis_cache=False,
            )

            cache_manager = CacheManager(
                cache_config=cache_config,
                cluster_manager=cluster_manager,
            )

            # Override with baseline strategy
            cache_manager.local_cache = create_cache(strat, max_size=5000)

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
                ground_truth=ground_truth,
                verbose=False,
            )

            return {
                "hit_rate": result.hit_rate,
                "avg_latency_ms": result.avg_latency_ms,
                "ndcg": result.avg_ndcg,
                "ndcg_degradation": result.ndcg_degradation,
            }

        results[strategy] = runner.run_experiment(experiment, strategy, verbose)

    # Add RecCache (our method)
    if verbose:
        print(f"\nEvaluating cache strategy: RECCACHE (ours)")

    def reccache_experiment(seed):
        np.random.seed(seed)

        cache_config = CacheConfig(
            local_cache_size=5000,
            use_redis_cache=False,
            quality_threshold=0.1,
        )

        cache_manager = CacheManager(
            cache_config=cache_config,
            cluster_manager=cluster_manager,
            use_quality_predictor=True,
            use_reranker=True,
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
            ground_truth=ground_truth,
            verbose=False,
        )

        return {
            "hit_rate": result.hit_rate,
            "avg_latency_ms": result.avg_latency_ms,
            "ndcg": result.avg_ndcg,
            "ndcg_degradation": result.ndcg_degradation,
        }

    results["reccache"] = runner.run_experiment(reccache_experiment, "reccache", verbose)

    return results


def run_ablation_study(
    recommender,
    train_data: InteractionData,
    test_data: InteractionData,
    cluster_manager: UserClusterManager,
    n_requests: int = 5000,
    n_runs: int = 5,
    verbose: bool = True,
) -> Dict:
    """Run ablation study on RecCache components."""
    ground_truth = build_ground_truth(test_data)

    def base_experiment(config: Dict) -> Dict[str, float]:
        np.random.seed(config.get("seed", 42))

        cache_config = CacheConfig(
            local_cache_size=config.get("cache_size", 5000),
            use_redis_cache=config.get("use_l2_cache", False),
            quality_threshold=config.get("quality_threshold", 0.1),
        )

        cache_manager = CacheManager(
            cache_config=cache_config,
            cluster_manager=cluster_manager if config.get("use_clustering", True) else None,
            use_quality_predictor=config.get("use_quality_predictor", True),
            use_reranker=config.get("use_reranker", True),
        )

        sim_config = SimulationConfig(
            n_requests=n_requests,
            n_warmup_requests=500,
            eval_sample_rate=0.1,
        )

        simulator = OnlineSimulator(
            recommender=recommender,
            cache_manager=cache_manager,
            cluster_manager=cluster_manager if config.get("use_clustering", True) else None,
            config=sim_config,
        )

        result = simulator.run_simulation(
            n_users=train_data.n_users,
            n_items=train_data.n_items,
            ground_truth=ground_truth,
            verbose=False,
        )

        return {
            "hit_rate": result.hit_rate,
            "avg_latency_ms": result.avg_latency_ms,
            "ndcg": result.avg_ndcg,
            "ndcg_degradation": result.ndcg_degradation,
        }

    base_config = {
        "cache_size": 5000,
        "use_l2_cache": False,
        "use_clustering": True,
        "use_quality_predictor": True,
        "use_reranker": True,
        "quality_threshold": 0.1,
    }

    ablations = {
        "w/o Quality Predictor": {"use_quality_predictor": False},
        "w/o Reranker": {"use_reranker": False},
        "w/o Clustering": {"use_clustering": False},
        "w/o Quality Predictor & Reranker": {
            "use_quality_predictor": False,
            "use_reranker": False,
        },
    }

    study = AblationStudy(base_experiment, base_config, n_runs=n_runs)
    results = study.run_ablation(ablations, verbose=verbose)

    if verbose:
        study.print_ablation_table(
            results,
            metrics=["hit_rate", "ndcg", "ndcg_degradation", "avg_latency_ms"],
        )

    return results


def run_parameter_sensitivity(
    recommender,
    train_data: InteractionData,
    test_data: InteractionData,
    cluster_manager_factory,  # Function to create cluster manager
    n_requests: int = 3000,
    n_runs: int = 3,
    verbose: bool = True,
) -> Dict:
    """Run parameter sensitivity analysis."""
    ground_truth = build_ground_truth(test_data)

    def experiment(config: Dict) -> Dict[str, float]:
        np.random.seed(config.get("seed", 42))

        n_clusters = config.get("n_clusters", 50)
        cache_size = config.get("cache_size", 5000)
        quality_threshold = config.get("quality_threshold", 0.1)

        # Create cluster manager with specified n_clusters
        cm = cluster_manager_factory(n_clusters)

        cache_config = CacheConfig(
            local_cache_size=cache_size,
            use_redis_cache=False,
            quality_threshold=quality_threshold,
        )

        cache_manager = CacheManager(
            cache_config=cache_config,
            cluster_manager=cm,
        )

        sim_config = SimulationConfig(
            n_requests=n_requests,
            n_warmup_requests=300,
            eval_sample_rate=0.1,
        )

        simulator = OnlineSimulator(
            recommender=recommender,
            cache_manager=cache_manager,
            cluster_manager=cm,
            config=sim_config,
        )

        result = simulator.run_simulation(
            n_users=train_data.n_users,
            n_items=train_data.n_items,
            ground_truth=ground_truth,
            verbose=False,
        )

        return {
            "hit_rate": result.hit_rate,
            "ndcg": result.avg_ndcg,
            "ndcg_degradation": result.ndcg_degradation,
        }

    base_config = {
        "n_clusters": 50,
        "cache_size": 5000,
        "quality_threshold": 0.1,
    }

    parameters = {
        "n_clusters": [10, 25, 50, 100, 200],
        "cache_size": [500, 1000, 2500, 5000, 10000],
        "quality_threshold": [0.01, 0.05, 0.1, 0.2, 0.3],
    }

    analyzer = ParameterSensitivityAnalysis(experiment, base_config, n_runs=n_runs)
    all_results = analyzer.analyze_multiple_parameters(parameters, verbose=verbose)

    if verbose:
        for param_name, param_results in all_results.items():
            analyzer.print_sensitivity_table(
                param_results, param_name,
                metrics=["hit_rate", "ndcg", "ndcg_degradation"],
            )

    return all_results


def run_user_group_analysis(
    recommender,
    train_data: InteractionData,
    test_data: InteractionData,
    cache_manager: CacheManager,
    cluster_manager: UserClusterManager,
    verbose: bool = True,
) -> Dict:
    """Analyze performance across user groups."""
    ground_truth = build_ground_truth(test_data)

    # Count user interactions
    user_interactions = defaultdict(int)
    for user_id in train_data.user_ids:
        user_interactions[int(user_id)] += 1

    # Group users
    user_groups = UserGroupAnalysis.group_by_activity(
        user_interactions,
        thresholds=[10, 50, 200],
    )

    if verbose:
        print("\nUser group sizes:")
        for group, users in user_groups.items():
            print(f"  {group}: {len(users)} users")

    # Generate recommendations for all users
    cached_recommender = CacheAwareRecommender(
        recommender=recommender,
        cache_manager=cache_manager,
    )

    recommendations = {}
    for user_id in tqdm(list(ground_truth.keys())[:1000], desc="Generating recommendations"):
        recs, _ = cached_recommender.recommend(user_id, n=20)
        recommendations[user_id] = recs

    # Evaluate per group
    group_results = UserGroupAnalysis.evaluate_per_group(
        recommendations, ground_truth, user_groups, k=20
    )

    if verbose:
        UserGroupAnalysis.print_group_analysis(
            group_results,
            metrics=["ndcg@k", "recall@k", "hit_rate"],
        )

    return group_results


def save_all_results(results: Dict, output_dir: str):
    """Save all experiment results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if hasattr(obj, 'mean'):  # AggregatedResult
            return {"mean": obj.mean, "std": obj.std, "n_runs": obj.n_runs}
        return obj

    # Save JSON
    with open(output_path / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path}")


def print_final_summary(all_results: Dict):
    """Print final experiment summary."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    for dataset, results in all_results.items():
        print(f"\n## Dataset: {dataset}")

        if "recommender_baselines" in results:
            print("\n### Recommender Baselines")
            for model, res in results["recommender_baselines"].items():
                if hasattr(res, 'mean'):
                    ndcg = res.mean.get("ndcg@20", 0)
                    print(f"  {model:15s}: NDCG@20 = {ndcg:.4f}")

        if "cache_baselines" in results:
            print("\n### Cache Strategy Baselines")
            for strategy, res in results["cache_baselines"].items():
                if hasattr(res, 'mean'):
                    hr = res.mean.get("hit_rate", 0)
                    ndcg = res.mean.get("ndcg", 0)
                    print(f"  {strategy:15s}: Hit Rate = {hr:.4f}, NDCG = {ndcg:.4f}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("="*80)
    print("RecCache Full Experiment Suite")
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"N runs: {args.n_runs}")
    print(f"Output: {args.output_dir}")

    # Load datasets
    datasets = load_all_datasets(args.datasets)

    all_results = {}

    for dataset_name, (train_data, val_data, test_data) in datasets.items():
        print(f"\n{'#'*80}")
        print(f"# Dataset: {dataset_name}")
        print('#'*80)

        dataset_results = {}

        # Train base recommender
        print("\nTraining base recommender (MF)...")
        base_recommender = MatrixFactorizationRecommender(
            n_users=train_data.n_users,
            n_items=train_data.n_items,
            embedding_dim=64,
        )
        base_recommender.fit(
            train_data.user_ids,
            train_data.item_ids,
            train_data.ratings,
            epochs=15,
            verbose=True,
        )

        # Setup clustering
        print("\nSetting up user clustering...")
        item_embeddings = base_recommender.get_all_item_embeddings()

        def create_cluster_manager(n_clusters):
            cm = UserClusterManager(
                n_clusters=n_clusters,
                embedding_dim=item_embeddings.shape[1],
                n_items=len(item_embeddings),
            )
            cm.set_item_embeddings(item_embeddings)
            cm.initialize_from_interactions(
                train_data.user_ids,
                train_data.item_ids,
                train_data.ratings,
            )
            return cm

        cluster_manager = create_cluster_manager(50)

        # 1. Recommender baselines
        if not args.skip_recommender_baselines:
            print("\n" + "="*60)
            print("1. RECOMMENDER MODEL BASELINES")
            print("="*60)
            dataset_results["recommender_baselines"] = run_recommender_baselines(
                train_data, test_data, n_runs=args.n_runs, verbose=True
            )

        # 2. Cache baselines
        if not args.skip_cache_baselines:
            print("\n" + "="*60)
            print("2. CACHE STRATEGY BASELINES")
            print("="*60)
            dataset_results["cache_baselines"] = run_cache_baselines(
                base_recommender, train_data, test_data, cluster_manager,
                n_requests=args.n_requests, n_runs=args.n_runs, verbose=True
            )

        # 3. Ablation study
        if not args.skip_ablation:
            print("\n" + "="*60)
            print("3. ABLATION STUDY")
            print("="*60)
            dataset_results["ablation"] = run_ablation_study(
                base_recommender, train_data, test_data, cluster_manager,
                n_requests=args.n_requests, n_runs=args.n_runs, verbose=True
            )

        # 4. Parameter sensitivity
        if not args.skip_sensitivity:
            print("\n" + "="*60)
            print("4. PARAMETER SENSITIVITY")
            print("="*60)
            dataset_results["sensitivity"] = run_parameter_sensitivity(
                base_recommender, train_data, test_data, create_cluster_manager,
                n_requests=min(3000, args.n_requests), n_runs=min(3, args.n_runs),
                verbose=True
            )

        # 5. User group analysis
        if not args.skip_user_groups:
            print("\n" + "="*60)
            print("5. USER GROUP ANALYSIS")
            print("="*60)
            cache_config = CacheConfig(local_cache_size=5000, use_redis_cache=False)
            cache_manager = CacheManager(
                cache_config=cache_config,
                cluster_manager=cluster_manager,
            )
            dataset_results["user_groups"] = run_user_group_analysis(
                base_recommender, train_data, test_data,
                cache_manager, cluster_manager, verbose=True
            )

        all_results[dataset_name] = dataset_results

    # Save results
    save_all_results(all_results, args.output_dir)

    # Print summary
    print_final_summary(all_results)

    # Statistical significance testing
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)

    tester = StatisticalTester()

    for dataset_name, results in all_results.items():
        if "cache_baselines" in results:
            print(f"\n{dataset_name} - Cache Strategies vs LRU baseline:")
            cache_results = results["cache_baselines"]

            # Convert to list format for testing
            all_runs = {}
            for method, res in cache_results.items():
                if hasattr(res, 'all_runs'):
                    all_runs[method] = res.all_runs

            if "lru" in all_runs:
                sig_tests = tester.compare_methods(all_runs, "lru", "hit_rate")
                for test in sig_tests:
                    print(f"  {test}")

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
