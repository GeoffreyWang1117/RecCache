#!/usr/bin/env python3
"""
Statistical significance tests for RecSys 2026 submission.

Computes per-user NDCG for all methods in S4 (end-to-end comparison),
then runs paired t-tests and Wilcoxon signed-rank tests comparing
each cached method against Fresh baseline.

Usage:
    conda activate reccache
    python scripts/run_significance_tests.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy import stats

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import RecommendationMetrics


# ---------------------------------------------------------------------------
# Config (same as run_recsys_experiments.py)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
    "amazon-electronics": {
        "max_samples": 1000000,
        "min_user": 3, "min_item": 3,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
    "amazon-arts": {
        "max_samples": 1000000,
        "min_user": 3, "min_item": 3,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
    "mind-small": {
        "max_samples": 200000,
        "min_user": 5, "min_item": 5,
        "implicit": True, "min_rating_gt": 0.5,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
}

N_TEST_USERS = 500
N_RECS = 10


def build_ground_truth(test_data, min_rating=4.0):
    gt = defaultdict(set)
    for uid, iid, r in zip(test_data.user_ids, test_data.item_ids, test_data.ratings):
        if r >= min_rating:
            gt[int(uid)].add(int(iid))
    return dict(gt)


def build_user_history(train_data):
    history = defaultdict(list)
    for uid, iid in zip(train_data.user_ids, train_data.item_ids):
        history[int(uid)].append(int(iid))
    return dict(history)


def get_per_user_ndcg(recommender, cluster_manager, item_embeddings,
                      ground_truth, test_users, method, cfg,
                      user_history=None):
    """Get per-user NDCG for a given method."""
    user_ndcgs = {}

    if method == "Fresh":
        for uid in test_users:
            if uid not in ground_truth:
                continue
            exclude = user_history.get(uid) if user_history else None
            recs = list(recommender.recommend(uid, n=N_RECS, exclude_items=exclude))
            user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(
                recs, ground_truth[uid], N_RECS
            )

    elif method == "Naive":
        criterion = ScoreRatioAcceptanceCriterion(threshold=0.0)
        config = SpeculativeConfig(
            top_k_clusters=1, acceptance_threshold=0.0, n_recs=N_RECS,
        )
        spec = SpeculativeRecommender(
            recommender=recommender, cluster_manager=cluster_manager,
            acceptance_criterion=criterion, config=config,
            item_embeddings=item_embeddings,
            user_history=user_history,
        )
        all_users = list(set(int(u) for u in cluster_manager._user_embeddings.keys()))
        spec.warm_cache(all_users)
        for uid in test_users:
            if uid not in ground_truth:
                continue
            sr = spec.recommend(uid)
            user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(
                sr.items, ground_truth[uid], N_RECS
            )

    elif method == "Speculative":
        criterion = ScoreRatioAcceptanceCriterion(threshold=0.35)
        config = SpeculativeConfig(
            top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS,
        )
        spec = SpeculativeRecommender(
            recommender=recommender, cluster_manager=cluster_manager,
            acceptance_criterion=criterion, config=config,
            item_embeddings=item_embeddings,
            user_history=user_history,
        )
        all_users = list(set(int(u) for u in cluster_manager._user_embeddings.keys()))
        spec.warm_cache(all_users)
        for uid in test_users:
            if uid not in ground_truth:
                continue
            sr = spec.recommend(uid)
            user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(
                sr.items, ground_truth[uid], N_RECS
            )

    elif method == "Spec+Pool":
        criterion = ScoreRatioAcceptanceCriterion(threshold=0.35)
        config = SpeculativeConfig(
            top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS,
            use_pool_retrieval=True, pool_size=200,
        )
        spec = SpeculativeRecommender(
            recommender=recommender, cluster_manager=cluster_manager,
            acceptance_criterion=criterion, config=config,
            item_embeddings=item_embeddings,
            user_history=user_history,
        )
        all_users = list(set(int(u) for u in cluster_manager._user_embeddings.keys()))
        spec.warm_cache(all_users)
        for uid in test_users:
            if uid not in ground_truth:
                continue
            sr = spec.recommend(uid)
            user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(
                sr.items, ground_truth[uid], N_RECS
            )

    return user_ndcgs


def run_significance_tests(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"\n{'='*70}")
    print(f"  Statistical Significance Tests — {dataset_name}")
    print(f"{'='*70}")

    # Load data
    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        dataset_name,
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
        max_samples=cfg["max_samples"],
    )
    gt = build_ground_truth(test, min_rating=cfg["min_rating_gt"])
    user_hist = build_user_history(train)
    test_users = [u for u in set(test.user_ids.tolist()) if u in gt][:N_TEST_USERS]

    # Train model
    np.random.seed(42)
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items,
        embedding_dim=cfg["embedding_dim"],
    )
    model.fit(train.user_ids, train.item_ids, train.ratings,
              epochs=cfg["epochs"], verbose=True)
    item_embs = model.get_all_item_embeddings()

    # Cluster manager
    n_clusters = min(cfg["n_clusters"], train.n_users // 2)
    cm = UserClusterManager(
        n_clusters=n_clusters,
        embedding_dim=item_embs.shape[1],
        n_items=len(item_embs),
    )
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # Get per-user NDCG for each method
    methods = ["Fresh", "Naive", "Speculative", "Spec+Pool"]
    per_user = {}

    for method in methods:
        print(f"  Computing per-user NDCG for {method}...", flush=True)
        per_user[method] = get_per_user_ndcg(
            model, cm, item_embs, gt, test_users, method, cfg,
            user_history=user_hist,
        )

    # Align users (only those present in all methods)
    common_users = set(per_user["Fresh"].keys())
    for method in methods:
        common_users &= set(per_user[method].keys())
    common_users = sorted(common_users)
    n = len(common_users)

    print(f"\n  {n} common users for paired tests")

    # Build aligned arrays
    arrays = {}
    for method in methods:
        arrays[method] = np.array([per_user[method][u] for u in common_users])

    # Run paired tests: each method vs Fresh
    results = {"dataset": dataset_name, "n_users": n, "comparisons": {}}

    print(f"\n  {'Comparison':<30} {'Mean Δ':>8} {'t-stat':>8} {'p (t)':>10} "
          f"{'p (Wilc.)':>10} {'Sig?':>5}")
    print(f"  {'-'*75}")

    for method in ["Naive", "Speculative", "Spec+Pool"]:
        diff = arrays[method] - arrays["Fresh"]
        mean_diff = np.mean(diff)

        # Paired t-test
        t_stat, p_ttest = stats.ttest_rel(arrays[method], arrays["Fresh"])

        # Wilcoxon signed-rank test
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(arrays[method], arrays["Fresh"])
        except ValueError:
            # All differences are zero
            w_stat, p_wilcoxon = 0.0, 1.0

        # Bonferroni correction (3 comparisons)
        p_ttest_corr = min(p_ttest * 3, 1.0)
        p_wilcoxon_corr = min(p_wilcoxon * 3, 1.0)

        sig = "***" if p_ttest_corr < 0.001 else "**" if p_ttest_corr < 0.01 else "*" if p_ttest_corr < 0.05 else "ns"

        results["comparisons"][f"{method} vs Fresh"] = {
            "mean_diff": float(mean_diff),
            "t_statistic": float(t_stat),
            "p_ttest": float(p_ttest),
            "p_ttest_bonferroni": float(p_ttest_corr),
            "p_wilcoxon": float(p_wilcoxon),
            "p_wilcoxon_bonferroni": float(p_wilcoxon_corr),
            "significant": sig != "ns",
            "significance_level": sig,
        }

        print(f"  {method+' vs Fresh':<30} {mean_diff:+8.4f} {t_stat:8.3f} "
              f"{p_ttest_corr:10.2e} {p_wilcoxon_corr:10.2e} {sig:>5}")

    # Also compare Speculative vs Naive
    diff = arrays["Speculative"] - arrays["Naive"]
    t_stat, p_val = stats.ttest_rel(arrays["Speculative"], arrays["Naive"])
    try:
        _, p_wilc = stats.wilcoxon(arrays["Speculative"], arrays["Naive"])
    except ValueError:
        p_wilc = 1.0
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    results["comparisons"]["Spec vs Naive"] = {
        "mean_diff": float(np.mean(diff)),
        "t_statistic": float(t_stat),
        "p_ttest": float(p_val),
        "significant": sig != "ns",
        "significance_level": sig,
    }
    print(f"  {'Spec vs Naive':<30} {np.mean(diff):+8.4f} {t_stat:8.3f} "
          f"{p_val:10.2e} {p_wilc:10.2e} {sig:>5}")

    # Summary statistics
    print(f"\n  Method means (± std):")
    for method in methods:
        arr = arrays[method]
        print(f"    {method:<16} {np.mean(arr):.4f} ± {np.std(arr):.4f}")

    return results


def main():
    all_results = {}
    for ds in DATASET_CONFIGS.keys():
        all_results[ds] = run_significance_tests(ds)

    out_path = Path("results") / "significance_tests.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
