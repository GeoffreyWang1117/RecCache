#!/usr/bin/env python3
"""
P2-2: Concept drift experiment.

Split ML-1M test data into temporal windows, measure epsilon (embedding
space alignment between user and cluster centroid) and actual NDCG loss
per window. Validates the regret bound O(T*epsilon) assumption.

Usage:
    conda activate reccache
    python scripts/run_concept_drift.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import RecommendationMetrics


N_RECS = 10
N_WINDOWS = 6


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


def run_experiment():
    print("=" * 70)
    print("  Concept Drift Experiment — ML-1M")
    print("=" * 70)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        "ml-1m", min_user_interactions=5, min_item_interactions=5,
    )
    gt = build_ground_truth(test, min_rating=4.0)

    # Train model
    np.random.seed(42)
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=64,
    )
    model.fit(train.user_ids, train.item_ids, train.ratings,
              epochs=15, verbose=True)
    item_embs = model.get_all_item_embeddings()

    # Build cluster manager
    cm = UserClusterManager(
        n_clusters=50, embedding_dim=item_embs.shape[1], n_items=len(item_embs),
    )
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # Build user history for exclusion
    user_hist = build_user_history(train)

    # Get ordered test users
    test_users_ordered = []
    seen = set()
    for uid in test.user_ids:
        uid = int(uid)
        if uid not in seen and uid in gt:
            test_users_ordered.append(uid)
            seen.add(uid)

    # Split into temporal windows
    window_size = len(test_users_ordered) // N_WINDOWS
    windows = []
    for i in range(N_WINDOWS):
        start = i * window_size
        end = start + window_size if i < N_WINDOWS - 1 else len(test_users_ordered)
        windows.append(test_users_ordered[start:end])

    print(f"  {len(test_users_ordered)} test users split into {N_WINDOWS} windows "
          f"(~{window_size} each)")

    # For each window, measure with and without re-clustering
    results_no_recluster = []
    results_with_recluster = []

    # --- Without re-clustering (stale clusters) ---
    print(f"\n  Running WITHOUT re-clustering...")
    criterion = ScoreRatioAcceptanceCriterion(threshold=0.35)
    config = SpeculativeConfig(
        top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS,
    )
    spec_stale = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=criterion, config=config,
        item_embeddings=item_embs,
        user_history=user_hist,
    )
    train_users = list(set(int(u) for u in train.user_ids))[:500]
    spec_stale.warm_cache(train_users)

    for wi, window_users in enumerate(windows):
        epsilons = []
        ndcgs_spec = []
        ndcgs_fresh = []

        for uid in window_users:
            if uid not in gt:
                continue

            # Measure epsilon: distance from user to nearest cluster centroid
            info = cm.get_user_cluster(uid)
            epsilons.append(info.distance_to_center)

            # Speculative NDCG
            sr = spec_stale.recommend(uid)
            ndcg_s = RecommendationMetrics.ndcg_at_k(sr.items, gt[uid], N_RECS)
            ndcgs_spec.append(ndcg_s)

            # Fresh NDCG
            exclude = user_hist.get(uid)
            fresh_recs = model.recommend(uid, n=N_RECS, exclude_items=exclude)
            ndcg_f = RecommendationMetrics.ndcg_at_k(fresh_recs, gt[uid], N_RECS)
            ndcgs_fresh.append(ndcg_f)

        regret = np.mean(ndcgs_fresh) - np.mean(ndcgs_spec) if ndcgs_fresh else 0
        results_no_recluster.append({
            "window": wi,
            "n_users": len(window_users),
            "epsilon_mean": float(np.mean(epsilons)) if epsilons else 0,
            "epsilon_std": float(np.std(epsilons)) if epsilons else 0,
            "ndcg_spec": float(np.mean(ndcgs_spec)) if ndcgs_spec else 0,
            "ndcg_fresh": float(np.mean(ndcgs_fresh)) if ndcgs_fresh else 0,
            "regret": float(regret),
        })
        print(f"    Window {wi}: ε={np.mean(epsilons):.4f}, "
              f"NDCG_spec={np.mean(ndcgs_spec):.4f}, "
              f"NDCG_fresh={np.mean(ndcgs_fresh):.4f}, "
              f"regret={regret:.4f}")

    # --- With re-clustering every window ---
    print(f"\n  Running WITH re-clustering per window...")
    for wi, window_users in enumerate(windows):
        # Re-initialize clusters with accumulated data up to this window
        cm_fresh = UserClusterManager(
            n_clusters=50, embedding_dim=item_embs.shape[1], n_items=len(item_embs),
        )
        cm_fresh.set_item_embeddings(item_embs)
        cm_fresh.initialize_from_interactions(
            train.user_ids, train.item_ids, train.ratings
        )
        # Also update with users from previous windows
        for prev_wi in range(wi):
            for uid in windows[prev_wi]:
                cm_fresh.get_user_cluster(uid)  # triggers embedding computation

        spec_fresh = SpeculativeRecommender(
            recommender=model, cluster_manager=cm_fresh,
            acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
            config=SpeculativeConfig(
                top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS,
            ),
            item_embeddings=item_embs,
            user_history=user_hist,
        )
        all_users_so_far = list(set(int(u) for u in train.user_ids))[:500]
        spec_fresh.warm_cache(all_users_so_far)

        epsilons = []
        ndcgs_spec = []
        ndcgs_fresh_list = []

        for uid in window_users:
            if uid not in gt:
                continue

            info = cm_fresh.get_user_cluster(uid)
            epsilons.append(info.distance_to_center)

            sr = spec_fresh.recommend(uid)
            ndcg_s = RecommendationMetrics.ndcg_at_k(sr.items, gt[uid], N_RECS)
            ndcgs_spec.append(ndcg_s)

            exclude = user_hist.get(uid)
            fresh_recs = model.recommend(uid, n=N_RECS, exclude_items=exclude)
            ndcg_f = RecommendationMetrics.ndcg_at_k(fresh_recs, gt[uid], N_RECS)
            ndcgs_fresh_list.append(ndcg_f)

        regret = np.mean(ndcgs_fresh_list) - np.mean(ndcgs_spec) if ndcgs_fresh_list else 0
        results_with_recluster.append({
            "window": wi,
            "n_users": len(window_users),
            "epsilon_mean": float(np.mean(epsilons)) if epsilons else 0,
            "epsilon_std": float(np.std(epsilons)) if epsilons else 0,
            "ndcg_spec": float(np.mean(ndcgs_spec)) if ndcgs_spec else 0,
            "ndcg_fresh": float(np.mean(ndcgs_fresh_list)) if ndcgs_fresh_list else 0,
            "regret": float(regret),
        })
        print(f"    Window {wi}: ε={np.mean(epsilons):.4f}, "
              f"NDCG_spec={np.mean(ndcgs_spec):.4f}, "
              f"regret={regret:.4f}")

    return {
        "n_windows": N_WINDOWS,
        "no_recluster": results_no_recluster,
        "with_recluster": results_with_recluster,
    }


def make_figures(results, output_dir):
    """Generate concept drift figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = list(range(results["n_windows"]))

    nr = results["no_recluster"]
    wr = results["with_recluster"]

    # 1. Epsilon over windows
    eps_nr = [w["epsilon_mean"] for w in nr]
    eps_wr = [w["epsilon_mean"] for w in wr]
    axes[0].plot(x, eps_nr, "o-", label="No re-cluster", color="#e74c3c")
    axes[0].plot(x, eps_wr, "s-", label="With re-cluster", color="#3498db")
    axes[0].set_xlabel("Temporal Window")
    axes[0].set_ylabel("Mean ε (dist to centroid)")
    axes[0].set_title("Embedding Alignment (ε)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 2. Regret over windows
    reg_nr = [w["regret"] for w in nr]
    reg_wr = [w["regret"] for w in wr]
    axes[1].plot(x, reg_nr, "o-", label="No re-cluster", color="#e74c3c")
    axes[1].plot(x, reg_wr, "s-", label="With re-cluster", color="#3498db")
    axes[1].set_xlabel("Temporal Window")
    axes[1].set_ylabel("Regret (Fresh - Spec NDCG)")
    axes[1].set_title("Per-Window Regret")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. Epsilon vs Regret scatter
    all_eps = eps_nr + eps_wr
    all_reg = reg_nr + reg_wr
    axes[2].scatter(eps_nr, reg_nr, marker="o", color="#e74c3c",
                     label="No re-cluster", alpha=0.7, s=60)
    axes[2].scatter(eps_wr, reg_wr, marker="s", color="#3498db",
                     label="With re-cluster", alpha=0.7, s=60)
    # Trend line
    if len(all_eps) > 2:
        z = np.polyfit(all_eps, all_reg, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_eps), max(all_eps), 50)
        axes[2].plot(x_line, p(x_line), "--", color="gray", alpha=0.5,
                      label=f"Trend (slope={z[0]:.3f})")
    axes[2].set_xlabel("Mean ε")
    axes[2].set_ylabel("Regret")
    axes[2].set_title("ε vs Regret (O(T·ε) validation)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"concept_drift.{ext}", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    results = run_experiment()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "concept_drift_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/concept_drift_results.json")

    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    make_figures(results, figures_dir)
    print(f"Figure saved to paper/figures/concept_drift.{{pdf,png}}")


if __name__ == "__main__":
    main()
