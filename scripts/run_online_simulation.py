#!/usr/bin/env python3
"""
P2-1: Online simulation experiment.

Simulate streaming requests in temporal order on ML-1M. Measure how
acceptance rate, NDCG, and MCG evolve over time as users arrive
sequentially. Validates online K-Means adaptation capability.

Usage:
    conda activate reccache
    python scripts/run_online_simulation.py
"""

import sys
import json
import time
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
from reccache.evaluation.metrics import RecommendationMetrics, SpeculativeMetrics


N_RECS = 10
WINDOW_SIZE = 100  # users per window for rolling metrics


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
    print("  Online Simulation — ML-1M (temporal order)")
    print("=" * 70)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        "ml-1m", min_user_interactions=5, min_item_interactions=5,
    )
    gt = build_ground_truth(test, min_rating=4.0)

    # Sort test interactions by timestamp if available, otherwise use order
    test_users_ordered = []
    seen = set()
    for uid in test.user_ids:
        uid = int(uid)
        if uid not in seen and uid in gt:
            test_users_ordered.append(uid)
            seen.add(uid)

    print(f"  {len(test_users_ordered)} test users in temporal order")

    # Train model on training data
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

    # Build speculative recommender
    criterion = ScoreRatioAcceptanceCriterion(threshold=0.35)
    config = SpeculativeConfig(
        top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS,
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=criterion, config=config,
        item_embeddings=item_embs,
        user_history=user_hist,
    )

    # Warm cache with training users
    train_users = list(set(int(u) for u in train.user_ids))[:500]
    spec.warm_cache(train_users)

    # Simulate online serving
    print(f"\n  Simulating {len(test_users_ordered)} sequential requests...")

    per_request = {
        "ndcg": [],
        "accepted": [],
        "mcg": [],  # 1 if non-nearest cluster, 0 otherwise
    }

    for i, uid in enumerate(test_users_ordered):
        sr = spec.recommend(uid)
        ndcg = RecommendationMetrics.ndcg_at_k(sr.items, gt[uid], N_RECS)

        per_request["ndcg"].append(ndcg)
        per_request["accepted"].append(1 if sr.accepted else 0)
        mcg = 1 if (sr.accepted and sr.accepted_cluster_rank > 0) else 0
        per_request["mcg"].append(mcg)

        if (i + 1) % 200 == 0:
            window = slice(max(0, i - WINDOW_SIZE + 1), i + 1)
            avg_ndcg = np.mean(per_request["ndcg"][window])
            avg_accept = np.mean(per_request["accepted"][window])
            print(f"    Request {i+1}: rolling NDCG={avg_ndcg:.4f}, "
                  f"Accept={avg_accept:.1%}")

    # Compute rolling metrics
    n = len(per_request["ndcg"])
    windows = []
    for start in range(0, n, WINDOW_SIZE):
        end = min(start + WINDOW_SIZE, n)
        if end - start < WINDOW_SIZE // 2:
            break
        windows.append({
            "start": start,
            "end": end,
            "ndcg": float(np.mean(per_request["ndcg"][start:end])),
            "accept_rate": float(np.mean(per_request["accepted"][start:end])),
            "mcg_rate": float(np.mean(per_request["mcg"][start:end])),
        })

    results = {
        "n_requests": n,
        "window_size": WINDOW_SIZE,
        "overall": {
            "ndcg": float(np.mean(per_request["ndcg"])),
            "accept_rate": float(np.mean(per_request["accepted"])),
            "mcg_rate": float(np.mean(per_request["mcg"])),
        },
        "windows": windows,
    }

    print(f"\n  Overall: NDCG={results['overall']['ndcg']:.4f}, "
          f"Accept={results['overall']['accept_rate']:.1%}, "
          f"MCG={results['overall']['mcg_rate']:.1%}")

    return results


def make_figures(results, output_dir):
    """Generate online simulation figure."""
    windows = results["windows"]
    x = [(w["start"] + w["end"]) / 2 for w in windows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # NDCG over time
    ndcgs = [w["ndcg"] for w in windows]
    axes[0].plot(x, ndcgs, "-", color="#3498db", linewidth=1.5)
    axes[0].axhline(y=results["overall"]["ndcg"], color="gray", linestyle="--",
                     alpha=0.5, label=f'Mean={results["overall"]["ndcg"]:.4f}')
    axes[0].set_xlabel("Request Number")
    axes[0].set_ylabel("NDCG@10")
    axes[0].set_title("Quality Over Time")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Accept rate over time
    accepts = [w["accept_rate"] for w in windows]
    axes[1].plot(x, accepts, "-", color="#2ecc71", linewidth=1.5)
    axes[1].axhline(y=results["overall"]["accept_rate"], color="gray",
                     linestyle="--", alpha=0.5,
                     label=f'Mean={results["overall"]["accept_rate"]:.1%}')
    axes[1].set_xlabel("Request Number")
    axes[1].set_ylabel("Acceptance Rate")
    axes[1].set_title("Cache Utilization Over Time")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # MCG over time
    mcgs = [w["mcg_rate"] for w in windows]
    axes[2].plot(x, mcgs, "-", color="#e67e22", linewidth=1.5)
    axes[2].axhline(y=results["overall"]["mcg_rate"], color="gray",
                     linestyle="--", alpha=0.5,
                     label=f'Mean={results["overall"]["mcg_rate"]:.1%}')
    axes[2].set_xlabel("Request Number")
    axes[2].set_ylabel("MCG Rate")
    axes[2].set_title("Multi-Cluster Gain Over Time")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"online_simulation.{ext}", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    results = run_experiment()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "online_simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/online_simulation_results.json")

    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    make_figures(results, figures_dir)
    print(f"Figure saved to paper/figures/online_simulation.{{pdf,png}}")


if __name__ == "__main__":
    main()
