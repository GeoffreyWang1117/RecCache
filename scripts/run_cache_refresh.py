#!/usr/bin/env python3
"""
Cache Refresh / Re-clustering Frequency Sensitivity.

Simulates temporal drift by splitting ML-1M interactions into T windows.
At each window boundary, optionally re-clusters users and refreshes caches.
Compares:
  - Never refresh (static clusters from window 0)
  - Refresh every W windows
  - Refresh every window (oracle-like)

Also computes cluster purity (intra-cluster cosine similarity) and
inter-cluster recommendation overlap per window.

Outputs: results/supp_cache_refresh.json
         paper/figures/supp_cache_refresh.pdf

Usage:
    conda activate reccache
    python scripts/run_cache_refresh.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

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
N_CLUSTERS = 50
EMB_DIM = 64
SEED = 42
N_WINDOWS = 6  # split test into 6 temporal windows


def cluster_purity(cm, user_ids, item_embs):
    """Mean intra-cluster cosine similarity among user embeddings."""
    cluster_users = defaultdict(list)
    for uid in user_ids:
        info = cm.get_user_cluster(uid)
        emb = cm.get_user_embedding(uid)
        if emb is not None:
            cluster_users[info.cluster_id].append(emb)

    purities = []
    for cid, embs in cluster_users.items():
        if len(embs) < 2:
            continue
        embs = np.array(embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs_n = embs / norms
        sim = embs_n @ embs_n.T
        n = len(embs)
        # mean off-diagonal similarity
        total = (sim.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
        purities.append(float(total))
    return float(np.mean(purities)) if purities else 0.0


def rec_overlap(recs_dict):
    """Mean pairwise Jaccard overlap of recommendation lists across users."""
    lists = list(recs_dict.values())
    if len(lists) < 2:
        return 0.0
    # Sample pairs for efficiency
    rng = np.random.default_rng(42)
    n_pairs = min(500, len(lists) * (len(lists)-1) // 2)
    overlaps = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(lists), size=2, replace=False)
        s1, s2 = set(lists[i]), set(lists[j])
        union = len(s1 | s2)
        overlaps.append(len(s1 & s2) / union if union > 0 else 0.0)
    return float(np.mean(overlaps))


def run_windows(model, train, test, n_windows, refresh_every, user_history, item_embs):
    """Run speculative serving across temporal windows with optional cache refresh."""
    gt = defaultdict(set)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= 4.0:
            gt[int(uid)].add(int(iid))

    # Split test users into temporal windows by order in test set
    seen = set()
    all_test_users = []
    for uid in test.user_ids:
        u = int(uid)
        if u not in seen and u in gt:
            all_test_users.append(u)
            seen.add(u)

    window_size = max(1, len(all_test_users) // n_windows)
    windows_users = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(all_test_users)
        windows_users.append(all_test_users[start:end])

    # Initial cluster manager
    cm = UserClusterManager(n_clusters=N_CLUSTERS, embedding_dim=EMB_DIM)
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    config = SpeculativeConfig(
        top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS, use_pool_retrieval=False,
    )

    def build_spec(cm_):
        s = SpeculativeRecommender(
            recommender=model, cluster_manager=cm_,
            acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
            config=config, item_embeddings=item_embs, user_history=user_history,
        )
        s.warm_cache(list(range(train.n_users)))
        return s

    spec = build_spec(cm)
    window_results = []

    for w_idx, w_users in enumerate(windows_users):
        # Check if we should refresh
        if refresh_every > 0 and w_idx > 0 and w_idx % refresh_every == 0:
            # Re-cluster with accumulated interactions
            cm = UserClusterManager(n_clusters=N_CLUSTERS, embedding_dim=EMB_DIM)
            cm.set_item_embeddings(item_embs)
            cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
            spec = build_spec(cm)

        ndcgs, recalls, accepts = [], [], []
        recs_dict = {}
        for uid in w_users:
            sr = spec.recommend(uid)
            relevant = gt.get(uid, set())
            if relevant:
                ndcgs.append(RecommendationMetrics.ndcg_at_k(sr.items, relevant, N_RECS))
                recalls.append(RecommendationMetrics.recall_at_k(sr.items, relevant, N_RECS))
            accepts.append(int(sr.accepted))
            recs_dict[uid] = sr.items

        purity = cluster_purity(cm, w_users, item_embs)
        overlap = rec_overlap(recs_dict)

        window_results.append({
            "window": w_idx,
            "n_users": len(w_users),
            "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "accept_rate": float(np.mean(accepts)),
            "cluster_purity": purity,
            "rec_overlap": overlap,
        })

    overall = {
        "ndcg": float(np.mean([w["ndcg"] for w in window_results])),
        "recall": float(np.mean([w["recall"] for w in window_results])),
        "accept_rate": float(np.mean([w["accept_rate"] for w in window_results])),
        "cluster_purity": float(np.mean([w["cluster_purity"] for w in window_results])),
        "rec_overlap": float(np.mean([w["rec_overlap"] for w in window_results])),
    }
    return {"overall": overall, "windows": window_results}


def main():
    np.random.seed(SEED)
    import torch; torch.manual_seed(SEED)
    print("Cache Refresh Sensitivity | ML-1M")

    loader = DataLoader("data")
    train, val, test = loader.load_dataset("ml-1m")

    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))

    print("  Training MF...")
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=EMB_DIM
    )
    model.fit(train.user_ids, train.item_ids, train.ratings, epochs=15, verbose=False)
    item_embs = model.get_all_item_embeddings()

    # Run with different refresh frequencies
    refresh_configs = {
        "never":    0,   # never refresh (static)
        "every_3":  3,   # refresh every 3 windows
        "every_2":  2,   # refresh every 2 windows
        "every_1":  1,   # refresh every window (oracle-like)
    }

    results = {}
    for label, freq in refresh_configs.items():
        print(f"  Refresh={label} (every {freq if freq > 0 else 'never'})...")
        results[label] = run_windows(model, train, test, N_WINDOWS, freq,
                                     dict(user_history), item_embs)
        o = results[label]["overall"]
        print(f"    NDCG={o['ndcg']:.4f} Accept={o['accept_rate']:.1%} "
              f"Purity={o['cluster_purity']:.3f} Overlap={o['rec_overlap']:.3f}")

    # Summary
    print(f"\n  {'Refresh':<12} {'NDCG':>8} {'Accept':>8} {'Purity':>8} {'Overlap':>8}")
    print("  " + "-" * 50)
    for label, r in results.items():
        o = r["overall"]
        print(f"  {label:<12} {o['ndcg']:>8.4f} {o['accept_rate']:>8.1%} "
              f"{o['cluster_purity']:>8.3f} {o['rec_overlap']:>8.3f}")

    # Save
    out = Path("results/supp_cache_refresh.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"never": "#c0392b", "every_3": "#e67e22", "every_2": "#2980b9", "every_1": "#27ae60"}

    for ax, (metric, label) in zip(axes, [("ndcg","NDCG@10"),("accept_rate","Accept Rate"),
                                           ("cluster_purity","Cluster Purity")]):
        for name, r in results.items():
            ws = r["windows"]
            xs = [w["window"] for w in ws]
            ys = [w[metric] for w in ws]
            ax.plot(xs, ys, "-o", color=colors.get(name,"gray"), label=name, markersize=5)
        ax.set_xlabel("Temporal Window")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig_path = Path("paper/figures/supp_cache_refresh.pdf")
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
