#!/usr/bin/env python3
"""
A4 — Cluster Stability and Acceptance vs Distance Analysis.

Two analyses on ML-1M:
  (1) Cluster-assignment stability under history truncation:
      For each user, compare cluster assignment using full history vs first
      50% of history. Fraction unchanged = stability proxy.
  (2) Acceptance probability vs distance to nearest cluster center:
      Scatter plot validates the ε-bound from §3.3 empirically.
  (3) Per-cluster acceptance distribution:
      Histogram + Gini for fairness across clusters.

Outputs: results/supp_a4_cluster_analysis.json
         paper/figures/supp_acceptance_vs_distance.pdf
         paper/figures/supp_cluster_stability.pdf

Usage:
    conda activate reccache
    python scripts/run_cluster_analysis.py
"""

import sys
import argparse
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


N_CLUSTERS = 50
EMB_DIM = 64
EPOCHS = 15
TOP_K = 3
THRESHOLD = 0.35
N_RECS = 10
SEED = 42
N_USERS_FOR_PLOT = 1500


def set_seed(seed):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v) - (n + 1) * v.sum()) / (n * v.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(exist_ok=True, parents=True)

    set_seed(SEED)
    print(f"A4: cluster analysis | {args.dataset}")

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(args.dataset)
    n_users, n_items = train.n_users, train.n_items
    print(f"  {n_users} users, {n_items} items, {len(train.user_ids)} train interactions")

    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))

    # Train MF
    print("  Training MF...")
    model = MatrixFactorizationRecommender(
        n_users=n_users, n_items=n_items, embedding_dim=EMB_DIM
    )
    model.fit(train.user_ids, train.item_ids, train.ratings, epochs=EPOCHS, verbose=False)
    item_embs = model.get_all_item_embeddings()

    # ---- Build "full history" cluster manager ----
    print("  Building full-history clusters...")
    cm_full = UserClusterManager(n_clusters=N_CLUSTERS, embedding_dim=EMB_DIM)
    cm_full.set_item_embeddings(item_embs)
    cm_full.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # ---- Build "half history" cluster manager (per-user temporal split) ----
    print("  Building half-history clusters (temporal stability test)...")
    half_uids, half_iids, half_rs = [], [], []
    user_seen = defaultdict(int)
    user_caps = {u: max(1, len(h) // 2) for u, h in user_history.items()}
    for uid, iid, r in zip(train.user_ids, train.item_ids, train.ratings):
        u = int(uid)
        if user_seen[u] < user_caps[u]:
            half_uids.append(u); half_iids.append(int(iid)); half_rs.append(float(r))
            user_seen[u] += 1
    half_uids = np.array(half_uids); half_iids = np.array(half_iids); half_rs = np.array(half_rs)

    cm_half = UserClusterManager(n_clusters=N_CLUSTERS, embedding_dim=EMB_DIM)
    cm_half.set_item_embeddings(item_embs)
    cm_half.initialize_from_interactions(half_uids, half_iids, half_rs)

    # Stability: fraction of users assigned to the same cluster under both
    print("  Measuring cluster-assignment stability...")
    stable_count = 0
    distance_diff = []
    eligible = list(user_history.keys())
    for uid in eligible:
        f = cm_full.get_user_cluster(uid)
        h = cm_half.get_user_cluster(uid)
        if f.cluster_id == h.cluster_id:
            stable_count += 1
        distance_diff.append(abs(f.distance_to_center - h.distance_to_center))
    stability = stable_count / len(eligible) if eligible else 0.0
    mean_dist_diff = float(np.mean(distance_diff)) if distance_diff else 0.0
    print(f"    Stability: {stability:.1%}  (mean distance diff: {mean_dist_diff:.4f})")

    # ---- Acceptance vs distance ----
    print("  Running speculative serving with full clusters...")
    config = SpeculativeConfig(
        top_k_clusters=TOP_K, acceptance_threshold=THRESHOLD,
        n_recs=N_RECS, use_pool_retrieval=False,
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm_full,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=THRESHOLD),
        config=config, item_embeddings=item_embs, user_history=dict(user_history),
    )
    spec.warm_cache(list(range(n_users)))

    sample = np.random.default_rng(SEED).choice(
        n_users, size=min(N_USERS_FOR_PLOT, n_users), replace=False
    )

    distances = []
    accept_probs = []
    accepts = []
    cluster_accept_count = Counter()
    cluster_total_count = Counter()

    for uid in sample:
        info = cm_full.get_user_cluster(int(uid))
        res = spec.recommend(int(uid))
        distances.append(info.distance_to_center)
        accept_probs.append(res.acceptance_prob)
        accepts.append(int(res.accepted))
        cluster_total_count[info.cluster_id] += 1
        if res.accepted:
            cluster_accept_count[res.accepted_cluster_id] += 1

    distances = np.array(distances)
    accept_probs = np.array(accept_probs)
    accepts = np.array(accepts)

    # Bin distances for trend line
    n_bins = 10
    bin_edges = np.quantile(distances, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(distances, bin_edges) - 1
    bin_centers, bin_accept_rate, bin_mean_prob = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 5:
            bin_centers.append(distances[mask].mean())
            bin_accept_rate.append(accepts[mask].mean())
            bin_mean_prob.append(accept_probs[mask].mean())

    # Per-cluster acceptance rate
    cluster_accept_rate = []
    for c, total in cluster_total_count.items():
        if total >= 5:
            cluster_accept_rate.append(cluster_accept_count.get(c, 0) / total)
    cluster_accept_rate = np.array(cluster_accept_rate)
    cluster_gini = gini(cluster_accept_rate * 100) if len(cluster_accept_rate) > 0 else 0.0

    print(f"    Sampled {len(sample)} users")
    print(f"    Acceptance rate: {accepts.mean():.1%}")
    print(f"    Mean distance:   {distances.mean():.4f}")
    print(f"    Spearman ρ(distance, accept_prob) = ",
          end="")
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(distances, accept_probs)
        print(f"{rho:.3f} (p={p:.2e})")
    except ImportError:
        print("scipy unavailable")
        rho = float("nan"); p = float("nan")
    print(f"    Per-cluster acceptance Gini: {cluster_gini:.3f} ({len(cluster_accept_rate)} clusters)")

    # ---- Save results ----
    results = {
        "dataset": args.dataset,
        "n_users": n_users,
        "n_items": n_items,
        "stability": {
            "fraction_unchanged_under_half_history": stability,
            "mean_distance_diff": mean_dist_diff,
            "n_users_evaluated": len(eligible),
        },
        "acceptance_vs_distance": {
            "n_sampled": len(sample),
            "overall_accept_rate": float(accepts.mean()),
            "mean_distance": float(distances.mean()),
            "spearman_rho": float(rho) if rho == rho else None,
            "spearman_p":   float(p)   if p   == p   else None,
            "binned": [
                {"distance": float(d), "accept_rate": float(a), "mean_prob": float(p)}
                for d, a, p in zip(bin_centers, bin_accept_rate, bin_mean_prob)
            ],
        },
        "per_cluster_fairness": {
            "n_clusters_evaluated": int(len(cluster_accept_rate)),
            "gini_acceptance":      float(cluster_gini),
            "min_accept_rate":      float(cluster_accept_rate.min()) if len(cluster_accept_rate) else 0.0,
            "max_accept_rate":      float(cluster_accept_rate.max()) if len(cluster_accept_rate) else 0.0,
            "mean_accept_rate":     float(cluster_accept_rate.mean()) if len(cluster_accept_rate) else 0.0,
        },
    }
    out = results_dir / "supp_a4_cluster_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")

    # ---- Figure 1: acceptance vs distance ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    a = ax[0]
    a.scatter(distances, accept_probs, s=4, alpha=0.25, color="#4C72B0", label="per user")
    if bin_centers:
        a.plot(bin_centers, bin_mean_prob, "-o", color="#C44E52", lw=2, label="binned mean")
    a.axhline(THRESHOLD, ls="--", color="gray", label=f"threshold τ={THRESHOLD}")
    a.set_xlabel("Distance to nearest cluster center  (ε)")
    a.set_ylabel("Acceptance probability  α")
    a.set_title(f"{args.dataset}: α vs ε  (Spearman ρ={rho:.2f})")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1]
    if bin_centers:
        a.plot(bin_centers, bin_accept_rate, "-s", color="#55A868", lw=2)
    a.set_xlabel("Distance to nearest cluster center  (ε)")
    a.set_ylabel("Empirical acceptance rate")
    a.set_title("Acceptance rate vs ε  (binned)")
    a.set_ylim(-0.05, 1.05)
    a.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = figures_dir / "supp_acceptance_vs_distance.pdf"
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()
    print(f"Figure: {fig_path}")

    # ---- Figure 2: cluster stability + per-cluster fairness ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    a = ax[0]
    a.bar(["Same cluster", "Different cluster"],
          [stability, 1 - stability], color=["#55A868", "#C44E52"])
    a.set_ylim(0, 1)
    a.set_ylabel("Fraction of users")
    a.set_title(f"Cluster stability under 50% history truncation\n({stability:.1%} unchanged)")

    a = ax[1]
    if len(cluster_accept_rate) > 0:
        a.hist(cluster_accept_rate, bins=20, color="#4C72B0", edgecolor="black")
        a.axvline(cluster_accept_rate.mean(), color="red", ls="--",
                  label=f"mean={cluster_accept_rate.mean():.2f}")
    a.set_xlabel("Per-cluster acceptance rate")
    a.set_ylabel("Number of clusters")
    a.set_title(f"Per-cluster acceptance distribution\n(Gini={cluster_gini:.2f})")
    a.legend(fontsize=8)

    plt.tight_layout()
    fig_path = figures_dir / "supp_cluster_stability.pdf"
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
