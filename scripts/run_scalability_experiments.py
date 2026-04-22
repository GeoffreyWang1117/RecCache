#!/usr/bin/env python3
"""
Scalability experiment: vary user count and measure latency/throughput.

Subsamples ML-1M at different user fractions (10%, 25%, 50%, 100%)
and measures per-request latency, throughput, and quality metrics.

Usage:
    conda activate reccache
    python scripts/run_scalability_experiments.py
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
USER_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
N_EVAL_USERS = 200  # per fraction


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


def subsample_data(train_data, fraction, seed=42):
    """Subsample users from training data."""
    rng = np.random.RandomState(seed)
    all_users = np.unique(train_data.user_ids)
    n_keep = max(int(len(all_users) * fraction), 10)
    keep_users = set(rng.choice(all_users, size=n_keep, replace=False).tolist())

    mask = np.array([u in keep_users for u in train_data.user_ids])
    from types import SimpleNamespace
    sub = SimpleNamespace(
        user_ids=train_data.user_ids[mask],
        item_ids=train_data.item_ids[mask],
        ratings=train_data.ratings[mask],
        n_users=train_data.n_users,
        n_items=train_data.n_items,
    )
    return sub, keep_users


def measure_latency(spec, test_users, n_warmup=10):
    """Measure per-request latency in ms."""
    # Warmup
    for uid in test_users[:n_warmup]:
        spec.recommend(uid)

    latencies = []
    for uid in test_users:
        t0 = time.perf_counter()
        spec.recommend(uid)
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


def run_experiment():
    print("=" * 70)
    print("  Scalability Experiment — ML-1M")
    print("=" * 70)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        "ml-1m", min_user_interactions=5, min_item_interactions=5,
    )
    gt = build_ground_truth(test, min_rating=4.0)

    results = {}

    for frac in USER_FRACTIONS:
        print(f"\n--- User fraction: {frac:.0%} ---")

        sub_train, keep_users = subsample_data(train, frac)
        n_active_users = len(keep_users)
        n_interactions = len(sub_train.user_ids)

        print(f"  Users: {n_active_users}, Interactions: {n_interactions}")

        # Train model
        np.random.seed(42)
        model = MatrixFactorizationRecommender(
            n_users=train.n_users, n_items=train.n_items, embedding_dim=64,
        )
        t_train_start = time.time()
        model.fit(sub_train.user_ids, sub_train.item_ids, sub_train.ratings,
                  epochs=15, verbose=False)
        train_time = time.time() - t_train_start

        item_embs = model.get_all_item_embeddings()

        # Cluster manager
        n_clusters = min(50, n_active_users // 2)
        cm = UserClusterManager(
            n_clusters=max(n_clusters, 2),
            embedding_dim=item_embs.shape[1],
            n_items=len(item_embs),
        )
        cm.set_item_embeddings(item_embs)
        cm.initialize_from_interactions(sub_train.user_ids, sub_train.item_ids, sub_train.ratings)

        # Build user history from subsampled training data
        user_hist = build_user_history(sub_train)

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

        # Warm cache
        t_warm_start = time.time()
        warm_users = list(keep_users)[:500]
        spec.warm_cache(warm_users)
        warm_time = time.time() - t_warm_start

        # Select test users
        test_users = [u for u in set(test.user_ids.tolist()) if u in gt and u in keep_users]
        test_users = test_users[:N_EVAL_USERS]

        if not test_users:
            print("  No test users available, skipping")
            continue

        # Measure latency (speculative)
        spec_latencies = measure_latency(spec, test_users)

        # Measure latency (fresh)
        fresh_latencies = []
        for uid in test_users:
            exclude = user_hist.get(uid)
            t0 = time.perf_counter()
            model.recommend(uid, n=N_RECS, exclude_items=exclude)
            fresh_latencies.append((time.perf_counter() - t0) * 1000)

        # Quality
        user_ndcgs_spec = {}
        spec_results = []
        for uid in test_users:
            if uid not in gt:
                continue
            sr = spec.recommend(uid)
            spec_results.append(sr)
            user_ndcgs_spec[uid] = RecommendationMetrics.ndcg_at_k(
                sr.items, gt[uid], N_RECS
            )

        frac_results = {
            "n_users": n_active_users,
            "n_interactions": n_interactions,
            "n_test_users": len(test_users),
            "train_time_s": train_time,
            "warm_time_s": warm_time,
            "spec_latency_ms": {
                "mean": float(np.mean(spec_latencies)),
                "p50": float(np.percentile(spec_latencies, 50)),
                "p95": float(np.percentile(spec_latencies, 95)),
                "p99": float(np.percentile(spec_latencies, 99)),
            },
            "fresh_latency_ms": {
                "mean": float(np.mean(fresh_latencies)),
                "p50": float(np.percentile(fresh_latencies, 50)),
                "p95": float(np.percentile(fresh_latencies, 95)),
                "p99": float(np.percentile(fresh_latencies, 99)),
            },
            "spec_throughput_rps": 1000.0 / np.mean(spec_latencies),
            "fresh_throughput_rps": 1000.0 / np.mean(fresh_latencies),
            "ndcg": float(np.mean(list(user_ndcgs_spec.values()))) if user_ndcgs_spec else 0.0,
            "accept_rate": SpeculativeMetrics.acceptance_rate(spec_results) if spec_results else 0.0,
        }
        results[f"{frac:.0%}"] = frac_results

        print(f"  Train: {train_time:.1f}s, Warm: {warm_time:.1f}s")
        print(f"  Spec latency:  mean={frac_results['spec_latency_ms']['mean']:.2f}ms, "
              f"p95={frac_results['spec_latency_ms']['p95']:.2f}ms")
        print(f"  Fresh latency: mean={frac_results['fresh_latency_ms']['mean']:.2f}ms, "
              f"p95={frac_results['fresh_latency_ms']['p95']:.2f}ms")
        print(f"  Throughput: spec={frac_results['spec_throughput_rps']:.0f} rps, "
              f"fresh={frac_results['fresh_throughput_rps']:.0f} rps")
        print(f"  NDCG: {frac_results['ndcg']:.4f}, Accept: {frac_results['accept_rate']:.1%}")

    return results


def make_figures(results, output_dir):
    """Generate scalability figures."""
    fracs = sorted(results.keys(), key=lambda x: float(x.strip('%')) / 100)
    n_users = [results[f]["n_users"] for f in fracs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. Latency vs users
    spec_lat = [results[f]["spec_latency_ms"]["mean"] for f in fracs]
    fresh_lat = [results[f]["fresh_latency_ms"]["mean"] for f in fracs]
    spec_p95 = [results[f]["spec_latency_ms"]["p95"] for f in fracs]
    fresh_p95 = [results[f]["fresh_latency_ms"]["p95"] for f in fracs]

    axes[0].plot(n_users, spec_lat, "o-", label="Spec (mean)", color="#3498db")
    axes[0].plot(n_users, spec_p95, "s--", label="Spec (p95)", color="#3498db", alpha=0.5)
    axes[0].plot(n_users, fresh_lat, "o-", label="Fresh (mean)", color="#e74c3c")
    axes[0].plot(n_users, fresh_p95, "s--", label="Fresh (p95)", color="#e74c3c", alpha=0.5)
    axes[0].set_xlabel("Number of Users")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency vs Scale")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 2. Throughput vs users
    spec_thr = [results[f]["spec_throughput_rps"] for f in fracs]
    fresh_thr = [results[f]["fresh_throughput_rps"] for f in fracs]

    axes[1].plot(n_users, spec_thr, "o-", label="Speculative", color="#3498db")
    axes[1].plot(n_users, fresh_thr, "o-", label="Fresh", color="#e74c3c")
    axes[1].set_xlabel("Number of Users")
    axes[1].set_ylabel("Throughput (req/s)")
    axes[1].set_title("Throughput vs Scale")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3. Quality vs users
    ndcgs = [results[f]["ndcg"] for f in fracs]
    accept_rates = [results[f]["accept_rate"] for f in fracs]

    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(n_users, ndcgs, "o-", label="NDCG@10", color="#2ecc71")
    l2 = ax3_twin.plot(n_users, accept_rates, "s-", label="Accept Rate", color="#9b59b6")
    ax3.set_xlabel("Number of Users")
    ax3.set_ylabel("NDCG@10", color="#2ecc71")
    ax3_twin.set_ylabel("Accept Rate", color="#9b59b6")
    ax3.set_title("Quality vs Scale")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"scalability.{ext}", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    results = run_experiment()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "scalability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    make_figures(results, figures_dir)
    print(f"Figure saved to paper/figures/scalability.{{pdf,png}}")


if __name__ == "__main__":
    main()
