#!/usr/bin/env python3
"""
Competitor Baseline Experiments — RecCache vs Strong Baselines.

S9  — BiLD-style low-rank draft model (trained MF d=16 as draft, MF d=64 as target).
      Tests: training-free cluster-cache vs trained low-rank draft (BiLD/SpecDec paradigm).

S10 — LASER-adapted relaxed verification (mean-ratio instead of product-ratio).
      Tests: strict score-ratio (ours) vs relaxed acceptance (LASER-style).

Both experiments run on ML-1M (primary benchmark) and optionally other datasets.

Usage:
    conda activate reccache
    python scripts/run_competitor_baselines.py               # all experiments
    python scripts/run_competitor_baselines.py --experiment s9
    python scripts/run_competitor_baselines.py --experiment s10
    python scripts/run_competitor_baselines.py --dataset ml-1m
"""

import sys
import argparse
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
from reccache.models.acceptance import (
    ScoreRatioAcceptanceCriterion,
    LASERRelaxedAcceptanceCriterion,
    LowRankDraftAcceptanceCriterion,
)
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import (
    RecommendationMetrics,
    SpeculativeMetrics,
    compute_coverage,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50,
        "embedding_dim": 64,
        "epochs": 15,
    },
    "amazon-electronics": {
        "max_samples": 1000000,
        "min_user": 3, "min_item": 3,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50,
        "embedding_dim": 64,
        "epochs": 15,
    },
}

N_RUNS = 3
N_TEST_USERS = 500
TOP_K = 3
THRESHOLD = 0.35
N_RECS = 10
LOW_RANK_DIM = 16


def set_seed(seed: int):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data + Model utilities
# ---------------------------------------------------------------------------
def load_and_preprocess(dataset_name: str, data_dir: str):
    cfg = DATASET_CONFIGS[dataset_name]
    loader = DataLoader(data_dir)
    train, val, test = loader.load_dataset(
        dataset_name,
        max_samples=cfg["max_samples"],
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
    )
    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))
    return train, val, test, dict(user_history), cfg


def train_mf(train, cfg, embedding_dim=None):
    dim = embedding_dim or cfg["embedding_dim"]
    model = MatrixFactorizationRecommender(
        n_users=train.n_users,
        n_items=train.n_items,
        embedding_dim=dim,
    )
    model.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"])
    return model


def build_cluster_manager(train, model, cfg, seed, embedding_dim=None):
    np.random.seed(seed)
    item_embs = model.get_all_item_embeddings()
    dim = embedding_dim or cfg["embedding_dim"]
    cm = UserClusterManager(
        n_clusters=cfg["n_clusters"],
        embedding_dim=dim,
    )
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
    return cm, item_embs


def get_test_users(test, n_users, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(test.user_ids)
    if len(unique) > n_users:
        unique = rng.choice(unique, size=n_users, replace=False)
    return unique.tolist()


def compute_metrics(results, test_lookup, item_embs, n_recs=10):
    """Aggregate NDCG, accept rate, speedup, MCG, coverage."""
    ndcgs, hrs, latencies, accepted_flags, cluster_ranks = [], [], [], [], []
    for r in results:
        uid = r.user_id
        true_items = test_lookup.get(uid, [])
        if not true_items:
            continue
        true_set = set(true_items)
        ndcgs.append(RecommendationMetrics.ndcg_at_k(r.items, true_set, k=n_recs))
        hrs.append(RecommendationMetrics.hit_rate(r.items, true_set, k=n_recs))
        latencies.append(r.latency_ms)
        accepted_flags.append(int(r.accepted))
        if r.accepted:
            cluster_ranks.append(r.accepted_cluster_rank)

    accept_rate = float(np.mean(accepted_flags)) if accepted_flags else 0.0
    mcg = float(np.mean([r > 0 for r in cluster_ranks])) if cluster_ranks else 0.0

    # Empirical speedup: mean fresh latency / mean request latency
    # (approximated here as 1/(1-accept*hit_speedup_factor))
    # Use actual latency distribution
    all_lat = np.array(latencies)
    mean_lat = float(np.mean(all_lat)) if len(all_lat) else 1.0

    cov = compute_coverage({r.user_id: r.items for r in results}, item_embs.shape[0])

    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hr": float(np.mean(hrs)) if hrs else 0.0,
        "accept_rate": accept_rate,
        "mean_latency_ms": mean_lat,
        "mcg": mcg,
        "coverage": cov,
        "n_evaluated": len(ndcgs),
    }


def run_speculative_system(
    model,
    cluster_manager,
    criterion,
    item_embs,
    test_users,
    user_history,
    train,
    config: SpeculativeConfig,
):
    """Warm cache and evaluate one speculative system configuration."""
    spec = SpeculativeRecommender(
        recommender=model,
        cluster_manager=cluster_manager,
        acceptance_criterion=criterion,
        config=config,
        item_embeddings=item_embs,
        user_history=user_history,
    )
    spec.warm_cache(list(range(train.n_users)))

    results = []
    for uid in test_users:
        result = spec.recommend(user_id=uid)
        results.append(result)

    return results, spec.get_stats()


# ---------------------------------------------------------------------------
# S9: BiLD-style Low-Rank Draft Comparison
# ---------------------------------------------------------------------------
def run_s9(dataset_name: str, data_dir: str, figures_dir: Path, seed: int):
    """
    Compare RecCache (cluster-cache) vs BiLD-style (trained low-rank draft).

    Both use the full MF(d=64) as target. RecCache uses cluster centroid as draft.
    BiLD uses a separately-trained MF(d=16) user embedding as draft.
    """
    print(f"\n[S9] BiLD-style low-rank draft | {dataset_name} | seed={seed}")

    train, val, test, user_history, cfg = load_and_preprocess(dataset_name, data_dir)

    min_rating = cfg.get("min_rating_gt", 4.0)
    test_lookup = defaultdict(list)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= min_rating:
            test_lookup[int(uid)].append(int(iid))

    set_seed(seed)

    # Train full model (d=64) — target model for both baselines
    print("  Training full MF (d=64) ...")
    full_model = train_mf(train, cfg, embedding_dim=64)

    # Train low-rank draft model (d=16)
    print("  Training low-rank MF (d=16) ...")
    lr_model = train_mf(train, cfg, embedding_dim=LOW_RANK_DIM)

    # Build cluster manager on full model embeddings
    cm, item_embs = build_cluster_manager(train, full_model, cfg, seed)

    # Extract low-rank user embeddings for draft criterion
    import torch
    with torch.no_grad():
        lr_user_embs_tensor = lr_model.user_embeddings.weight.cpu().numpy()  # (n_users, 16)
    draft_user_embs = {i: lr_user_embs_tensor[i] for i in range(len(lr_user_embs_tensor))}

    test_users = get_test_users(test, N_TEST_USERS, seed)

    config = SpeculativeConfig(
        top_k_clusters=TOP_K,
        acceptance_threshold=THRESHOLD,
        n_recs=N_RECS,
        use_pool_retrieval=False,
    )

    # Fresh baseline
    print("  Fresh (full model)...")
    from reccache.models.speculative import SpeculativeResult
    fresh_results = []
    for uid in test_users:
        exclude = user_history.get(uid)
        t_start = time.time()
        items = list(full_model.recommend(uid, n=N_RECS, exclude_items=exclude))
        lat = (time.time() - t_start) * 1000
        fresh_results.append(SpeculativeResult(
            user_id=uid, items=items, accepted=False,
            acceptance_prob=0.0, accepted_cluster_id=None,
            accepted_cluster_rank=-1, latency_ms=lat, phase="fresh",
        ))

    fresh_metrics = compute_metrics(fresh_results, test_lookup, item_embs, N_RECS)
    fresh_ndcg = fresh_metrics["ndcg"]
    print(f"    Fresh NDCG={fresh_ndcg:.4f}, lat={fresh_metrics['mean_latency_ms']:.2f}ms")

    # RecCache (cluster-centroid draft, score-ratio acceptance)
    print("  RecCache (cluster-centroid draft, score-ratio)...")
    sr_criterion = ScoreRatioAcceptanceCriterion(threshold=THRESHOLD)
    reccache_results, reccache_stats = run_speculative_system(
        full_model, cm, sr_criterion, item_embs,
        test_users, user_history, train, config
    )
    reccache_metrics = compute_metrics(reccache_results, test_lookup, item_embs, N_RECS)
    reccache_metrics["retention"] = reccache_metrics["ndcg"] / fresh_ndcg if fresh_ndcg > 0 else 0
    print(f"    RecCache NDCG={reccache_metrics['ndcg']:.4f} "
          f"(ret={reccache_metrics['retention']:.0%}), "
          f"accept={reccache_metrics['accept_rate']:.0%}, "
          f"MCG={reccache_metrics['mcg']:.0%}")

    # BiLD-style (low-rank draft, score-ratio acceptance)
    print("  BiLD-style (low-rank MF d=16 draft, score-ratio)...")
    lr_criterion = LowRankDraftAcceptanceCriterion(
        draft_user_embeddings=draft_user_embs,
        threshold=THRESHOLD,
    )

    # For BiLD-style, we still need cluster cache (draft items come from cluster)
    # but acceptance uses low-rank user embedding instead of cluster center
    # Patch: wrap criterion to pass user_id into compute_acceptance
    class _LRSpecRec(SpeculativeRecommender):
        def recommend(self, user_id):
            import time as _time
            start = _time.time()
            self._stats["total_requests"] += 1

            candidates = self.cluster_manager.get_nearest_clusters(
                user_id, top_k=self.config.top_k_clusters
            )
            user_emb = self.cluster_manager.get_user_embedding(user_id)

            best_result = None
            best_rank = -1
            best_items = None

            for rank, cand in enumerate(candidates):
                cached = self._cluster_cache.get(cand.cluster_id)
                if cached is None:
                    continue
                # Pass user_id for low-rank lookup
                result = self.acceptance_criterion.compute_acceptance(
                    user_embedding=user_emb,
                    cluster_center=cand.center,
                    cluster_id=cand.cluster_id,
                    cached_item_ids=cached,
                    item_embeddings=self.item_embeddings,
                    user_id=user_id,
                )
                self._stats["acceptance_probs"].append(result.acceptance_prob)
                if result.acceptance_prob >= self.config.acceptance_threshold:
                    if best_result is None or result.acceptance_prob > best_result.acceptance_prob:
                        best_result = result
                        best_rank = rank
                        best_items = list(cached)

            from reccache.models.speculative import SpeculativeResult
            if best_result is not None and best_items is not None:
                self._stats["accepted"] += 1
                self._stats["cluster_rank_counts"][best_rank] += 1
                elapsed = (_time.time() - start) * 1000
                return SpeculativeResult(
                    user_id=user_id, items=best_items[:self.config.n_recs],
                    accepted=True, acceptance_prob=best_result.acceptance_prob,
                    accepted_cluster_id=best_result.cluster_id,
                    accepted_cluster_rank=best_rank,
                    latency_ms=elapsed, phase="accept",
                )

            exclude = self.user_history.get(user_id)
            items = list(self.recommender.recommend(user_id, n=self.config.n_recs, exclude_items=exclude))
            if candidates:
                self._cluster_cache[candidates[0].cluster_id] = list(items)
            self._stats["rejected"] += 1
            elapsed = (_time.time() - start) * 1000
            return SpeculativeResult(
                user_id=user_id, items=items[:self.config.n_recs],
                accepted=False, acceptance_prob=0.0,
                accepted_cluster_id=None, accepted_cluster_rank=-1,
                latency_ms=elapsed, phase="residual",
            )

    bild_spec = _LRSpecRec(
        recommender=full_model,
        cluster_manager=cm,
        acceptance_criterion=lr_criterion,
        config=config,
        item_embeddings=item_embs,
        user_history=user_history,
    )
    bild_spec.warm_cache(list(range(train.n_users)))
    bild_results = [bild_spec.recommend(uid) for uid in test_users]

    bild_metrics = compute_metrics(bild_results, test_lookup, item_embs, N_RECS)
    bild_metrics["retention"] = bild_metrics["ndcg"] / fresh_ndcg if fresh_ndcg > 0 else 0
    print(f"    BiLD-style NDCG={bild_metrics['ndcg']:.4f} "
          f"(ret={bild_metrics['retention']:.0%}), "
          f"accept={bild_metrics['accept_rate']:.0%}, "
          f"MCG={bild_metrics['mcg']:.0%}")

    return {
        "fresh": fresh_metrics,
        "reccache_k3": reccache_metrics,
        "bild_lr16": bild_metrics,
        "fresh_ndcg": fresh_ndcg,
    }


# ---------------------------------------------------------------------------
# S10: LASER-Adapted Relaxed Verification Comparison
# ---------------------------------------------------------------------------
def run_s10(dataset_name: str, data_dir: str, figures_dir: Path, seed: int):
    """
    Compare acceptance criteria on same cluster-cache setup:
      1. ScoreRatio (ours — product rule)
      2. LASER-Relaxed (mean rule)
      3. LASER-Relaxed (intermediate: relaxation=0.5)
    """
    print(f"\n[S10] LASER-adapted relaxed verification | {dataset_name} | seed={seed}")

    train, val, test, user_history, cfg = load_and_preprocess(dataset_name, data_dir)

    min_rating = cfg.get("min_rating_gt", 4.0)
    test_lookup = defaultdict(list)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= min_rating:
            test_lookup[int(uid)].append(int(iid))

    set_seed(seed)

    print("  Training MF (d=64)...")
    model = train_mf(train, cfg, embedding_dim=64)
    cm, item_embs = build_cluster_manager(train, model, cfg, seed)

    test_users = get_test_users(test, N_TEST_USERS, seed)

    config = SpeculativeConfig(
        top_k_clusters=TOP_K,
        acceptance_threshold=THRESHOLD,
        n_recs=N_RECS,
        use_pool_retrieval=False,
    )

    # Fresh
    from reccache.models.speculative import SpeculativeResult as _SR
    fresh_results = []
    for uid in test_users:
        exclude = user_history.get(uid)
        t_start = time.time()
        items = list(model.recommend(uid, n=N_RECS, exclude_items=exclude))
        lat = (time.time() - t_start) * 1000
        fresh_results.append(_SR(
            user_id=uid, items=items, accepted=False,
            acceptance_prob=0.0, accepted_cluster_id=None,
            accepted_cluster_rank=-1, latency_ms=lat, phase="fresh",
        ))

    fresh_metrics = compute_metrics(fresh_results, test_lookup, item_embs, N_RECS)
    fresh_ndcg = fresh_metrics["ndcg"]
    print(f"    Fresh NDCG={fresh_ndcg:.4f}")

    criteria = {
        "ScoreRatio (ours)": ScoreRatioAcceptanceCriterion(threshold=THRESHOLD),
        "LASER-Relaxed (mean)": LASERRelaxedAcceptanceCriterion(threshold=THRESHOLD, relaxation=0.0),
        "LASER-Relaxed (r=0.5)": LASERRelaxedAcceptanceCriterion(threshold=THRESHOLD, relaxation=0.5),
    }

    metrics_by_criterion = {}
    for name, criterion in criteria.items():
        results, stats = run_speculative_system(
            model, cm, criterion, item_embs,
            test_users, user_history, train, config,
        )
        m = compute_metrics(results, test_lookup, item_embs, N_RECS)
        m["retention"] = m["ndcg"] / fresh_ndcg if fresh_ndcg > 0 else 0
        metrics_by_criterion[name] = m
        print(f"    {name}: NDCG={m['ndcg']:.4f} (ret={m['retention']:.0%}), "
              f"accept={m['accept_rate']:.0%}")

    return {
        "fresh": fresh_metrics,
        "by_criterion": metrics_by_criterion,
        "fresh_ndcg": fresh_ndcg,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_s9(all_results, figures_dir: Path):
    """Bar chart: Fresh / RecCache / BiLD-style across datasets."""
    datasets = list(all_results.keys())
    methods = ["Fresh", "RecCache K=3", "BiLD (d=16)"]
    keys = ["fresh", "reccache_k3", "bild_lr16"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics_to_plot = [("ndcg", "NDCG@10"), ("accept_rate", "Acceptance Rate"), ("mcg", "MCG")]

    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        vals = []
        for k in keys:
            row = []
            for ds in datasets:
                r = all_results[ds]
                if k == "fresh" and metric_key in ("accept_rate", "mcg"):
                    row.append(0.0)
                else:
                    row.append(r[k].get(metric_key, 0.0))
            vals.append(row)

        x = np.arange(len(datasets))
        width = 0.25
        for i, (method, val) in enumerate(zip(methods, vals)):
            ax.bar(x + i * width, val, width, label=method)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric_label)
        ax.set_title(f"S9: {metric_label}")
        ax.set_xticks(x + width)
        ax.set_xticklabels([ds.replace("amazon-", "A-") for ds in datasets], rotation=15, ha="right")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = figures_dir / "s9_competitor_lowrank.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")


def plot_s10(all_results, figures_dir: Path):
    """Scatter: acceptance rate vs NDCG retention for each criterion."""
    datasets = list(all_results.keys())
    markers = ["o", "s", "^", "D"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        r = all_results[ds]
        criteria = list(r["by_criterion"].keys())
        for i, name in enumerate(criteria):
            m = r["by_criterion"][name]
            ax.scatter(
                m["accept_rate"], m["retention"],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                s=100, label=name, zorder=3
            )
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="Fresh (1.0)")
        ax.set_xlabel("Acceptance Rate")
        ax.set_title(ds.replace("amazon-", "A-"))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("NDCG Retention")
    fig.suptitle("S10: Criterion Comparison — Acceptance Rate vs NDCG Retention", fontsize=11)
    plt.tight_layout()
    path = figures_dir / "s10_laser_criterion.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="+", default=["s9", "s10"],
                        choices=["s9", "s10"],
                        help="Experiments to run")
    parser.add_argument("--dataset", nargs="+", default=["ml-1m"],
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Datasets to evaluate on")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True, parents=True)

    experiments = [e.lower() for e in args.experiment]

    # -------------------------------------------------------------------------
    # S9 — BiLD-style low-rank draft
    # -------------------------------------------------------------------------
    if "s9" in experiments:
        print("\n" + "=" * 60)
        print("S9: BiLD-Style Low-Rank Draft Comparison")
        print("=" * 60)

        s9_all = {}
        for dataset in args.dataset:
            seed_results = []
            for seed in args.seeds:
                r = run_s9(dataset, args.data_dir, figures_dir, seed)
                seed_results.append(r)

            # Average over seeds
            def avg(key, subkey=None):
                if subkey:
                    vals = [sr[key][subkey] for sr in seed_results if subkey in sr[key]]
                else:
                    vals = [sr[key] for sr in seed_results]
                return float(np.mean(vals)) if vals else 0.0

            s9_all[dataset] = {
                "fresh": {k: avg("fresh", k) for k in seed_results[0]["fresh"]},
                "reccache_k3": {k: avg("reccache_k3", k) for k in seed_results[0]["reccache_k3"]},
                "bild_lr16": {k: avg("bild_lr16", k) for k in seed_results[0]["bild_lr16"]},
            }

        # Print summary table
        print("\n  S9 Summary (averaged over seeds):")
        print(f"  {'Dataset':<25} {'Method':<22} {'NDCG@10':>8} {'Retention':>10} {'Accept':>8} {'MCG':>6}")
        print("  " + "-" * 85)
        for ds, r in s9_all.items():
            fresh_ndcg = r["fresh"]["ndcg"]
            for method, key in [("Fresh", "fresh"), ("RecCache K=3", "reccache_k3"), ("BiLD d=16", "bild_lr16")]:
                m = r[key]
                retention = m["ndcg"] / fresh_ndcg if fresh_ndcg > 0 and method != "Fresh" else 1.0
                accept = m.get("accept_rate", 0) if method != "Fresh" else float("nan")
                mcg = m.get("mcg", 0) if method != "Fresh" else float("nan")
                print(f"  {ds:<25} {method:<22} {m['ndcg']:>8.4f} {retention:>10.1%} "
                      f"{accept:>8.1%} {mcg:>6.1%}" if method != "Fresh" else
                      f"  {ds:<25} {method:<22} {m['ndcg']:>8.4f} {'—':>10} {'—':>8} {'—':>6}")

        plot_s9(s9_all, figures_dir)

        out_path = results_dir / "s9_competitor_lowrank.json"
        with open(out_path, "w") as f:
            json.dump(s9_all, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    # -------------------------------------------------------------------------
    # S10 — LASER-adapted relaxed verification
    # -------------------------------------------------------------------------
    if "s10" in experiments:
        print("\n" + "=" * 60)
        print("S10: LASER-Adapted Relaxed Verification Comparison")
        print("=" * 60)

        s10_all = {}
        for dataset in args.dataset:
            seed_results = []
            for seed in args.seeds:
                r = run_s10(dataset, args.data_dir, figures_dir, seed)
                seed_results.append(r)

            # Average over seeds
            criteria = list(seed_results[0]["by_criterion"].keys())
            fresh_avg = {k: float(np.mean([sr["fresh"][k] for sr in seed_results]))
                         for k in seed_results[0]["fresh"]}
            by_criterion_avg = {}
            for name in criteria:
                by_criterion_avg[name] = {
                    k: float(np.mean([sr["by_criterion"][name][k] for sr in seed_results]))
                    for k in seed_results[0]["by_criterion"][name]
                }

            s10_all[dataset] = {
                "fresh": fresh_avg,
                "by_criterion": by_criterion_avg,
            }

        # Print summary table
        print("\n  S10 Summary (averaged over seeds):")
        print(f"  {'Dataset':<25} {'Criterion':<30} {'NDCG@10':>8} {'Retention':>10} {'Accept':>8}")
        print("  " + "-" * 85)
        for ds, r in s10_all.items():
            fresh_ndcg = r["fresh"]["ndcg"]
            print(f"  {ds:<25} {'Fresh':<30} {fresh_ndcg:>8.4f} {'—':>10} {'—':>8}")
            for name, m in r["by_criterion"].items():
                retention = m["ndcg"] / fresh_ndcg if fresh_ndcg > 0 else 0
                print(f"  {'':25} {name:<30} {m['ndcg']:>8.4f} {retention:>10.1%} {m['accept_rate']:>8.1%}")

        plot_s10(s10_all, figures_dir)

        out_path = results_dir / "s10_laser_criterion.json"
        with open(out_path, "w") as f:
            json.dump(s10_all, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
