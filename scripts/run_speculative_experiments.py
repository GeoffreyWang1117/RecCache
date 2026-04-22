#!/usr/bin/env python3
"""
Speculative Recommendation experiments for RecCache.

Evaluates recommendation caching reframed as speculative serving
(analogous to speculative decoding in LLM inference).

Groups:
  S1 — Single vs Multi-cluster Speculation (K sweep)
  S2 — Acceptance Criterion Comparison
  S3 — Acceptance Threshold Sweep (Pareto front)
  S4 — Comparison to Baselines

Usage:
    python scripts/run_speculative_experiments.py                # all groups
    python scripts/run_speculative_experiments.py --group S1     # single group
    python scripts/run_speculative_experiments.py --group S1 S3  # multiple groups
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
    CosineAcceptanceCriterion,
    ScoreRatioAcceptanceCriterion,
    HeuristicAcceptanceCriterion,
)
from reccache.models.speculative import (
    SpeculativeRecommender,
    SpeculativeConfig,
)
from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker
from reccache.evaluation.metrics import (
    RecommendationMetrics,
    SpeculativeMetrics,
    compute_ild,
    compute_coverage,
    compute_tail_user_ndcg,
)


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
    "amazon-movies": {
        "max_samples": 1000000,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
}

GROUP_DATASETS = {
    "S1": ["ml-1m", "amazon-movies"],
    "S2": ["ml-1m", "amazon-movies"],
    "S3": ["ml-1m", "amazon-movies"],
    "S4": ["ml-1m", "amazon-movies"],
}

N_RUNS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)


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


def build_interaction_counts(train_data):
    counts = defaultdict(int)
    for uid in train_data.user_ids:
        counts[int(uid)] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Cached data / model loading (shared across groups)
# ---------------------------------------------------------------------------
_data_cache = {}


def load_dataset_cached(name):
    if name in _data_cache:
        return _data_cache[name]

    cfg = DATASET_CONFIGS[name]
    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        name,
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
        max_samples=cfg["max_samples"],
    )
    gt = build_ground_truth(test, min_rating=cfg["min_rating_gt"])

    print(f"[{name}] {train.n_users} users, {train.n_items} items, "
          f"{len(train.user_ids)} interactions, {len(gt)} GT users")

    _data_cache[name] = (train, val, test, gt, cfg)
    return _data_cache[name]


_model_cache = {}


def get_trained_model(name, train, cfg):
    if name in _model_cache:
        return _model_cache[name]

    print(f"  Training MF on {name}...", flush=True)
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items,
        embedding_dim=cfg["embedding_dim"],
    )
    model.fit(
        train.user_ids, train.item_ids, train.ratings,
        epochs=cfg["epochs"], verbose=True,
    )
    _model_cache[name] = model
    return model


_cluster_cache_store = {}


def get_cluster_manager(name, train, item_embeddings, cfg):
    if name in _cluster_cache_store:
        return _cluster_cache_store[name]

    n_clusters = min(cfg["n_clusters"], train.n_users // 2)
    cm = UserClusterManager(
        n_clusters=n_clusters,
        embedding_dim=item_embeddings.shape[1],
        n_items=len(item_embeddings),
    )
    cm.set_item_embeddings(item_embeddings)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
    _cluster_cache_store[name] = cm
    return cm


# ---------------------------------------------------------------------------
# Core evaluation helper
# ---------------------------------------------------------------------------
def evaluate_speculative(
    recommender,
    cluster_manager,
    acceptance_criterion,
    ground_truth,
    user_ids,
    item_embeddings,
    interaction_counts=None,
    top_k_clusters=3,
    threshold=0.5,
    n_recs=10,
    warm_user_ids=None,
    reranker=None,
):
    """
    Create a SpeculativeRecommender, warm its cache, evaluate on user_ids,
    and return aggregated metrics.
    """
    config = SpeculativeConfig(
        top_k_clusters=top_k_clusters,
        acceptance_threshold=threshold,
        n_recs=n_recs,
        rerank_on_accept=reranker is not None,
    )
    spec = SpeculativeRecommender(
        recommender=recommender,
        cluster_manager=cluster_manager,
        acceptance_criterion=acceptance_criterion,
        config=config,
        reranker=reranker,
        item_embeddings=item_embeddings,
    )

    # Warm cache
    if warm_user_ids is not None:
        spec.warm_cache(warm_user_ids)

    # Evaluate
    results = []
    recommendations = {}
    user_ndcgs = {}

    for uid in user_ids:
        if uid not in ground_truth:
            continue
        sr = spec.recommend(uid)
        results.append(sr)
        recommendations[uid] = sr.items
        user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(
            sr.items, ground_truth[uid], n_recs
        )

    if not results:
        return {
            "ndcg": 0.0, "accept_rate": 0.0, "coverage": 0.0,
            "ild": 0.0, "speedup": 1.0, "tail_ndcg": 0.0,
            "multi_cluster_gain": 0.0,
        }

    # Aggregate
    accept_rate = SpeculativeMetrics.acceptance_rate(results)
    speedup = SpeculativeMetrics.speedup_estimate(results)
    mc_gain = SpeculativeMetrics.multi_cluster_gain(results)

    ilds = [compute_ild(recs, item_embeddings) for recs in recommendations.values()]
    coverage = compute_coverage(recommendations, item_embeddings.shape[0])

    tail_ndcg = 0.0
    if interaction_counts is not None:
        tail_ndcg = compute_tail_user_ndcg(user_ndcgs, interaction_counts, threshold=5)

    return {
        "ndcg": float(np.mean(list(user_ndcgs.values()))),
        "accept_rate": accept_rate,
        "coverage": coverage,
        "ild": float(np.mean(ilds)),
        "speedup": speedup,
        "tail_ndcg": tail_ndcg,
        "multi_cluster_gain": mc_gain,
    }


# ===========================================================================
# Group S1: Single vs Multi-cluster Speculation
# ===========================================================================
def run_group_s1(datasets=None):
    """K in {1, 3, 5, 7} nearest clusters.

    Uses ScoreRatioAcceptanceCriterion (the paper's speculative decoding
    analogy) because item-level acceptance genuinely varies across clusters:
    ~60% of users prefer a non-nearest cluster's cached items.  Cosine
    acceptance cannot show multi-cluster gain since, with normalised
    embeddings, rank 0 always dominates.
    """
    datasets = datasets or GROUP_DATASETS["S1"]
    print(f"\n{'='*70}")
    print("GROUP S1: Single vs Multi-cluster Speculation")
    print(f"{'='*70}\n")

    k_values = [1, 3, 5, 7]
    threshold = 0.35  # calibrated: ScoreRatio alpha median ~0.45 at temp=1.0
    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg)
        item_embs = model.get_all_item_embeddings()
        cm = get_cluster_manager(ds_name, train, item_embs, cfg)
        int_counts = build_interaction_counts(train)

        sample_users = list(gt.keys())[:500]
        warm_users = list(gt.keys())

        ds_results = {}

        for k in k_values:
            print(f"  K={k}", flush=True)
            run_metrics = defaultdict(list)

            for run_i in range(N_RUNS):
                set_seed(42 + run_i)
                np.random.shuffle(sample_users)

                criterion = ScoreRatioAcceptanceCriterion(
                    threshold=threshold, temperature=1.0,
                )
                m = evaluate_speculative(
                    recommender=model, cluster_manager=cm,
                    acceptance_criterion=criterion,
                    ground_truth=gt, user_ids=sample_users,
                    item_embeddings=item_embs,
                    interaction_counts=int_counts,
                    top_k_clusters=k, threshold=threshold,
                    warm_user_ids=warm_users,
                )
                for metric, val in m.items():
                    run_metrics[metric].append(val)

            ds_results[f"K={k}"] = {
                metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for metric, vals in run_metrics.items()
            }
            r = ds_results[f"K={k}"]
            print(f"    NDCG={r['ndcg']['mean']:.4f}  "
                  f"Accept={r['accept_rate']['mean']:.4f}  "
                  f"MCGain={r['multi_cluster_gain']['mean']:.4f}  "
                  f"Speedup={r['speedup']['mean']:.2f}x")

        results[ds_name] = ds_results

    return results


# ===========================================================================
# Group S2: Acceptance Criterion Comparison
# ===========================================================================
def run_group_s2(datasets=None):
    """K=3, compare Cosine vs Score-Ratio vs Heuristic (QA)."""
    datasets = datasets or GROUP_DATASETS["S2"]
    print(f"\n{'='*70}")
    print("GROUP S2: Acceptance Criterion Comparison")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg)
        item_embs = model.get_all_item_embeddings()
        cm = get_cluster_manager(ds_name, train, item_embs, cfg)
        int_counts = build_interaction_counts(train)

        sample_users = list(gt.keys())[:500]
        warm_users = list(gt.keys())

        criteria = {
            "Cosine": CosineAcceptanceCriterion(threshold=0.5),
            "ScoreRatio": ScoreRatioAcceptanceCriterion(threshold=0.3),
            "Heuristic (QA)": HeuristicAcceptanceCriterion(
                QualityPredictor(hidden_dim=32, quality_threshold=0.15),
                threshold=0.5,
            ),
        }

        # Use threshold per criterion (tuned)
        thresholds = {"Cosine": 0.5, "ScoreRatio": 0.3, "Heuristic (QA)": 0.5}

        ds_results = {}

        for cname, criterion in criteria.items():
            print(f"  Criterion: {cname}", flush=True)
            run_metrics = defaultdict(list)

            for run_i in range(N_RUNS):
                set_seed(42 + run_i)
                np.random.shuffle(sample_users)

                m = evaluate_speculative(
                    recommender=model, cluster_manager=cm,
                    acceptance_criterion=criterion,
                    ground_truth=gt, user_ids=sample_users,
                    item_embeddings=item_embs,
                    interaction_counts=int_counts,
                    top_k_clusters=3,
                    threshold=thresholds[cname],
                    warm_user_ids=warm_users,
                )
                for metric, val in m.items():
                    run_metrics[metric].append(val)

            ds_results[cname] = {
                metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for metric, vals in run_metrics.items()
            }
            r = ds_results[cname]
            print(f"    NDCG={r['ndcg']['mean']:.4f}  "
                  f"Accept={r['accept_rate']['mean']:.4f}  "
                  f"ILD={r['ild']['mean']:.4f}  "
                  f"Cov={r['coverage']['mean']:.4f}")

        results[ds_name] = ds_results

    return results


# ===========================================================================
# Group S3: Acceptance Threshold Sweep (Pareto front)
# ===========================================================================
def run_group_s3(datasets=None):
    """Threshold sweep with ScoreRatio (K=3).

    Uses ScoreRatioAcceptanceCriterion so that multi-cluster dynamics are
    visible in the Pareto front — analogous to "draft length vs speedup"
    plots in speculative decoding papers.
    """
    datasets = datasets or GROUP_DATASETS["S3"]
    print(f"\n{'='*70}")
    print("GROUP S3: Acceptance Threshold Sweep (ScoreRatio, K=3)")
    print(f"{'='*70}\n")

    threshold_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg)
        item_embs = model.get_all_item_embeddings()
        cm = get_cluster_manager(ds_name, train, item_embs, cfg)
        int_counts = build_interaction_counts(train)

        sample_users = list(gt.keys())[:500]
        warm_users = list(gt.keys())

        ds_results = {}

        for thr in threshold_values:
            print(f"  threshold={thr:.2f}", flush=True)
            run_metrics = defaultdict(list)

            for run_i in range(N_RUNS):
                set_seed(42 + run_i)
                np.random.shuffle(sample_users)

                criterion = ScoreRatioAcceptanceCriterion(
                    threshold=thr, temperature=1.0,
                )
                m = evaluate_speculative(
                    recommender=model, cluster_manager=cm,
                    acceptance_criterion=criterion,
                    ground_truth=gt, user_ids=sample_users,
                    item_embeddings=item_embs,
                    interaction_counts=int_counts,
                    top_k_clusters=3, threshold=thr,
                    warm_user_ids=warm_users,
                )
                for metric, val in m.items():
                    run_metrics[metric].append(val)

            ds_results[f"thr={thr:.2f}"] = {
                metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for metric, vals in run_metrics.items()
            }
            r = ds_results[f"thr={thr:.2f}"]
            print(f"    NDCG={r['ndcg']['mean']:.4f}  "
                  f"Accept={r['accept_rate']['mean']:.4f}  "
                  f"MCGain={r['multi_cluster_gain']['mean']:.4f}  "
                  f"Speedup={r['speedup']['mean']:.2f}x")

        results[ds_name] = ds_results

    return results


# ===========================================================================
# Group S4: Comparison to Baselines
# ===========================================================================
def run_group_s4(datasets=None):
    """Fresh, Naive cache, QA-bypass (current), Speculative (new)."""
    datasets = datasets or GROUP_DATASETS["S4"]
    print(f"\n{'='*70}")
    print("GROUP S4: Comparison to Baselines")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg)
        item_embs = model.get_all_item_embeddings()
        cm = get_cluster_manager(ds_name, train, item_embs, cfg)
        int_counts = build_interaction_counts(train)

        sample_users = list(gt.keys())[:500]
        warm_users = list(gt.keys())

        ds_results = {}

        # --- Fresh (no cache) ---
        print("  Method: Fresh (no cache)", flush=True)
        run_metrics = defaultdict(list)
        for run_i in range(N_RUNS):
            set_seed(42 + run_i)
            np.random.shuffle(sample_users)
            recommendations = {}
            user_ndcgs = {}
            for uid in sample_users:
                if uid not in gt:
                    continue
                recs = list(model.recommend(uid, n=10))
                recommendations[uid] = recs
                user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(recs, gt[uid], 10)

            run_metrics["ndcg"].append(float(np.mean(list(user_ndcgs.values()))))
            run_metrics["accept_rate"].append(0.0)
            run_metrics["speedup"].append(1.0)
            ilds = [compute_ild(r, item_embs) for r in recommendations.values()]
            run_metrics["ild"].append(float(np.mean(ilds)))
            run_metrics["coverage"].append(compute_coverage(recommendations, item_embs.shape[0]))
            run_metrics["tail_ndcg"].append(
                compute_tail_user_ndcg(user_ndcgs, int_counts, threshold=5)
            )

        ds_results["Fresh"] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in run_metrics.items()
        }
        r = ds_results["Fresh"]
        print(f"    NDCG={r['ndcg']['mean']:.4f}  Speedup=1.00x")

        # --- Naive cache (always accept nearest cluster, no threshold) ---
        print("  Method: Naive cache (always accept)", flush=True)
        run_metrics = defaultdict(list)
        for run_i in range(N_RUNS):
            set_seed(42 + run_i)
            np.random.shuffle(sample_users)
            criterion = CosineAcceptanceCriterion(threshold=0.0)  # always accept
            m = evaluate_speculative(
                recommender=model, cluster_manager=cm,
                acceptance_criterion=criterion,
                ground_truth=gt, user_ids=sample_users,
                item_embeddings=item_embs,
                interaction_counts=int_counts,
                top_k_clusters=1, threshold=0.0,
                warm_user_ids=warm_users,
            )
            for metric, val in m.items():
                run_metrics[metric].append(val)

        ds_results["Naive Cache"] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in run_metrics.items()
        }
        r = ds_results["Naive Cache"]
        print(f"    NDCG={r['ndcg']['mean']:.4f}  "
              f"Accept={r['accept_rate']['mean']:.4f}  "
              f"Speedup={r['speedup']['mean']:.2f}x")

        # --- QA-bypass (current system) ---
        print("  Method: QA-bypass (current)", flush=True)
        run_metrics = defaultdict(list)
        for run_i in range(N_RUNS):
            set_seed(42 + run_i)
            np.random.shuffle(sample_users)

            qp = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
            criterion = HeuristicAcceptanceCriterion(qp, threshold=0.5)
            m = evaluate_speculative(
                recommender=model, cluster_manager=cm,
                acceptance_criterion=criterion,
                ground_truth=gt, user_ids=sample_users,
                item_embeddings=item_embs,
                interaction_counts=int_counts,
                top_k_clusters=1, threshold=0.5,
                warm_user_ids=warm_users,
            )
            for metric, val in m.items():
                run_metrics[metric].append(val)

        ds_results["QA-bypass"] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in run_metrics.items()
        }
        r = ds_results["QA-bypass"]
        print(f"    NDCG={r['ndcg']['mean']:.4f}  "
              f"Accept={r['accept_rate']['mean']:.4f}  "
              f"Speedup={r['speedup']['mean']:.2f}x")

        # --- Speculative (K=3, ScoreRatio) ---
        print("  Method: Speculative (K=3, ScoreRatio)", flush=True)
        run_metrics = defaultdict(list)
        for run_i in range(N_RUNS):
            set_seed(42 + run_i)
            np.random.shuffle(sample_users)

            criterion = ScoreRatioAcceptanceCriterion(threshold=0.35, temperature=1.0)
            m = evaluate_speculative(
                recommender=model, cluster_manager=cm,
                acceptance_criterion=criterion,
                ground_truth=gt, user_ids=sample_users,
                item_embeddings=item_embs,
                interaction_counts=int_counts,
                top_k_clusters=3, threshold=0.35,
                warm_user_ids=warm_users,
            )
            for metric, val in m.items():
                run_metrics[metric].append(val)

        ds_results["Speculative"] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in run_metrics.items()
        }
        r = ds_results["Speculative"]
        print(f"    NDCG={r['ndcg']['mean']:.4f}  "
              f"Accept={r['accept_rate']['mean']:.4f}  "
              f"MCGain={r['multi_cluster_gain']['mean']:.4f}  "
              f"Speedup={r['speedup']['mean']:.2f}x")

        # --- Speculative + Reranker ---
        print("  Method: Speculative + Reranker", flush=True)
        run_metrics = defaultdict(list)
        user_hist = build_user_history(train)
        for run_i in range(N_RUNS):
            set_seed(42 + run_i)
            np.random.shuffle(sample_users)

            criterion = ScoreRatioAcceptanceCriterion(threshold=0.35, temperature=1.0)
            reranker = LightweightReranker(
                history_weight=0.3, recency_weight=0.3, diversity_weight=0.2,
            )
            reranker.set_item_embeddings(item_embs)
            for uid in sample_users:
                if uid in user_hist:
                    reranker.set_user_history(uid, user_hist[uid][-20:])

            m = evaluate_speculative(
                recommender=model, cluster_manager=cm,
                acceptance_criterion=criterion,
                ground_truth=gt, user_ids=sample_users,
                item_embeddings=item_embs,
                interaction_counts=int_counts,
                top_k_clusters=3, threshold=0.35,
                warm_user_ids=warm_users,
                reranker=reranker,
            )
            for metric, val in m.items():
                run_metrics[metric].append(val)

        ds_results["Speculative+Rerank"] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in run_metrics.items()
        }
        r = ds_results["Speculative+Rerank"]
        print(f"    NDCG={r['ndcg']['mean']:.4f}  "
              f"Accept={r['accept_rate']['mean']:.4f}  "
              f"MCGain={r['multi_cluster_gain']['mean']:.4f}  "
              f"Speedup={r['speedup']['mean']:.2f}x")

        results[ds_name] = ds_results

    return results


# ===========================================================================
# Figure generation
# ===========================================================================
def generate_figures(all_results, fig_dir="paper/figures"):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- S1: K vs acceptance_rate & NDCG ---
    if "S1" in all_results:
        for ds_name, ds_data in all_results["S1"].items():
            k_vals = []
            ndcgs = []
            accept_rates = []
            mc_gains = []
            for key in sorted(ds_data.keys(), key=lambda x: int(x.split("=")[1])):
                k_vals.append(int(key.split("=")[1]))
                ndcgs.append(ds_data[key]["ndcg"]["mean"])
                accept_rates.append(ds_data[key]["accept_rate"]["mean"])
                mc_gains.append(ds_data[key]["multi_cluster_gain"]["mean"])

            fig, ax1 = plt.subplots(figsize=(7, 4.5))
            color1 = "#1f77b4"
            color2 = "#d62728"

            ax1.plot(k_vals, accept_rates, "o-", linewidth=2, markersize=8,
                     color=color1, label="Acceptance Rate")
            ax1.set_xlabel("Number of Candidate Clusters (K)", fontsize=11)
            ax1.set_ylabel("Acceptance Rate", fontsize=11, color=color1)
            ax1.tick_params(axis="y", labelcolor=color1)

            ax2 = ax1.twinx()
            ax2.plot(k_vals, ndcgs, "s--", linewidth=2, markersize=8,
                     color=color2, label="NDCG@10")
            ax2.set_ylabel("NDCG@10", fontsize=11, color=color2)
            ax2.tick_params(axis="y", labelcolor=color2)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

            ax1.set_title(f"Multi-Cluster Speculation ({ds_name.upper()})",
                          fontsize=12, fontweight="bold")
            ax1.set_xticks(k_vals)
            plt.tight_layout()
            plt.savefig(fig_dir / f"s1_multicluster_{ds_name}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"s1_multicluster_{ds_name}.png",
                        bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: s1_multicluster_{ds_name}.pdf")

    # --- S2: Grouped bar chart ---
    if "S2" in all_results:
        for ds_name, ds_data in all_results["S2"].items():
            criteria = list(ds_data.keys())
            metrics_to_plot = ["ndcg", "accept_rate", "ild", "coverage"]
            metric_labels = ["NDCG@10", "Accept Rate", "ILD", "Coverage"]

            x = np.arange(len(criteria))
            width = 0.2
            fig, ax = plt.subplots(figsize=(10, 5))

            for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
                vals = [ds_data[c][metric]["mean"] for c in criteria]
                errs = [ds_data[c][metric]["std"] for c in criteria]
                ax.bar(x + i * width, vals, width, yerr=errs, label=label,
                       capsize=3, alpha=0.85)

            ax.set_xlabel("Acceptance Criterion", fontsize=11)
            ax.set_ylabel("Value", fontsize=11)
            ax.set_title(f"Acceptance Criterion Comparison ({ds_name.upper()})",
                          fontsize=12, fontweight="bold")
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(criteria, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / f"s2_criteria_{ds_name}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"s2_criteria_{ds_name}.png",
                        bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: s2_criteria_{ds_name}.pdf")

    # --- S3: Pareto front (most important figure) ---
    if "S3" in all_results:
        for ds_name, ds_data in all_results["S3"].items():
            thrs = []
            ndcgs = []
            accept_rates = []
            speedups = []
            mc_gains = []
            for key in sorted(ds_data.keys(), key=lambda x: float(x.split("=")[1])):
                thrs.append(float(key.split("=")[1]))
                ndcgs.append(ds_data[key]["ndcg"]["mean"])
                accept_rates.append(ds_data[key]["accept_rate"]["mean"])
                speedups.append(ds_data[key]["speedup"]["mean"])
                mc_gains.append(ds_data[key].get("multi_cluster_gain", {}).get("mean", 0))

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Left: acceptance_rate vs NDCG (Pareto front)
            sc = axes[0].scatter(accept_rates, ndcgs, c=thrs, cmap="coolwarm",
                                 s=100, edgecolors="black", linewidth=0.5, zorder=3)
            axes[0].plot(accept_rates, ndcgs, "k--", alpha=0.3, linewidth=1)
            for i, thr in enumerate(thrs):
                axes[0].annotate(f"{thr:.2f}", (accept_rates[i], ndcgs[i]),
                                 textcoords="offset points", xytext=(6, 6),
                                 fontsize=7, color="gray")
            axes[0].set_xlabel("Acceptance Rate (Cache Hit Rate)", fontsize=11)
            axes[0].set_ylabel("NDCG@10", fontsize=11)
            axes[0].set_title("Quality-Speed Pareto Front", fontsize=12, fontweight="bold")
            plt.colorbar(sc, ax=axes[0], label="Threshold")

            # Middle: threshold vs speedup
            axes[1].plot(thrs, speedups, "o-", linewidth=2, markersize=8,
                         color="#2ca02c")
            axes[1].fill_between(thrs, 1, speedups, alpha=0.15, color="#2ca02c")
            axes[1].set_xlabel("Acceptance Threshold", fontsize=11)
            axes[1].set_ylabel("Speedup (x)", fontsize=11)
            axes[1].set_title("Throughput Gain vs Threshold", fontsize=12, fontweight="bold")
            axes[1].axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
            axes[1].grid(True, alpha=0.3)

            # Right: threshold vs multi-cluster gain
            axes[2].plot(thrs, mc_gains, "D-", linewidth=2, markersize=8,
                         color="#9467bd")
            axes[2].fill_between(thrs, 0, mc_gains, alpha=0.15, color="#9467bd")
            axes[2].set_xlabel("Acceptance Threshold", fontsize=11)
            axes[2].set_ylabel("Multi-Cluster Gain", fontsize=11)
            axes[2].set_title("Non-Nearest Cluster Recovery", fontsize=12, fontweight="bold")
            axes[2].set_ylim(bottom=0)
            axes[2].grid(True, alpha=0.3)

            plt.suptitle(f"ScoreRatio Threshold Sweep ({ds_name.upper()})",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / f"s3_pareto_{ds_name}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"s3_pareto_{ds_name}.png",
                        bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: s3_pareto_{ds_name}.pdf")

    # --- S4: Summary table as a figure ---
    if "S4" in all_results:
        for ds_name, ds_data in all_results["S4"].items():
            methods = list(ds_data.keys())
            cols = ["ndcg", "accept_rate", "speedup", "coverage", "ild", "tail_ndcg"]
            col_labels = ["NDCG@10", "Accept Rate", "Speedup", "Coverage", "ILD", "Tail NDCG"]

            fig, ax = plt.subplots(figsize=(12, 2.5))
            ax.axis("off")

            cell_text = []
            for method in methods:
                row = []
                for col in cols:
                    mean = ds_data[method].get(col, {}).get("mean", 0)
                    std = ds_data[method].get(col, {}).get("std", 0)
                    if col == "speedup":
                        row.append(f"{mean:.2f}x")
                    else:
                        row.append(f"{mean:.4f}")
                cell_text.append(row)

            table = ax.table(
                cellText=cell_text,
                rowLabels=methods,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.4)

            # Highlight best in each column
            for j in range(len(cols)):
                vals = []
                for i in range(len(methods)):
                    v = ds_data[methods[i]].get(cols[j], {}).get("mean", 0)
                    vals.append(v)
                # For speedup, coverage, ndcg: higher is better. For ild: higher is better too
                best_idx = int(np.argmax(vals))
                table[best_idx + 1, j].set_facecolor("#d4edda")

            ax.set_title(f"Method Comparison ({ds_name.upper()})",
                         fontsize=12, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(fig_dir / f"s4_comparison_{ds_name}.pdf",
                        bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"s4_comparison_{ds_name}.png",
                        bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: s4_comparison_{ds_name}.pdf")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="RecCache Speculative Experiments")
    parser.add_argument(
        "--group", nargs="*", default=None,
        help="Which experiment groups to run (S1 S2 S3 S4). Default: all.",
    )
    args = parser.parse_args()

    groups_to_run = [g.upper() for g in args.group] if args.group else ["S1", "S2", "S3", "S4"]

    print("=" * 70)
    print("RecCache Speculative Recommendation Experiments")
    print(f"Groups: {', '.join(groups_to_run)}")
    print("=" * 70, flush=True)

    all_results = {}
    start_time = time.time()

    group_functions = {
        "S1": run_group_s1,
        "S2": run_group_s2,
        "S3": run_group_s3,
        "S4": run_group_s4,
    }

    for group in groups_to_run:
        if group not in group_functions:
            print(f"WARNING: Unknown group '{group}', skipping")
            continue
        try:
            all_results[group] = group_functions[group]()
        except Exception as e:
            print(f"\nERROR in Group {group}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")

    # Generate figures
    if all_results:
        print("\nGenerating figures...")
        generate_figures(all_results)

    # Save results
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_path = Path("results/speculative_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("Speculative experiments complete!")


if __name__ == "__main__":
    main()
