#!/usr/bin/env python3
"""
Comprehensive experiments for RecSys 2026 submission.

Extends the original speculative experiments with:
  S5 — Pool Retrieval vs Static Cache (core new contribution)
  S6 — Semantic vs Random Clustering under Pool Retrieval
  S7 — Pool Size Sweep

All original groups S1-S4 also run on expanded datasets:
  ml-1m, amazon-movies, amazon-toys, mind-small

Usage:
    python scripts/run_recsys_experiments.py                    # all groups
    python scripts/run_recsys_experiments.py --group S5         # pool retrieval only
    python scripts/run_recsys_experiments.py --group S5 S6 S7   # new groups only
    python scripts/run_recsys_experiments.py --group S1 S4      # original groups on expanded datasets
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

ALL_DATASETS = ["ml-1m", "amazon-electronics", "amazon-arts", "mind-small"]

# Which datasets to use per group — all groups now use all 4 datasets
GROUP_DATASETS = {
    "S1": ALL_DATASETS,
    "S2": ALL_DATASETS,
    "S3": ["ml-1m", "amazon-electronics"],  # threshold sweep: 2 main datasets
    "S4": ALL_DATASETS,
    "S5": ALL_DATASETS,
    "S6": ALL_DATASETS,
    "S7": ["ml-1m", "amazon-electronics"],  # pool size sweep: 2 main datasets
}

N_RUNS = 3
N_TEST_USERS = 500


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
# Cached data / model loading
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


def get_trained_model(name, train, cfg, verbose=True):
    """Train a fresh model (no caching — each seed gets an independent model)."""
    if verbose:
        print(f"  Training MF on {name}...", flush=True)
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items,
        embedding_dim=cfg["embedding_dim"],
    )
    model.fit(
        train.user_ids, train.item_ids, train.ratings,
        epochs=cfg["epochs"], verbose=verbose,
    )
    return model


def get_cluster_manager(name, train, item_embeddings, cfg, random_clusters=False):
    """Build a fresh cluster manager (no caching — each seed gets independent clusters)."""
    n_clusters = min(cfg["n_clusters"], train.n_users // 2)
    cm = UserClusterManager(
        n_clusters=n_clusters,
        embedding_dim=item_embeddings.shape[1],
        n_items=len(item_embeddings),
    )
    cm.set_item_embeddings(item_embeddings)

    if random_clusters:
        n_users = len(set(train.user_ids.tolist()))
        random_embs = np.random.randn(n_users, item_embeddings.shape[1]).astype(np.float32)
        for i in range(n_users):
            random_embs[i] /= np.linalg.norm(random_embs[i]) + 1e-8
        cm.clusterer.initialize(random_embs)
        cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
    else:
        cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    return cm


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_speculative(
    recommender, cluster_manager, acceptance_criterion, ground_truth,
    user_ids, item_embeddings, interaction_counts=None,
    top_k_clusters=3, threshold=0.5, n_recs=10,
    warm_user_ids=None, reranker=None,
    use_pool_retrieval=False, pool_size=200,
    user_history=None, fresh_latency_ms=0.0,
):
    """Create a SpeculativeRecommender, warm cache, evaluate, return metrics."""
    config = SpeculativeConfig(
        top_k_clusters=top_k_clusters,
        acceptance_threshold=threshold,
        n_recs=n_recs,
        rerank_on_accept=reranker is not None,
        use_pool_retrieval=use_pool_retrieval,
        pool_size=pool_size,
    )
    spec = SpeculativeRecommender(
        recommender=recommender,
        cluster_manager=cluster_manager,
        acceptance_criterion=acceptance_criterion,
        config=config,
        reranker=reranker,
        item_embeddings=item_embeddings,
        user_history=user_history,
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
            "multi_cluster_gain": 0.0, "pool_retrieval_rate": 0.0,
        }

    # Aggregate
    accept_rate = SpeculativeMetrics.acceptance_rate(results)
    speedup = SpeculativeMetrics.speedup_estimate(results, fresh_latency_ms=fresh_latency_ms)
    mc_gain = SpeculativeMetrics.multi_cluster_gain(results)

    pool_count = sum(1 for r in results if r.retrieval_personalised)
    pool_rate = pool_count / len(results) if results else 0.0

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
        "pool_retrieval_rate": pool_rate,
    }


def evaluate_fresh(model, ground_truth, user_ids, item_embeddings,
                   interaction_counts, n_recs=10, user_history=None):
    """Evaluate fresh (no cache) as upper bound. Returns measured latency."""
    import time as _time
    recommendations = {}
    user_ndcgs = {}
    latencies = []
    for uid in user_ids:
        if uid not in ground_truth:
            continue
        exclude = user_history.get(uid) if user_history else None
        t0 = _time.perf_counter()
        recs = list(model.recommend(uid, n=n_recs, exclude_items=exclude))
        latencies.append((_time.perf_counter() - t0) * 1000)
        recommendations[uid] = recs
        user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(recs, ground_truth[uid], n_recs)

    if not user_ndcgs:
        return {"ndcg": 0.0, "accept_rate": 0.0, "coverage": 0.0,
                "ild": 0.0, "speedup": 1.0, "tail_ndcg": 0.0,
                "multi_cluster_gain": 0.0, "pool_retrieval_rate": 0.0,
                "fresh_latency_ms": 0.0}

    ilds = [compute_ild(r, item_embeddings) for r in recommendations.values()]
    return {
        "ndcg": float(np.mean(list(user_ndcgs.values()))),
        "accept_rate": 0.0,
        "coverage": compute_coverage(recommendations, item_embeddings.shape[0]),
        "ild": float(np.mean(ilds)),
        "speedup": 1.0,
        "tail_ndcg": compute_tail_user_ndcg(user_ndcgs, interaction_counts, threshold=5),
        "multi_cluster_gain": 0.0,
        "pool_retrieval_rate": 0.0,
        "fresh_latency_ms": float(np.mean(latencies)),
    }


def run_multi_seed(eval_fn, n_runs=N_RUNS):
    """Run eval_fn for multiple seeds and aggregate mean/std."""
    run_metrics = defaultdict(list)
    for run_i in range(n_runs):
        set_seed(42 + run_i)
        m = eval_fn(run_i)
        for metric, val in m.items():
            if isinstance(val, (int, float)):
                run_metrics[metric].append(val)
    return {
        metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for metric, vals in run_metrics.items()
    }


_model_cache = {}  # Cache model per dataset (deterministic, expensive to retrain)


def build_dataset_context(ds_name, seed=42):
    """Build model, clusters, user_history, fresh_latency for a dataset+seed.

    Model is trained once per dataset (deterministic given same data).
    Clusters are re-initialized per seed (K-means is stochastic) to
    provide independent variance across runs.
    """
    import torch
    set_seed(seed)

    train, val, test, gt, cfg = load_dataset_cached(ds_name)

    # Train model once per dataset (deterministic — re-training just wastes time)
    if ds_name not in _model_cache:
        _model_cache[ds_name] = get_trained_model(ds_name, train, cfg, verbose=True)
    model = _model_cache[ds_name]

    item_embs = model.get_all_item_embeddings()
    # Fresh cluster init per seed (K-means is stochastic)
    cm = get_cluster_manager(ds_name, train, item_embs, cfg)
    user_hist = build_user_history(train)
    int_counts = build_interaction_counts(train)

    sample_users = [u for u in gt.keys()][:N_TEST_USERS]
    warm_users = list(gt.keys())

    # Measure fresh latency once
    import time as _time
    latencies = []
    for uid in sample_users[:50]:
        if uid not in gt:
            continue
        exclude = user_hist.get(uid)
        t0 = _time.perf_counter()
        model.recommend(uid, n=10, exclude_items=exclude)
        latencies.append((_time.perf_counter() - t0) * 1000)
    fresh_lat = float(np.mean(latencies)) if latencies else 0.3

    return {
        "train": train, "test": test, "gt": gt, "cfg": cfg,
        "model": model, "item_embs": item_embs, "cm": cm,
        "user_hist": user_hist, "int_counts": int_counts,
        "sample_users": sample_users, "warm_users": warm_users,
        "fresh_latency_ms": fresh_lat,
    }


def print_result(label, r):
    ndcg = r.get("ndcg", {}).get("mean", 0)
    accept = r.get("accept_rate", {}).get("mean", 0)
    mcg = r.get("multi_cluster_gain", {}).get("mean", 0)
    spd = r.get("speedup", {}).get("mean", 0)
    cov = r.get("coverage", {}).get("mean", 0)
    pool = r.get("pool_retrieval_rate", {}).get("mean", 0)
    parts = [f"NDCG={ndcg:.4f}", f"Accept={accept:.3f}"]
    if mcg > 0:
        parts.append(f"MCG={mcg:.3f}")
    if spd > 1:
        parts.append(f"Spd={spd:.1f}x")
    if cov > 0:
        parts.append(f"Cov={cov:.4f}")
    if pool > 0:
        parts.append(f"Pool={pool:.3f}")
    print(f"    {label}: {', '.join(parts)}")


# ===========================================================================
# Group S1: Multi-cluster Speculation (expanded datasets)
# ===========================================================================
def run_group_s1(datasets=None):
    datasets = datasets or GROUP_DATASETS["S1"]
    print(f"\n{'='*70}\nS1: Multi-cluster Speculation\n{'='*70}")

    k_values = [1, 3, 5, 7]
    threshold = 0.35
    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")

        ds_results = {}
        for k in k_values:
            print(f"  K={k}", flush=True)
            def eval_fn(run_i, _k=k):
                ctx = build_dataset_context(ds_name, seed=42 + run_i)
                criterion = ScoreRatioAcceptanceCriterion(threshold=threshold, temperature=1.0)
                return evaluate_speculative(
                    recommender=ctx["model"], cluster_manager=ctx["cm"],
                    acceptance_criterion=criterion, ground_truth=ctx["gt"],
                    user_ids=ctx["sample_users"], item_embeddings=ctx["item_embs"],
                    interaction_counts=ctx["int_counts"],
                    top_k_clusters=_k, threshold=threshold,
                    warm_user_ids=ctx["warm_users"],
                    user_history=ctx["user_hist"],
                    fresh_latency_ms=ctx["fresh_latency_ms"],
                )
            ds_results[f"K={k}"] = run_multi_seed(eval_fn)
            print_result(f"K={k}", ds_results[f"K={k}"])

        results[ds_name] = ds_results
    return results


# ===========================================================================
# Group S2: Acceptance Criterion Comparison
# ===========================================================================
def run_group_s2(datasets=None):
    datasets = datasets or GROUP_DATASETS["S2"]
    print(f"\n{'='*70}\nS2: Acceptance Criterion Comparison\n{'='*70}")

    results = {}
    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")

        criteria = {
            "Cosine": (CosineAcceptanceCriterion(threshold=0.5), 0.5),
            "ScoreRatio": (ScoreRatioAcceptanceCriterion(threshold=0.3), 0.3),
            "Heuristic": (HeuristicAcceptanceCriterion(
                QualityPredictor(hidden_dim=32, quality_threshold=0.15), threshold=0.5), 0.5),
        }

        ds_results = {}
        for cname, (criterion, thr) in criteria.items():
            print(f"  {cname}", flush=True)
            def eval_fn(run_i, _c=criterion, _t=thr):
                ctx = build_dataset_context(ds_name, seed=42 + run_i)
                return evaluate_speculative(
                    recommender=ctx["model"], cluster_manager=ctx["cm"],
                    acceptance_criterion=_c, ground_truth=ctx["gt"],
                    user_ids=ctx["sample_users"], item_embeddings=ctx["item_embs"],
                    interaction_counts=ctx["int_counts"],
                    top_k_clusters=3, threshold=_t, warm_user_ids=ctx["warm_users"],
                    user_history=ctx["user_hist"],
                    fresh_latency_ms=ctx["fresh_latency_ms"],
                )
            ds_results[cname] = run_multi_seed(eval_fn)
            print_result(cname, ds_results[cname])

        results[ds_name] = ds_results
    return results


# ===========================================================================
# Group S3: Threshold Sweep (Pareto front)
# ===========================================================================
def run_group_s3(datasets=None):
    datasets = datasets or GROUP_DATASETS["S3"]
    print(f"\n{'='*70}\nS3: Acceptance Threshold Sweep (Pareto)\n{'='*70}")

    threshold_values = [0.05, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60]
    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")

        ds_results = {}
        for thr in threshold_values:
            print(f"  thr={thr:.2f}", flush=True)
            def eval_fn(run_i, _t=thr):
                ctx = build_dataset_context(ds_name, seed=42 + run_i)
                criterion = ScoreRatioAcceptanceCriterion(threshold=_t, temperature=1.0)
                return evaluate_speculative(
                    recommender=ctx["model"], cluster_manager=ctx["cm"],
                    acceptance_criterion=criterion, ground_truth=ctx["gt"],
                    user_ids=ctx["sample_users"], item_embeddings=ctx["item_embs"],
                    interaction_counts=ctx["int_counts"],
                    top_k_clusters=3, threshold=_t, warm_user_ids=ctx["warm_users"],
                    user_history=ctx["user_hist"],
                    fresh_latency_ms=ctx["fresh_latency_ms"],
                )
            ds_results[f"thr={thr:.2f}"] = run_multi_seed(eval_fn)
            print_result(f"thr={thr:.2f}", ds_results[f"thr={thr:.2f}"])

        # Also add Fresh baseline
        print("  Fresh (no cache)", flush=True)
        def eval_fresh(run_i):
            ctx = build_dataset_context(ds_name, seed=42 + run_i)
            return evaluate_fresh(ctx["model"], ctx["gt"], ctx["sample_users"],
                                  ctx["item_embs"], ctx["int_counts"],
                                  user_history=ctx["user_hist"])
        ds_results["Fresh"] = run_multi_seed(eval_fresh)
        print_result("Fresh", ds_results["Fresh"])

        results[ds_name] = ds_results
    return results


# ===========================================================================
# Group S4: End-to-End Comparison (expanded datasets)
# ===========================================================================
def run_group_s4(datasets=None):
    datasets = datasets or GROUP_DATASETS["S4"]
    print(f"\n{'='*70}\nS4: End-to-End Comparison\n{'='*70}")

    results = {}
    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")

        ds_results = {}

        method_names = ["Fresh", "Naive Cache", "QA-bypass", "Speculative", "Speculative+Pool"]

        for mname in method_names:
            print(f"  {mname}", flush=True)
            def eval_fn(run_i, _mname=mname):
                ctx = build_dataset_context(ds_name, seed=42 + run_i)
                if _mname == "Fresh":
                    return evaluate_fresh(ctx["model"], ctx["gt"], ctx["sample_users"],
                                          ctx["item_embs"], ctx["int_counts"],
                                          user_history=ctx["user_hist"])
                elif _mname == "Naive Cache":
                    return evaluate_speculative(
                        recommender=ctx["model"], cluster_manager=ctx["cm"],
                        acceptance_criterion=CosineAcceptanceCriterion(threshold=0.0),
                        ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                        item_embeddings=ctx["item_embs"],
                        interaction_counts=ctx["int_counts"],
                        top_k_clusters=1, threshold=0.0, warm_user_ids=ctx["warm_users"],
                        user_history=ctx["user_hist"],
                        fresh_latency_ms=ctx["fresh_latency_ms"],
                    )
                elif _mname == "QA-bypass":
                    return evaluate_speculative(
                        recommender=ctx["model"], cluster_manager=ctx["cm"],
                        acceptance_criterion=HeuristicAcceptanceCriterion(
                            QualityPredictor(hidden_dim=32, quality_threshold=0.15), threshold=0.5),
                        ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                        item_embeddings=ctx["item_embs"],
                        interaction_counts=ctx["int_counts"],
                        top_k_clusters=1, threshold=0.5, warm_user_ids=ctx["warm_users"],
                        user_history=ctx["user_hist"],
                        fresh_latency_ms=ctx["fresh_latency_ms"],
                    )
                elif _mname == "Speculative":
                    return evaluate_speculative(
                        recommender=ctx["model"], cluster_manager=ctx["cm"],
                        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35, temperature=1.0),
                        ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                        item_embeddings=ctx["item_embs"],
                        interaction_counts=ctx["int_counts"],
                        top_k_clusters=3, threshold=0.35, warm_user_ids=ctx["warm_users"],
                        user_history=ctx["user_hist"],
                        fresh_latency_ms=ctx["fresh_latency_ms"],
                    )
                elif _mname == "Speculative+Pool":
                    return evaluate_speculative(
                        recommender=ctx["model"], cluster_manager=ctx["cm"],
                        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35, temperature=1.0),
                        ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                        item_embeddings=ctx["item_embs"],
                        interaction_counts=ctx["int_counts"],
                        top_k_clusters=3, threshold=0.35, warm_user_ids=ctx["warm_users"],
                        user_history=ctx["user_hist"],
                        fresh_latency_ms=ctx["fresh_latency_ms"],
                        use_pool_retrieval=True, pool_size=200,
                    )
            ds_results[mname] = run_multi_seed(eval_fn)
            print_result(mname, ds_results[mname])

        results[ds_name] = ds_results
    return results


# ===========================================================================
# Group S5: Pool Retrieval vs Static Cache (NEW — core contribution)
# ===========================================================================
def run_group_s5(datasets=None):
    """Compare static list cache vs embedding pool retrieval.

    This is the headline experiment: pool retrieval enables per-user
    personalisation within clusters, addressing the core limitation that
    all users in a cluster get identical recommendations.
    """
    datasets = datasets or GROUP_DATASETS["S5"]
    print(f"\n{'='*70}\nS5: Pool Retrieval vs Static Cache\n{'='*70}")

    results = {}
    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")

        ds_results = {}

        # Fresh baseline
        print("  Fresh", flush=True)
        def eval_fresh(ri):
            ctx = build_dataset_context(ds_name, seed=42 + ri)
            return evaluate_fresh(ctx["model"], ctx["gt"], ctx["sample_users"],
                                  ctx["item_embs"], ctx["int_counts"],
                                  user_history=ctx["user_hist"])
        ds_results["Fresh"] = run_multi_seed(eval_fresh)
        print_result("Fresh", ds_results["Fresh"])

        # Static cache (original)
        for k in [1, 3]:
            label = f"Static K={k}"
            print(f"  {label}", flush=True)
            def eval_static(ri, _k=k):
                ctx = build_dataset_context(ds_name, seed=42 + ri)
                return evaluate_speculative(
                    recommender=ctx["model"], cluster_manager=ctx["cm"],
                    acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
                    ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                    item_embeddings=ctx["item_embs"],
                    interaction_counts=ctx["int_counts"],
                    top_k_clusters=_k, threshold=0.35, warm_user_ids=ctx["warm_users"],
                    user_history=ctx["user_hist"],
                    fresh_latency_ms=ctx["fresh_latency_ms"],
                    use_pool_retrieval=False,
                )
            ds_results[label] = run_multi_seed(eval_static)
            print_result(label, ds_results[label])

        # Pool retrieval
        for k in [1, 3]:
            label = f"Pool K={k}"
            print(f"  {label}", flush=True)
            def eval_pool(ri, _k=k):
                ctx = build_dataset_context(ds_name, seed=42 + ri)
                return evaluate_speculative(
                    recommender=ctx["model"], cluster_manager=ctx["cm"],
                    acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
                    ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                    item_embeddings=ctx["item_embs"],
                    interaction_counts=ctx["int_counts"],
                    top_k_clusters=_k, threshold=0.35, warm_user_ids=ctx["warm_users"],
                    user_history=ctx["user_hist"],
                    fresh_latency_ms=ctx["fresh_latency_ms"],
                    use_pool_retrieval=True, pool_size=200,
                )
            ds_results[label] = run_multi_seed(eval_pool)
            print_result(label, ds_results[label])

        results[ds_name] = ds_results
    return results


# ===========================================================================
# Group S6: Semantic vs Random Clustering under Pool Retrieval
# ===========================================================================
def run_group_s6(datasets=None):
    """With pool retrieval, does semantic clustering finally outperform random?

    The original RecCache showed semantic ≈ random clustering because all
    users in a cluster got the same list. Pool retrieval should make
    clustering quality matter.
    """
    datasets = datasets or GROUP_DATASETS["S6"]
    print(f"\n{'='*70}\nS6: Semantic vs Random Clustering (Pool Retrieval)\n{'='*70}")

    results = {}
    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")

        ds_results = {}

        configs = [
            ("Semantic+Static", False, False),
            ("Random+Static", True, False),
            ("Semantic+Pool", False, True),
            ("Random+Pool", True, True),
        ]

        for label, is_random, use_pool in configs:
            print(f"  {label}", flush=True)
            def eval_fn(ri, _random=is_random, _pool=use_pool):
                ctx = build_dataset_context(ds_name, seed=42 + ri)
                if _random:
                    cm = get_cluster_manager(ds_name, ctx["train"], ctx["item_embs"],
                                             ctx["cfg"], random_clusters=True)
                else:
                    cm = ctx["cm"]
                return evaluate_speculative(
                    recommender=ctx["model"], cluster_manager=cm,
                    acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
                    ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                    item_embeddings=ctx["item_embs"],
                    interaction_counts=ctx["int_counts"],
                    top_k_clusters=3, threshold=0.35, warm_user_ids=ctx["warm_users"],
                    user_history=ctx["user_hist"],
                    fresh_latency_ms=ctx["fresh_latency_ms"],
                    use_pool_retrieval=_pool, pool_size=200,
                )
            ds_results[label] = run_multi_seed(eval_fn)
            print_result(label, ds_results[label])

        results[ds_name] = ds_results
    return results


# ===========================================================================
# Group S7: Pool Size Sweep
# ===========================================================================
def run_group_s7(datasets=None):
    """How does pool size affect quality and coverage?"""
    datasets = datasets or GROUP_DATASETS["S7"]
    print(f"\n{'='*70}\nS7: Pool Size Sweep\n{'='*70}")

    pool_sizes = [50, 100, 200, 500]
    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")

        ds_results = {}
        for ps in pool_sizes:
            label = f"pool={ps}"
            print(f"  {label}", flush=True)
            def eval_fn(ri, _ps=ps):
                ctx = build_dataset_context(ds_name, seed=42 + ri)
                return evaluate_speculative(
                    recommender=ctx["model"], cluster_manager=ctx["cm"],
                    acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
                    ground_truth=ctx["gt"], user_ids=ctx["sample_users"],
                    item_embeddings=ctx["item_embs"],
                    interaction_counts=ctx["int_counts"],
                    top_k_clusters=3, threshold=0.35, warm_user_ids=ctx["warm_users"],
                    user_history=ctx["user_hist"],
                    fresh_latency_ms=ctx["fresh_latency_ms"],
                    use_pool_retrieval=True, pool_size=_ps,
                )
            ds_results[label] = run_multi_seed(eval_fn)
            print_result(label, ds_results[label])

        results[ds_name] = ds_results
    return results


# ===========================================================================
# Figure generation
# ===========================================================================
def generate_figures(all_results, fig_dir="paper/figures"):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # ---- S5: Pool vs Static — per-dataset figures ----
    if "S5" in all_results:
        for ds_name, ds_data in all_results["S5"].items():
            methods = [k for k in ds_data.keys() if k != "Fresh"]
            ndcgs = [ds_data[m]["ndcg"]["mean"] for m in methods]
            covs = [ds_data[m]["coverage"]["mean"] for m in methods]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            x = np.arange(len(methods))
            colors = ["#1f77b4" if "Static" in m else "#2ca02c" for m in methods]

            axes[0].bar(x, ndcgs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
            axes[0].set_ylabel("NDCG@10")
            axes[0].set_title("Quality")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
            if "Fresh" in ds_data:
                axes[0].axhline(y=ds_data["Fresh"]["ndcg"]["mean"], color="red",
                               linestyle="--", alpha=0.7, label="Fresh")
                axes[0].legend(fontsize=8)

            axes[1].bar(x, covs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
            axes[1].set_ylabel("Coverage")
            axes[1].set_title("Catalog Coverage")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(methods, rotation=25, ha="right", fontsize=8)

            plt.suptitle(f"Pool vs Static ({ds_name})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / f"s5_pool_vs_static_{ds_name}.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"s5_pool_vs_static_{ds_name}.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: s5_pool_vs_static_{ds_name}.pdf")

        # ---- S5: Cross-dataset coverage ratio summary ----
        ds_names = list(all_results["S5"].keys())
        static_covs, pool_covs, fresh_covs = [], [], []
        for ds in ds_names:
            d = all_results["S5"][ds]
            static_covs.append(d.get("Static K=3", {}).get("coverage", {}).get("mean", 0))
            pool_covs.append(d.get("Pool K=3", {}).get("coverage", {}).get("mean", 0))
            fresh_covs.append(d.get("Fresh", {}).get("coverage", {}).get("mean", 0))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(ds_names))
        w = 0.25
        ax.bar(x - w, fresh_covs, w, label="Fresh", color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x, static_covs, w, label="Static K=3", color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x + w, pool_covs, w, label="Pool K=3", color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Coverage", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("amazon-", "A-") for n in ds_names], fontsize=10)
        ax.legend(fontsize=9)
        ax.set_title("Coverage: Pool Retrieval vs Static Cache (all datasets)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir / "s5_coverage_all.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(fig_dir / "s5_coverage_all.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("  Saved: s5_coverage_all.pdf")

    # ---- S4: Cross-dataset method comparison ----
    if "S4" in all_results:
        ds_names = list(all_results["S4"].keys())
        methods_order = ["Fresh", "Naive Cache", "QA-bypass", "Speculative", "Speculative+Pool"]
        method_colors = {"Fresh": "#d62728", "Naive Cache": "#7f7f7f", "QA-bypass": "#ff7f0e",
                         "Speculative": "#1f77b4", "Speculative+Pool": "#2ca02c"}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: NDCG grouped bar
        x = np.arange(len(ds_names))
        n_methods = len(methods_order)
        w = 0.8 / n_methods
        for i, m in enumerate(methods_order):
            vals = [all_results["S4"][ds].get(m, {}).get("ndcg", {}).get("mean", 0) for ds in ds_names]
            stds = [all_results["S4"][ds].get(m, {}).get("ndcg", {}).get("std", 0) for ds in ds_names]
            axes[0].bar(x + i * w - 0.4 + w/2, vals, w, yerr=stds, label=m,
                       color=method_colors.get(m, "#333"), alpha=0.85, capsize=2,
                       edgecolor="black", linewidth=0.3)
        axes[0].set_ylabel("NDCG@10", fontsize=11)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.replace("amazon-", "A-") for n in ds_names], fontsize=9)
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].set_title("Recommendation Quality", fontweight="bold")

        # Right: Coverage grouped bar
        for i, m in enumerate(methods_order):
            vals = [all_results["S4"][ds].get(m, {}).get("coverage", {}).get("mean", 0) for ds in ds_names]
            axes[1].bar(x + i * w - 0.4 + w/2, vals, w, label=m,
                       color=method_colors.get(m, "#333"), alpha=0.85,
                       edgecolor="black", linewidth=0.3)
        axes[1].set_ylabel("Coverage", fontsize=11)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([n.replace("amazon-", "A-") for n in ds_names], fontsize=9)
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].set_title("Catalog Coverage", fontweight="bold")

        plt.suptitle("End-to-End Comparison (all datasets)", fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig(fig_dir / "s4_all_datasets.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(fig_dir / "s4_all_datasets.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("  Saved: s4_all_datasets.pdf")

        # Per-dataset summary table figures
        for ds_name, ds_data in all_results["S4"].items():
            methods = list(ds_data.keys())
            cols = ["ndcg", "accept_rate", "speedup", "coverage", "multi_cluster_gain"]
            col_labels = ["NDCG@10", "Accept", "Speedup", "Coverage", "MCG"]

            fig, ax = plt.subplots(figsize=(10, 2.2))
            ax.axis("off")
            cell_text = []
            for method in methods:
                row = []
                for col in cols:
                    mean = ds_data[method].get(col, {}).get("mean", 0)
                    std = ds_data[method].get(col, {}).get("std", 0)
                    if col == "speedup":
                        row.append(f"{mean:.1f}x")
                    elif col in ("accept_rate", "multi_cluster_gain"):
                        row.append(f"{mean:.1%}")
                    else:
                        row.append(f"{mean:.4f}±{std:.4f}")
                cell_text.append(row)

            table = ax.table(cellText=cell_text, rowLabels=methods, colLabels=col_labels,
                            cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.3)
            ax.set_title(f"S4: {ds_name}", fontsize=11, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.savefig(fig_dir / f"s4_table_{ds_name}.pdf", bbox_inches="tight", dpi=300)
            plt.close()
            print(f"  Saved: s4_table_{ds_name}.pdf")

    # ---- S6: Semantic vs Random clustering ----
    if "S6" in all_results:
        for ds_name, ds_data in all_results["S6"].items():
            methods = list(ds_data.keys())
            ndcgs = [ds_data[m]["ndcg"]["mean"] for m in methods]
            ndcg_stds = [ds_data[m]["ndcg"]["std"] for m in methods]

            colors = []
            for m in methods:
                if "Semantic" in m and "Pool" in m: colors.append("#2ca02c")
                elif "Random" in m and "Pool" in m: colors.append("#d62728")
                elif "Semantic" in m: colors.append("#1f77b4")
                else: colors.append("#ff7f0e")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x = np.arange(len(methods))
            ax.bar(x, ndcgs, yerr=ndcg_stds, color=colors, alpha=0.85,
                   capsize=5, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel("NDCG@10")
            ax.set_title(f"Clustering Quality ({ds_name})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / f"s6_clustering_{ds_name}.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"s6_clustering_{ds_name}.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: s6_clustering_{ds_name}.pdf")

    # ---- S3: Pareto front ----
    if "S3" in all_results:
        for ds_name, ds_data in all_results["S3"].items():
            thrs, ndcgs, accept_rates, speedups, mc_gains = [], [], [], [], []
            for key in sorted(k for k in ds_data.keys() if k.startswith("thr=")):
                thrs.append(float(key.split("=")[1]))
                ndcgs.append(ds_data[key]["ndcg"]["mean"])
                accept_rates.append(ds_data[key]["accept_rate"]["mean"])
                speedups.append(ds_data[key]["speedup"]["mean"])
                mc_gains.append(ds_data[key].get("multi_cluster_gain", {}).get("mean", 0))

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            sc = axes[0].scatter(accept_rates, ndcgs, c=thrs, cmap="coolwarm",
                                 s=100, edgecolors="black", linewidth=0.5, zorder=3)
            axes[0].plot(accept_rates, ndcgs, "k--", alpha=0.3)
            axes[0].set_xlabel("Acceptance Rate")
            axes[0].set_ylabel("NDCG@10")
            axes[0].set_title("Quality-Speed Pareto Front", fontweight="bold")
            plt.colorbar(sc, ax=axes[0], label="Threshold")

            axes[1].plot(thrs, speedups, "o-", linewidth=2, markersize=8, color="#2ca02c")
            axes[1].fill_between(thrs, 1, speedups, alpha=0.15, color="#2ca02c")
            axes[1].set_xlabel("Acceptance Threshold")
            axes[1].set_ylabel("Speedup (x)")
            axes[1].set_title("Throughput vs Threshold", fontweight="bold")

            axes[2].plot(thrs, mc_gains, "D-", linewidth=2, markersize=8, color="#9467bd")
            axes[2].fill_between(thrs, 0, mc_gains, alpha=0.15, color="#9467bd")
            axes[2].set_xlabel("Acceptance Threshold")
            axes[2].set_ylabel("Multi-Cluster Gain")
            axes[2].set_title("Non-Nearest Recovery", fontweight="bold")

            plt.suptitle(f"Pareto Front ({ds_name})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / f"s3_pareto_{ds_name}.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"s3_pareto_{ds_name}.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: s3_pareto_{ds_name}.pdf")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="RecCache RecSys 2026 Experiments")
    parser.add_argument(
        "--group", nargs="*", default=None,
        help="Groups to run (S1-S7). Default: all.",
    )
    args = parser.parse_args()

    all_groups = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    groups_to_run = [g.upper() for g in args.group] if args.group else all_groups

    print("=" * 70)
    print("RecCache — RecSys 2026 Comprehensive Experiments")
    print(f"Groups: {', '.join(groups_to_run)}")
    print(f"Datasets: {list(DATASET_CONFIGS.keys())}")
    print(f"Runs per config: {N_RUNS}, Test users: {N_TEST_USERS}")
    print("=" * 70, flush=True)

    all_results = {}
    start_time = time.time()

    group_functions = {
        "S1": run_group_s1, "S2": run_group_s2, "S3": run_group_s3,
        "S4": run_group_s4, "S5": run_group_s5, "S6": run_group_s6,
        "S7": run_group_s7,
    }

    for group in groups_to_run:
        if group not in group_functions:
            print(f"WARNING: Unknown group '{group}'")
            continue
        try:
            all_results[group] = group_functions[group]()
        except Exception as e:
            print(f"\nERROR in {group}: {e}")
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
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    results_path = Path("results/recsys2026_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for group, gdata in all_results.items():
        print(f"\n--- {group} ---")
        for ds_name, ds_data in gdata.items():
            print(f"  [{ds_name}]")
            for method, metrics in ds_data.items():
                ndcg = metrics.get("ndcg", {}).get("mean", 0)
                acc = metrics.get("accept_rate", {}).get("mean", 0)
                print(f"    {method:25s} NDCG={ndcg:.4f}  Accept={acc:.3f}")

    print("\nExperiments complete!")


if __name__ == "__main__":
    main()
