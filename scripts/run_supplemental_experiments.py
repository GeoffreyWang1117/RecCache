#!/usr/bin/env python3
"""
Supplemental experiments for RecCache paper revision.

Addresses 5 reviewer concerns:
  1. RecCache ≈ LRU+Clustering (Group A: multi-metric ablation)
  2. Synthetic traffic (Group C: temporal replay)
  3. Missing strong baselines (Group E: LeCaR + embedding alternatives)
  4. No proof semantic clustering matters (Group B: random vs semantic)
  5. No online re-clustering dynamics (Group F: adaptation over time)

Plus Group D: capacity sweep heatmap.

Usage:
    python scripts/run_supplemental_experiments.py               # all groups
    python scripts/run_supplemental_experiments.py --group A     # single group
    python scripts/run_supplemental_experiments.py --group A B   # multiple groups
"""

import sys
import argparse
import json
import time
import copy
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.models.baselines import LightGCNRecommender, create_recommender
from reccache.cache.baselines import create_cache, LRUCache
from reccache.cache.manager import CacheManager, RecommendationRequest
from reccache.cache.oracle import compute_oracle_bounds
from reccache.clustering.online_kmeans import OnlineKMeans
from reccache.clustering.user_cluster import UserClusterManager
from reccache.evaluation.metrics import (
    RecommendationMetrics,
    compute_ild,
    compute_coverage,
    compute_tail_user_ndcg,
)
from reccache.evaluation.simulator import OnlineSimulator, SimulationConfig
from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker
from reccache.utils.config import CacheConfig


# ---------------------------------------------------------------------------
# Dataset configs (same as run_complete_experiments.py)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15, "n_requests": 5000,
    },
    "amazon-movies": {
        "max_samples": 1000000,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15, "n_requests": 5000,
    },
}

# Datasets used by each group
GROUP_DATASETS = {
    "A": ["ml-1m", "amazon-movies"],
    "B": ["ml-1m", "amazon-movies"],
    "C": ["ml-1m", "amazon-movies"],
    "D": ["ml-1m"],
    "E": ["ml-1m"],
    "F": ["ml-1m"],
}

N_RUNS = 3  # repeated runs for error bars


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


def generate_zipf_sequence(n_users, n_requests, alpha=1.2, seed=42):
    rng = np.random.RandomState(seed)
    weights = 1.0 / np.arange(1, n_users + 1) ** alpha
    weights /= weights.sum()
    return list(rng.choice(n_users, size=n_requests, p=weights))


def temporal_replay_sequence(test_data):
    """Sort test interactions by real timestamp and return (user_id, item_id) pairs."""
    if test_data.timestamps is None:
        raise ValueError("Dataset has no timestamps for temporal replay")
    order = np.argsort(test_data.timestamps)
    return [(int(test_data.user_ids[i]), int(test_data.item_ids[i])) for i in order]


def session_replay_sequence(test_data, gap=1800):
    """Group interactions into sessions (gap > 1800s = new session), replay chronologically."""
    if test_data.timestamps is None:
        raise ValueError("Dataset has no timestamps for session replay")

    # Build per-user interaction lists sorted by time
    user_events = defaultdict(list)
    for uid, iid, ts in zip(test_data.user_ids, test_data.item_ids, test_data.timestamps):
        user_events[int(uid)].append((float(ts), int(iid)))

    sessions = []
    for uid, events in user_events.items():
        events.sort(key=lambda x: x[0])
        current_session = []
        for ts, iid in events:
            if current_session and ts - current_session[-1][0] > gap:
                sessions.append((current_session[0][0], uid, [x[1] for x in current_session]))
                current_session = []
            current_session.append((ts, iid))
        if current_session:
            sessions.append((current_session[0][0], uid, [x[1] for x in current_session]))

    # Sort sessions by start time
    sessions.sort(key=lambda x: x[0])
    # Flatten: each interaction in session order
    sequence = []
    for _, uid, items in sessions:
        for iid in items:
            sequence.append((uid, iid))
    return sequence


def evaluate_recs_for_users(recommender, user_ids, ground_truth, k=10,
                            item_embeddings=None, interaction_counts=None):
    """Evaluate a recommender, returning per-user metrics and aggregates."""
    recommendations = {}
    user_ndcgs = {}
    user_hit_rates = {}

    for uid in user_ids:
        if uid not in ground_truth:
            continue
        recs = recommender.recommend(uid, n=k)
        recommendations[uid] = recs
        relevant = ground_truth[uid]
        user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(recs, relevant, k)
        user_hit_rates[uid] = RecommendationMetrics.hit_rate(recs, relevant, k)

    n_users = len(recommendations)
    if n_users == 0:
        return {"ndcg": 0, "hit_rate": 0, "ild": 0, "coverage": 0, "tail_ndcg": 0}

    result = {
        "ndcg": float(np.mean(list(user_ndcgs.values()))),
        "hit_rate": float(np.mean(list(user_hit_rates.values()))),
    }

    # ILD
    if item_embeddings is not None:
        ilds = []
        for uid, recs in recommendations.items():
            ilds.append(compute_ild(recs, item_embeddings))
        result["ild"] = float(np.mean(ilds))
    else:
        result["ild"] = 0.0

    # Coverage
    result["coverage"] = compute_coverage(recommendations, item_embeddings.shape[0] if item_embeddings is not None else 1)

    # Tail-user NDCG
    if interaction_counts is not None:
        result["tail_ndcg"] = compute_tail_user_ndcg(user_ndcgs, interaction_counts, threshold=5)
    else:
        result["tail_ndcg"] = 0.0

    return result


def evaluate_recs_through_cache(
    recommender, user_ids, ground_truth, k=10,
    cluster_manager=None, reranker=None, quality_predictor=None,
    item_embeddings=None, interaction_counts=None,
):
    """
    Evaluate recommendations as actually served through the cache pipeline.

    Simulates what each user would receive:
    - No clustering: each user gets their own model.recommend() output
    - With clustering: users in the same cluster share the first-cached result
    - With QA: outlier users (far from cluster center) bypass cache
    - With reranker: cached recs are reranked per-user for personalization
    """
    cluster_cache = {}  # cluster_id -> cached recommendations
    recommendations = {}
    user_ndcgs = {}

    for uid in user_ids:
        if uid not in ground_truth:
            continue

        if cluster_manager is not None:
            info = cluster_manager.get_user_cluster(uid)
            cache_key = info.cluster_id

            # Check QA bypass: users far from center get fresh recs
            bypass_cache = False
            if quality_predictor is not None:
                prediction = quality_predictor.predict(
                    distance_to_center=info.distance_to_center,
                    cluster_size=info.cluster_size,
                )
                bypass_cache = not prediction.use_cache

            if bypass_cache:
                recs = list(recommender.recommend(uid, n=k))
            elif cache_key in cluster_cache:
                # Cache hit: serve shared cluster recs
                recs = list(cluster_cache[cache_key])
            else:
                # Cache miss: generate and cache for the cluster
                recs = list(recommender.recommend(uid, n=k))
                cluster_cache[cache_key] = list(recs)

            # Apply reranker to personalize cached/shared recs
            if reranker is not None:
                rerank_result = reranker.rerank(uid, recs, top_k=k)
                recs = rerank_result.items
        else:
            # No clustering: each user gets their own personalized recs
            recs = list(recommender.recommend(uid, n=k))

        recommendations[uid] = recs
        relevant = ground_truth[uid]
        user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(recs, relevant, k)

    n_users = len(recommendations)
    if n_users == 0:
        return {"ndcg": 0, "ild": 0, "coverage": 0, "tail_ndcg": 0}

    result = {
        "ndcg": float(np.mean(list(user_ndcgs.values()))),
    }

    # ILD
    if item_embeddings is not None:
        ilds = [compute_ild(recs, item_embeddings) for recs in recommendations.values()]
        result["ild"] = float(np.mean(ilds))
    else:
        result["ild"] = 0.0

    # Coverage
    result["coverage"] = compute_coverage(
        recommendations,
        item_embeddings.shape[0] if item_embeddings is not None else 1,
    )

    # Tail-user NDCG
    if interaction_counts is not None:
        result["tail_ndcg"] = compute_tail_user_ndcg(user_ndcgs, interaction_counts, threshold=5)
    else:
        result["tail_ndcg"] = 0.0

    return result


def simulate_cache_run(
    recommender, cluster_manager, ground_truth, train_data,
    n_requests, cache_size=5000, use_quality=False, use_reranker=False,
    item_embeddings=None, user_history=None, eviction="lru",
    seed=42,
):
    """Run one cache simulation, return hit_rate and NDCG."""
    set_seed(seed)

    cache_config = CacheConfig(
        local_cache_size=cache_size,
        use_redis_cache=False,
        quality_threshold=0.15,
    )
    cm = cluster_manager
    cache_manager = CacheManager(cache_config=cache_config, cluster_manager=cm)

    if not use_quality:
        cache_manager.local_cache = create_cache(eviction, max_size=cache_size)
    else:
        qp = QualityPredictor(hidden_dim=32, quality_threshold=0.15)
        cache_manager.set_quality_predictor(qp)

    reranker = None
    if use_reranker and item_embeddings is not None and user_history is not None:
        reranker = LightweightReranker(
            history_weight=0.3, recency_weight=0.3, diversity_weight=0.2
        )
        reranker.set_item_embeddings(item_embeddings)
        for uid, hist in user_history.items():
            reranker.set_user_history(uid, hist[-20:])

    sim_config = SimulationConfig(
        n_requests=n_requests,
        n_warmup_requests=min(500, n_requests // 5),
        eval_sample_rate=0.1,
    )
    simulator = OnlineSimulator(
        recommender=recommender,
        cache_manager=cache_manager,
        cluster_manager=cm,
        reranker=reranker,
        config=sim_config,
    )

    result = simulator.run_simulation(
        n_users=train_data.n_users,
        n_items=train_data.n_items,
        ground_truth=ground_truth,
        verbose=False,
    )
    return {"hit_rate": result.hit_rate, "ndcg": result.avg_ndcg}


def simulate_temporal_cache(
    recommender, cluster_manager, ground_truth, interaction_sequence,
    cache_size=5000, eviction="lru", k=10,
):
    """
    Simulate caching over a temporal interaction sequence.
    Returns hit_rate, ndcg, and speedup estimate.
    """
    cache = create_cache(eviction, max_size=cache_size, ttl=999999)
    hits = 0
    misses = 0
    ndcg_scores = []

    for uid, iid in interaction_sequence:
        # Cache key is cluster if cluster_manager, else user
        if cluster_manager is not None:
            info = cluster_manager.get_user_cluster(uid)
            cache_key = f"c{info.cluster_id}"
        else:
            cache_key = f"u{uid}"

        result = cache.get(cache_key)
        if result is not None:
            hits += 1
            recs = result
        else:
            misses += 1
            recs = recommender.recommend(uid, n=k)
            cache.put(cache_key, recs)

        if uid in ground_truth:
            ndcg = RecommendationMetrics.ndcg_at_k(recs, ground_truth[uid], k)
            ndcg_scores.append(ndcg)

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0
    avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0
    # Speedup estimate: cache lookup ~ 0.01ms, fresh compute ~ 5ms
    speedup = 1.0 / (hit_rate * 0.01 / 5.0 + (1 - hit_rate)) if hit_rate < 1 else 500.0

    return {"hit_rate": hit_rate, "ndcg": avg_ndcg, "speedup": speedup}


# ---------------------------------------------------------------------------
# Data loading helper (caches across groups)
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


def get_trained_model(name, train, cfg, model_type="mf"):
    cache_key = (name, model_type)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    print(f"  Training {model_type.upper()} on {name}...", flush=True)
    if model_type == "mf":
        model = MatrixFactorizationRecommender(
            n_users=train.n_users, n_items=train.n_items,
            embedding_dim=cfg["embedding_dim"],
        )
        model.fit(
            train.user_ids, train.item_ids, train.ratings,
            epochs=cfg["epochs"], verbose=True,
        )
    elif model_type == "lightgcn":
        model = LightGCNRecommender(
            n_users=train.n_users, n_items=train.n_items,
            embedding_dim=cfg["embedding_dim"],
        )
        model.fit(
            train.user_ids, train.item_ids, train.ratings,
            epochs=cfg["epochs"], verbose=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    _model_cache[cache_key] = model
    return model


_cluster_cache = {}


def get_cluster_manager(name, train, item_embeddings, cfg):
    if name in _cluster_cache:
        return _cluster_cache[name]

    n_clusters = min(cfg["n_clusters"], train.n_users // 2)
    cm = UserClusterManager(
        n_clusters=n_clusters,
        embedding_dim=item_embeddings.shape[1],
        n_items=len(item_embeddings),
    )
    cm.set_item_embeddings(item_embeddings)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
    _cluster_cache[name] = cm
    return cm


# ===========================================================================
# Group A: Rich Multi-Metric Ablation
# ===========================================================================
def run_group_a(datasets=None):
    """
    Rich multi-metric ablation: NDCG@10, ILD, Coverage, Tail-User-NDCG.
    Shows that QA-eviction and reranker improve diversity even if not NDCG.
    """
    datasets = datasets or GROUP_DATASETS["A"]
    print(f"\n{'='*70}")
    print("GROUP A: Rich Multi-Metric Ablation")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg, "mf")
        item_embs = model.get_all_item_embeddings()
        cm = get_cluster_manager(ds_name, train, item_embs, cfg)
        user_hist = build_user_history(train)
        int_counts = build_interaction_counts(train)

        # The 4 ablation variants
        variants = {
            "LRU (no clustering)": dict(clustering=False, quality=False, reranker=False),
            "LRU + Clustering": dict(clustering=True, quality=False, reranker=False),
            "LRU + Clustering + QA": dict(clustering=True, quality=True, reranker=False),
            "Full RecCache": dict(clustering=True, quality=True, reranker=True),
        }

        ds_results = {}

        for vname, vcfg in variants.items():
            print(f"  Variant: {vname}", flush=True)
            run_metrics = {"ndcg": [], "ild": [], "coverage": [], "tail_ndcg": [], "hit_rate": []}

            for run_i in range(N_RUNS):
                seed = 42 + run_i

                # Simulate cache to get hit rate
                sim = simulate_cache_run(
                    recommender=model,
                    cluster_manager=cm if vcfg["clustering"] else None,
                    ground_truth=gt, train_data=train,
                    n_requests=cfg["n_requests"], cache_size=5000,
                    use_quality=vcfg["quality"], use_reranker=vcfg["reranker"],
                    item_embeddings=item_embs, user_history=user_hist,
                    seed=seed,
                )
                run_metrics["hit_rate"].append(sim["hit_rate"])

                # Evaluate quality through the actual cache pipeline
                set_seed(seed)
                sample_users = list(gt.keys())[:500]
                np.random.shuffle(sample_users)  # randomize cache fill order

                # Build reranker for this variant if needed
                variant_reranker = None
                if vcfg["reranker"] and item_embs is not None and user_hist is not None:
                    variant_reranker = LightweightReranker(
                        history_weight=0.3, recency_weight=0.3, diversity_weight=0.2,
                    )
                    variant_reranker.set_item_embeddings(item_embs)
                    for uid in sample_users:
                        if uid in user_hist:
                            variant_reranker.set_user_history(uid, user_hist[uid][-20:])

                # Build QA predictor for this variant if needed
                variant_qp = None
                if vcfg["quality"]:
                    variant_qp = QualityPredictor(hidden_dim=32, quality_threshold=0.15)

                eval_result = evaluate_recs_through_cache(
                    recommender=model,
                    user_ids=sample_users,
                    ground_truth=gt,
                    k=10,
                    cluster_manager=cm if vcfg["clustering"] else None,
                    reranker=variant_reranker,
                    quality_predictor=variant_qp,
                    item_embeddings=item_embs,
                    interaction_counts=int_counts,
                )
                run_metrics["ndcg"].append(eval_result["ndcg"])
                run_metrics["ild"].append(eval_result["ild"])
                run_metrics["coverage"].append(eval_result["coverage"])
                run_metrics["tail_ndcg"].append(eval_result["tail_ndcg"])

            ds_results[vname] = {
                metric: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
                for metric, vals in run_metrics.items()
            }

            print(f"    NDCG={ds_results[vname]['ndcg']['mean']:.4f}  "
                  f"ILD={ds_results[vname]['ild']['mean']:.4f}  "
                  f"Cov={ds_results[vname]['coverage']['mean']:.4f}  "
                  f"Tail={ds_results[vname]['tail_ndcg']['mean']:.4f}  "
                  f"HR={ds_results[vname]['hit_rate']['mean']:.4f}")

        results[ds_name] = ds_results

    return results


# ===========================================================================
# Group B: Random vs Semantic Clustering
# ===========================================================================
def run_group_b(datasets=None):
    """
    Compare semantic (MF), random, and frequency-based clustering.
    Proves that MF embeddings actually help.
    """
    datasets = datasets or GROUP_DATASETS["B"]
    print(f"\n{'='*70}")
    print("GROUP B: Random vs Semantic Clustering")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg, "mf")
        item_embs = model.get_all_item_embeddings()
        n_clusters = min(cfg["n_clusters"], train.n_users // 2)

        # --- Semantic clusters (current approach) ---
        cm_semantic = get_cluster_manager(ds_name, train, item_embs, cfg)

        # --- Random cluster assignment ---
        class RandomClusterManager:
            """Assigns users to random clusters (same K)."""
            def __init__(self, n_clusters, n_users, seed=42):
                rng = np.random.RandomState(seed)
                self._assignments = {u: int(rng.randint(0, n_clusters)) for u in range(n_users)}
                self.n_clusters = n_clusters

            def get_user_cluster(self, user_id):
                from reccache.clustering.user_cluster import UserClusterInfo
                cid = self._assignments.get(user_id, 0)
                return UserClusterInfo(
                    cluster_id=cid,
                    distance_to_center=0.5,
                    cluster_size=100,
                    embedding=np.zeros(64, dtype=np.float32),
                )

        cm_random = RandomClusterManager(n_clusters, train.n_users)

        # --- Frequency-based clusters (bucket by activity level) ---
        class FrequencyClusterManager:
            """Clusters users into K buckets by interaction count."""
            def __init__(self, n_clusters, train_data):
                counts = defaultdict(int)
                for uid in train_data.user_ids:
                    counts[int(uid)] += 1
                sorted_users = sorted(counts.keys(), key=lambda u: counts[u])
                bucket_size = max(1, len(sorted_users) // n_clusters)
                self._assignments = {}
                for i, uid in enumerate(sorted_users):
                    self._assignments[uid] = min(i // bucket_size, n_clusters - 1)
                self.n_clusters = n_clusters

            def get_user_cluster(self, user_id):
                from reccache.clustering.user_cluster import UserClusterInfo
                cid = self._assignments.get(user_id, 0)
                return UserClusterInfo(
                    cluster_id=cid,
                    distance_to_center=0.5,
                    cluster_size=100,
                    embedding=np.zeros(64, dtype=np.float32),
                )

        cm_freq = FrequencyClusterManager(n_clusters, train)

        clustering_variants = {
            "Semantic (MF)": cm_semantic,
            "Random": cm_random,
            "Frequency": cm_freq,
        }

        ds_results = {}

        for cname, cm in clustering_variants.items():
            print(f"  Clustering: {cname}", flush=True)
            run_metrics = {"ndcg": [], "hit_rate": [], "coverage": []}

            for run_i in range(N_RUNS):
                seed = 42 + run_i
                sim = simulate_cache_run(
                    recommender=model,
                    cluster_manager=cm,
                    ground_truth=gt, train_data=train,
                    n_requests=cfg["n_requests"], cache_size=5000,
                    seed=seed,
                )
                run_metrics["hit_rate"].append(sim["hit_rate"])
                run_metrics["ndcg"].append(sim["ndcg"])

                # Coverage
                set_seed(seed)
                sample_users = list(gt.keys())[:500]
                recs_dict = {uid: model.recommend(uid, n=10) for uid in sample_users if uid in gt}
                run_metrics["coverage"].append(
                    compute_coverage(recs_dict, train.n_items)
                )

            ds_results[cname] = {
                metric: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for metric, v in run_metrics.items()
            }
            print(f"    NDCG={ds_results[cname]['ndcg']['mean']:.4f}  "
                  f"HR={ds_results[cname]['hit_rate']['mean']:.4f}  "
                  f"Cov={ds_results[cname]['coverage']['mean']:.4f}")

        results[ds_name] = ds_results

    return results


# ===========================================================================
# Group C: Temporal Replay
# ===========================================================================
def run_group_c(datasets=None):
    """
    Compare Zipf sampling vs temporal replay vs session-aware replay.
    """
    datasets = datasets or GROUP_DATASETS["C"]
    print(f"\n{'='*70}")
    print("GROUP C: Temporal Replay")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg, "mf")
        item_embs = model.get_all_item_embeddings()
        cm = get_cluster_manager(ds_name, train, item_embs, cfg)

        ds_results = {}

        # --- Zipf (current) ---
        print("  Traffic: Zipf (synthetic)", flush=True)
        zipf_runs = {"ndcg": [], "hit_rate": [], "speedup": []}
        for run_i in range(N_RUNS):
            sim = simulate_cache_run(
                recommender=model, cluster_manager=cm,
                ground_truth=gt, train_data=train,
                n_requests=cfg["n_requests"], cache_size=5000, seed=42 + run_i,
            )
            zipf_runs["hit_rate"].append(sim["hit_rate"])
            zipf_runs["ndcg"].append(sim["ndcg"])
            zipf_runs["speedup"].append(1.0 / (sim["hit_rate"] * 0.01 / 5.0 + (1 - sim["hit_rate"])))
        ds_results["Zipf (synthetic)"] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in zipf_runs.items()
        }
        print(f"    HR={ds_results['Zipf (synthetic)']['hit_rate']['mean']:.4f}  "
              f"NDCG={ds_results['Zipf (synthetic)']['ndcg']['mean']:.4f}")

        # --- Temporal replay ---
        has_timestamps = test.timestamps is not None and test.timestamps.max() > 0
        if has_timestamps:
            print("  Traffic: Temporal replay", flush=True)
            temporal_seq = temporal_replay_sequence(test)
            temporal_result = simulate_temporal_cache(
                model, cm, gt, temporal_seq, cache_size=5000,
            )
            ds_results["Temporal replay"] = {
                m: {"mean": float(v), "std": 0.0} for m, v in temporal_result.items()
            }
            print(f"    HR={temporal_result['hit_rate']:.4f}  "
                  f"NDCG={temporal_result['ndcg']:.4f}")

            # --- Session-aware replay ---
            print("  Traffic: Session-aware replay", flush=True)
            session_seq = session_replay_sequence(test, gap=1800)
            session_result = simulate_temporal_cache(
                model, cm, gt, session_seq, cache_size=5000,
            )
            ds_results["Session replay"] = {
                m: {"mean": float(v), "std": 0.0} for m, v in session_result.items()
            }
            print(f"    HR={session_result['hit_rate']:.4f}  "
                  f"NDCG={session_result['ndcg']:.4f}")
        else:
            print(f"  WARNING: {ds_name} has no timestamps, skipping temporal/session replay")

        results[ds_name] = ds_results

    return results


# ===========================================================================
# Group D: Cache Capacity Sweep
# ===========================================================================
def run_group_d(datasets=None):
    """
    Sweep K (cluster count) and cache_size, produce heatmap.
    """
    datasets = datasets or GROUP_DATASETS["D"]
    print(f"\n{'='*70}")
    print("GROUP D: Cache Capacity Sweep")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg, "mf")
        item_embs = model.get_all_item_embeddings()

        max_k = min(500, train.n_users // 3)
        k_values = sorted(set(k for k in [10, 25, 50, 100, 200, 500] if k <= max_k))
        cache_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

        heatmap = {}  # (K, cache_size) -> {"ndcg": ..., "hit_rate": ...}

        for k in k_values:
            print(f"  K={k}", flush=True)
            cm_k = UserClusterManager(
                n_clusters=k,
                embedding_dim=item_embs.shape[1],
                n_items=len(item_embs),
            )
            cm_k.set_item_embeddings(item_embs)
            cm_k.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

            for cs in cache_sizes:
                sim = simulate_cache_run(
                    recommender=model, cluster_manager=cm_k,
                    ground_truth=gt, train_data=train,
                    n_requests=min(3000, cfg["n_requests"]),
                    cache_size=cs, seed=42,
                )
                heatmap[f"{k}_{cs}"] = sim
                print(f"    cache_size={cs:>5}: HR={sim['hit_rate']:.4f} NDCG={sim['ndcg']:.4f}")

        # Also compute LRU (no clustering) across cache sizes for comparison
        lru_results = {}
        for cs in cache_sizes:
            sim = simulate_cache_run(
                recommender=model, cluster_manager=None,
                ground_truth=gt, train_data=train,
                n_requests=min(3000, cfg["n_requests"]),
                cache_size=cs, seed=42,
            )
            lru_results[str(cs)] = sim

        results[ds_name] = {
            "heatmap": heatmap,
            "k_values": k_values,
            "cache_sizes": cache_sizes,
            "lru_baseline": lru_results,
        }

    return results


# ===========================================================================
# Group E: Alternative Embeddings + LeCaR Baseline
# ===========================================================================
def run_group_e(datasets=None):
    """
    Test MF vs LightGCN vs Random embeddings for clustering,
    and LRU/LFU/ARC/LeCaR/Oracle cache baselines.
    """
    datasets = datasets or GROUP_DATASETS["E"]
    print(f"\n{'='*70}")
    print("GROUP E: Alternative Embeddings + LeCaR Baseline")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        n_clusters = min(cfg["n_clusters"], train.n_users // 2)

        # --- Train different embedding models ---
        mf_model = get_trained_model(ds_name, train, cfg, "mf")
        mf_embs = mf_model.get_all_item_embeddings()

        # LightGCN embeddings
        print("  Training LightGCN...", flush=True)
        lgcn_model = get_trained_model(ds_name, train, cfg, "lightgcn")
        lgcn_embs = lgcn_model.get_all_item_embeddings()

        # Random embeddings
        rng = np.random.RandomState(42)
        random_embs = rng.randn(train.n_items, cfg["embedding_dim"]).astype(np.float32)
        random_embs /= np.linalg.norm(random_embs, axis=1, keepdims=True) + 1e-8

        embedding_variants = {
            "MF": (mf_model, mf_embs),
            "LightGCN": (lgcn_model, lgcn_embs),
            "Random": (mf_model, random_embs),  # Use MF recommender but cluster with random embs
        }

        # --- Part 1: Embedding alternatives ---
        print("\n  Part 1: Embedding alternatives for clustering")
        emb_results = {}

        for ename, (rec_model, embs) in embedding_variants.items():
            print(f"    Embedding: {ename}", flush=True)
            cm = UserClusterManager(
                n_clusters=n_clusters,
                embedding_dim=embs.shape[1],
                n_items=len(embs),
            )
            cm.set_item_embeddings(embs)
            cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

            run_metrics = {"ndcg": [], "hit_rate": []}
            for run_i in range(N_RUNS):
                sim = simulate_cache_run(
                    recommender=rec_model, cluster_manager=cm,
                    ground_truth=gt, train_data=train,
                    n_requests=cfg["n_requests"], cache_size=5000, seed=42 + run_i,
                )
                run_metrics["hit_rate"].append(sim["hit_rate"])
                run_metrics["ndcg"].append(sim["ndcg"])

            emb_results[ename] = {
                m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for m, v in run_metrics.items()
            }
            print(f"      HR={emb_results[ename]['hit_rate']['mean']:.4f}  "
                  f"NDCG={emb_results[ename]['ndcg']['mean']:.4f}")

        # --- Part 2: Cache strategy baselines with same (MF) clustering ---
        print("\n  Part 2: Cache strategy baselines (with MF clustering)")
        cm_mf = get_cluster_manager(ds_name, train, mf_embs, cfg)

        cache_strategies = ["lru", "lfu", "arc", "lecar", "oracle"]
        cache_results = {}

        for strategy in cache_strategies:
            print(f"    Strategy: {strategy}", flush=True)
            run_metrics = {"ndcg": [], "hit_rate": []}

            for run_i in range(N_RUNS):
                seed = 42 + run_i
                set_seed(seed)

                cache_config = CacheConfig(
                    local_cache_size=5000, use_redis_cache=False,
                )
                cache_manager = CacheManager(
                    cache_config=cache_config, cluster_manager=cm_mf
                )
                cache_manager.local_cache = create_cache(strategy, max_size=5000, ttl=999999)

                # For oracle, we need to set future accesses
                if strategy == "oracle":
                    seq = generate_zipf_sequence(train.n_users, cfg["n_requests"], seed=seed)
                    # Convert to cluster keys
                    cluster_keys = []
                    for uid in seq:
                        info = cm_mf.get_user_cluster(uid)
                        cluster_keys.append(f"c{info.cluster_id}:")
                    cache_manager.local_cache.set_future_accesses(cluster_keys)

                sim_config = SimulationConfig(
                    n_requests=cfg["n_requests"],
                    n_warmup_requests=500,
                    eval_sample_rate=0.1,
                )
                simulator = OnlineSimulator(
                    recommender=mf_model,
                    cache_manager=cache_manager,
                    cluster_manager=cm_mf,
                    config=sim_config,
                )

                result = simulator.run_simulation(
                    n_users=train.n_users, n_items=train.n_items,
                    ground_truth=gt, verbose=False,
                )
                run_metrics["hit_rate"].append(result.hit_rate)
                run_metrics["ndcg"].append(result.avg_ndcg)

            cache_results[strategy] = {
                m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for m, v in run_metrics.items()
            }
            print(f"      HR={cache_results[strategy]['hit_rate']['mean']:.4f}  "
                  f"NDCG={cache_results[strategy]['ndcg']['mean']:.4f}")

        results[ds_name] = {
            "embedding_alternatives": emb_results,
            "cache_baselines": cache_results,
        }

    return results


# ===========================================================================
# Group F: Online Re-Clustering
# ===========================================================================
def run_group_f(datasets=None):
    """
    Track NDCG, cluster purity, and assignment drift across 4 temporal windows.
    Compare static vs online vs full retrain clustering.
    """
    datasets = datasets or GROUP_DATASETS["F"]
    print(f"\n{'='*70}")
    print("GROUP F: Online Re-Clustering")
    print(f"{'='*70}\n")

    results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        train, val, test, gt, cfg = load_dataset_cached(ds_name)
        model = get_trained_model(ds_name, train, cfg, "mf")
        item_embs = model.get_all_item_embeddings()
        n_clusters = min(cfg["n_clusters"], train.n_users // 2)

        # Check for timestamps
        if test.timestamps is None or test.timestamps.max() == 0:
            print(f"  WARNING: {ds_name} has no timestamps, skipping Group F")
            continue

        # Split test data into 4 temporal windows
        sorted_idx = np.argsort(test.timestamps)
        window_size = len(sorted_idx) // 4
        windows = []
        for w in range(4):
            start = w * window_size
            end = (w + 1) * window_size if w < 3 else len(sorted_idx)
            idx = sorted_idx[start:end]
            windows.append({
                "user_ids": test.user_ids[idx],
                "item_ids": test.item_ids[idx],
                "ratings": test.ratings[idx],
                "timestamps": test.timestamps[idx],
            })
            print(f"  Window {w}: {len(idx)} interactions")

        strategies = {}

        # --- Static clustering ---
        print("\n  Strategy: Static (train once, never update)")
        cm_static = UserClusterManager(
            n_clusters=n_clusters, embedding_dim=item_embs.shape[1],
            n_items=len(item_embs),
        )
        cm_static.set_item_embeddings(item_embs)
        cm_static.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

        static_results = []
        prev_assignments = None
        for w, window in enumerate(windows):
            # Build GT for this window
            w_gt = defaultdict(set)
            for uid, iid, r in zip(window["user_ids"], window["item_ids"], window["ratings"]):
                if r >= cfg["min_rating_gt"]:
                    w_gt[int(uid)].add(int(iid))
            w_gt = dict(w_gt)

            # Evaluate
            ndcg_scores = []
            for uid in set(int(u) for u in window["user_ids"]):
                if uid in w_gt:
                    recs = model.recommend(uid, n=10)
                    ndcg_scores.append(RecommendationMetrics.ndcg_at_k(recs, w_gt[uid], 10))

            # Assignment drift
            curr_assignments = {}
            for uid in set(int(u) for u in window["user_ids"]):
                info = cm_static.get_user_cluster(uid)
                curr_assignments[uid] = info.cluster_id

            drift = 0.0
            if prev_assignments is not None:
                common = set(curr_assignments.keys()) & set(prev_assignments.keys())
                if common:
                    drift = sum(1 for u in common if curr_assignments[u] != prev_assignments[u]) / len(common)

            static_results.append({
                "window": w,
                "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0,
                "drift": drift,
                "n_users": len(set(int(u) for u in window["user_ids"])),
            })
            prev_assignments = curr_assignments
            print(f"    W{w}: NDCG={static_results[-1]['ndcg']:.4f}  drift={drift:.4f}")

        strategies["static"] = static_results

        # --- Online re-clustering (partial_fit after each window) ---
        print("\n  Strategy: Online re-clustering (partial_fit)")
        cm_online = UserClusterManager(
            n_clusters=n_clusters, embedding_dim=item_embs.shape[1],
            n_items=len(item_embs),
        )
        cm_online.set_item_embeddings(item_embs)
        cm_online.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

        online_results = []
        prev_assignments = None
        for w, window in enumerate(windows):
            w_gt = defaultdict(set)
            for uid, iid, r in zip(window["user_ids"], window["item_ids"], window["ratings"]):
                if r >= cfg["min_rating_gt"]:
                    w_gt[int(uid)].add(int(iid))
            w_gt = dict(w_gt)

            ndcg_scores = []
            for uid in set(int(u) for u in window["user_ids"]):
                if uid in w_gt:
                    recs = model.recommend(uid, n=10)
                    ndcg_scores.append(RecommendationMetrics.ndcg_at_k(recs, w_gt[uid], 10))

            curr_assignments = {}
            for uid in set(int(u) for u in window["user_ids"]):
                info = cm_online.get_user_cluster(uid)
                curr_assignments[uid] = info.cluster_id

            drift = 0.0
            if prev_assignments is not None:
                common = set(curr_assignments.keys()) & set(prev_assignments.keys())
                if common:
                    drift = sum(1 for u in common if curr_assignments[u] != prev_assignments[u]) / len(common)

            online_results.append({
                "window": w,
                "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0,
                "drift": drift,
                "n_users": len(set(int(u) for u in window["user_ids"])),
            })
            prev_assignments = curr_assignments
            print(f"    W{w}: NDCG={online_results[-1]['ndcg']:.4f}  drift={drift:.4f}")

            # Partial fit on this window's interactions
            for uid, iid, rating in zip(window["user_ids"], window["item_ids"], window["ratings"]):
                cm_online.update_user_behavior(int(uid), int(iid), float(rating), update_cluster=False)
            # Force cluster update
            cm_online._update_clusters()

        strategies["online"] = online_results

        # --- Full retrain after each window ---
        print("\n  Strategy: Full retrain after each window")
        retrain_results = []
        prev_assignments = None
        accumulated_users = train.user_ids.tolist()
        accumulated_items = train.item_ids.tolist()
        accumulated_ratings = train.ratings.tolist()

        for w, window in enumerate(windows):
            w_gt = defaultdict(set)
            for uid, iid, r in zip(window["user_ids"], window["item_ids"], window["ratings"]):
                if r >= cfg["min_rating_gt"]:
                    w_gt[int(uid)].add(int(iid))
            w_gt = dict(w_gt)

            # Retrain clustering from scratch on accumulated data
            cm_retrain = UserClusterManager(
                n_clusters=n_clusters, embedding_dim=item_embs.shape[1],
                n_items=len(item_embs),
            )
            cm_retrain.set_item_embeddings(item_embs)
            cm_retrain.initialize_from_interactions(
                np.array(accumulated_users, dtype=np.int32),
                np.array(accumulated_items, dtype=np.int32),
                np.array(accumulated_ratings, dtype=np.float32),
            )

            ndcg_scores = []
            for uid in set(int(u) for u in window["user_ids"]):
                if uid in w_gt:
                    recs = model.recommend(uid, n=10)
                    ndcg_scores.append(RecommendationMetrics.ndcg_at_k(recs, w_gt[uid], 10))

            curr_assignments = {}
            for uid in set(int(u) for u in window["user_ids"]):
                info = cm_retrain.get_user_cluster(uid)
                curr_assignments[uid] = info.cluster_id

            drift = 0.0
            if prev_assignments is not None:
                common = set(curr_assignments.keys()) & set(prev_assignments.keys())
                if common:
                    drift = sum(1 for u in common if curr_assignments[u] != prev_assignments[u]) / len(common)

            retrain_results.append({
                "window": w,
                "ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0,
                "drift": drift,
                "n_users": len(set(int(u) for u in window["user_ids"])),
            })
            prev_assignments = curr_assignments
            print(f"    W{w}: NDCG={retrain_results[-1]['ndcg']:.4f}  drift={drift:.4f}")

            # Accumulate this window's data
            accumulated_users.extend(int(u) for u in window["user_ids"])
            accumulated_items.extend(int(i) for i in window["item_ids"])
            accumulated_ratings.extend(float(r) for r in window["ratings"])

        strategies["full_retrain"] = retrain_results

        results[ds_name] = strategies

    return results


# ===========================================================================
# Figure generation
# ===========================================================================
def generate_figures(all_results, fig_dir="paper/figures"):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Group D heatmap ---
    if "D" in all_results:
        for ds_name, ds_data in all_results["D"].items():
            heatmap = ds_data["heatmap"]
            k_values = ds_data["k_values"]
            cache_sizes = ds_data["cache_sizes"]

            # Build NDCG matrix
            ndcg_matrix = np.zeros((len(k_values), len(cache_sizes)))
            hr_matrix = np.zeros((len(k_values), len(cache_sizes)))
            for i, k in enumerate(k_values):
                for j, cs in enumerate(cache_sizes):
                    key = f"{k}_{cs}"
                    ndcg_matrix[i, j] = heatmap[key]["ndcg"]
                    hr_matrix[i, j] = heatmap[key]["hit_rate"]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Hit rate heatmap
            im0 = axes[0].imshow(hr_matrix, aspect="auto", cmap="YlOrRd")
            axes[0].set_xticks(range(len(cache_sizes)))
            axes[0].set_xticklabels(cache_sizes, fontsize=8)
            axes[0].set_yticks(range(len(k_values)))
            axes[0].set_yticklabels(k_values, fontsize=8)
            axes[0].set_xlabel("Cache Size", fontsize=11)
            axes[0].set_ylabel("Number of Clusters (K)", fontsize=11)
            axes[0].set_title("Hit Rate", fontsize=12, fontweight="bold")
            for i in range(len(k_values)):
                for j in range(len(cache_sizes)):
                    axes[0].text(j, i, f"{hr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
            plt.colorbar(im0, ax=axes[0])

            # NDCG heatmap
            im1 = axes[1].imshow(ndcg_matrix, aspect="auto", cmap="YlGn")
            axes[1].set_xticks(range(len(cache_sizes)))
            axes[1].set_xticklabels(cache_sizes, fontsize=8)
            axes[1].set_yticks(range(len(k_values)))
            axes[1].set_yticklabels(k_values, fontsize=8)
            axes[1].set_xlabel("Cache Size", fontsize=11)
            axes[1].set_ylabel("Number of Clusters (K)", fontsize=11)
            axes[1].set_title("NDCG@10", fontsize=12, fontweight="bold")
            for i in range(len(k_values)):
                for j in range(len(cache_sizes)):
                    axes[1].text(j, i, f"{ndcg_matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)
            plt.colorbar(im1, ax=axes[1])

            plt.suptitle(f"Capacity Sweep ({ds_name.upper()})", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / "capacity_sweep_heatmap.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / "capacity_sweep_heatmap.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: capacity_sweep_heatmap.pdf")

            # Line plot: RecCache vs LRU across cache sizes
            fig, ax = plt.subplots(figsize=(7, 4.5))
            lru_hrs = [ds_data["lru_baseline"][str(cs)]["hit_rate"] for cs in cache_sizes]
            # Use K=50 for RecCache line
            k50_idx = k_values.index(50) if 50 in k_values else 0
            rc_hrs = [hr_matrix[k50_idx, j] for j in range(len(cache_sizes))]

            ax.plot(cache_sizes, lru_hrs, "s--", linewidth=2, markersize=7,
                    color="#7f7f7f", label="LRU (no clustering)")
            ax.plot(cache_sizes, rc_hrs, "o-", linewidth=2, markersize=7,
                    color="#2ca02c", label="RecCache (K=50)")
            ax.set_xlabel("Cache Size", fontsize=11)
            ax.set_ylabel("Hit Rate", fontsize=11)
            ax.set_xscale("log")
            ax.set_title(f"RecCache vs LRU ({ds_name.upper()})", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / "reccache_vs_lru_cachesize.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / "reccache_vs_lru_cachesize.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: reccache_vs_lru_cachesize.pdf")

    # --- Group F: online re-clustering over time ---
    if "F" in all_results:
        for ds_name, ds_data in all_results["F"].items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            colors = {"static": "#d62728", "online": "#2ca02c", "full_retrain": "#1f77b4"}
            labels = {"static": "Static", "online": "Online (partial_fit)", "full_retrain": "Full Retrain"}

            for strategy, strat_results in ds_data.items():
                windows_x = [r["window"] for r in strat_results]
                ndcgs = [r["ndcg"] for r in strat_results]
                drifts = [r["drift"] for r in strat_results]

                axes[0].plot(windows_x, ndcgs, "o-", linewidth=2, markersize=7,
                            color=colors[strategy], label=labels[strategy])
                axes[1].plot(windows_x, drifts, "s--", linewidth=2, markersize=7,
                            color=colors[strategy], label=labels[strategy])

            axes[0].set_xlabel("Temporal Window", fontsize=11)
            axes[0].set_ylabel("NDCG@10", fontsize=11)
            axes[0].set_title("Recommendation Quality Over Time", fontsize=12, fontweight="bold")
            axes[0].legend(fontsize=9)
            axes[0].set_xticks([0, 1, 2, 3])
            axes[0].set_xticklabels(["W0", "W1", "W2", "W3"])

            axes[1].set_xlabel("Temporal Window", fontsize=11)
            axes[1].set_ylabel("Assignment Drift (%)", fontsize=11)
            axes[1].set_title("Cluster Assignment Drift", fontsize=12, fontweight="bold")
            axes[1].legend(fontsize=9)
            axes[1].set_xticks([0, 1, 2, 3])
            axes[1].set_xticklabels(["W0", "W1", "W2", "W3"])

            plt.suptitle(f"Online Re-Clustering ({ds_name.upper()})", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / "online_reclustering.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / "online_reclustering.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: online_reclustering.pdf")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="RecCache Supplemental Experiments")
    parser.add_argument(
        "--group", nargs="*", default=None,
        help="Which experiment groups to run (A B C D E F). Default: all.",
    )
    args = parser.parse_args()

    groups_to_run = [g.upper() for g in args.group] if args.group else ["A", "B", "C", "D", "E", "F"]

    print("=" * 70)
    print("RecCache Supplemental Experiments")
    print(f"Groups: {', '.join(groups_to_run)}")
    print("=" * 70, flush=True)

    all_results = {}
    start_time = time.time()

    group_functions = {
        "A": run_group_a,
        "B": run_group_b,
        "C": run_group_c,
        "D": run_group_d,
        "E": run_group_e,
        "F": run_group_f,
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

    results_path = Path("results/supplemental_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("Supplemental experiments complete!")


if __name__ == "__main__":
    main()
