#!/usr/bin/env python3
"""
S13 — LightGCN Backbone Experiment.

Tests speculative recommendation serving with LightGCN as the target model.
Uses LightGCN's learned graph-convolutional user/item embeddings for both
clustering and score-ratio acceptance. Compared against MF baseline.

Key question: does MCG remain ~50% across GNN geometry (structural property)?

Usage:
    conda activate reccache
    python scripts/run_lightgcn_experiments.py
    python scripts/run_lightgcn_experiments.py --dataset ml-1m
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
from reccache.models.baselines import LightGCNRecommender
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig, SpeculativeResult
from reccache.evaluation.metrics import RecommendationMetrics, compute_coverage


DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None, "min_user": 5, "min_item": 5, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64, "epochs_mf": 15, "epochs_lgcn": 10,
        "n_test_users": 500,
    },
    "amazon-electronics": {
        "max_samples": 1_000_000, "min_user": 3, "min_item": 3, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64, "epochs_mf": 15, "epochs_lgcn": 10,
        "n_test_users": 500,
    },
}

TOP_K = 3
THRESHOLD = 0.35
N_RECS = 10
N_RUNS = 3


def set_seed(seed):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def load_data(dataset_name, cfg):
    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        dataset_name,
        max_samples=cfg["max_samples"],
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
    )
    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))
    min_r = cfg["min_rating_gt"]
    test_lookup = defaultdict(list)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= min_r:
            test_lookup[int(uid)].append(int(iid))
    return train, test, dict(user_history), test_lookup


def get_test_users(test_lookup, n, seed):
    rng = np.random.default_rng(seed)
    uids = [u for u, items in test_lookup.items() if len(items) > 0]
    if len(uids) > n:
        uids = rng.choice(uids, size=n, replace=False).tolist()
    return uids


def aggregate(results, test_lookup, n_items, n_recs=10):
    ndcgs, hrs, lats, accepts, ranks = [], [], [], [], []
    for r in results:
        gt = set(test_lookup.get(r.user_id, []))
        if not gt: continue
        ndcgs.append(RecommendationMetrics.ndcg_at_k(r.items, gt, k=n_recs))
        hrs.append(RecommendationMetrics.hit_rate(r.items, gt, k=n_recs))
        lats.append(r.latency_ms)
        accepts.append(int(r.accepted))
        if r.accepted: ranks.append(r.accepted_cluster_rank)
    accept_rate = float(np.mean(accepts)) if accepts else 0.0
    mcg = float(np.mean([v > 0 for v in ranks])) if ranks else 0.0
    cov = compute_coverage({r.user_id: r.items for r in results}, n_items)
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hr": float(np.mean(hrs)) if hrs else 0.0,
        "accept_rate": accept_rate,
        "mean_latency_ms": float(np.mean(lats)) if lats else 1.0,
        "mcg": mcg, "coverage": cov, "n": len(ndcgs),
    }


def build_cm_from_model_embeddings(model, train, cfg):
    """Build cluster manager directly from model's user/item embeddings."""
    item_embs = model.get_all_item_embeddings()
    cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=cfg["embedding_dim"])
    cm.set_item_embeddings(item_embs)

    # Seed with interaction-based user embeddings first, then override with model embeddings
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # For LightGCN: override _user_embeddings with learned GCN user embeddings,
    # then re-initialize clusterer. Cluster assignments are computed on-the-fly
    # by clusterer.predict() — no separate assignment dict needed.
    if hasattr(model, "_cached_user_emb") and model._cached_user_emb is not None:
        user_embs = model._cached_user_emb  # (n_users, dim)
        for uid in range(len(user_embs)):
            cm._user_embeddings[uid] = user_embs[uid].astype(np.float32)
        # Re-initialize clusterer with GCN user embeddings
        all_embs = np.array([cm._user_embeddings[i] for i in range(len(user_embs))])
        cm.clusterer.initialize(all_embs)
        print(f"    LightGCN: overrode {len(user_embs)} user embeddings from GCN, re-initialized clusters")

    return cm, item_embs


def run_speculative(model, cm, item_embs, test_users, user_history, train,
                    use_pool=False, label=""):
    config = SpeculativeConfig(
        top_k_clusters=TOP_K, acceptance_threshold=THRESHOLD,
        n_recs=N_RECS, use_pool_retrieval=use_pool, pool_size=200,
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=THRESHOLD),
        config=config, item_embeddings=item_embs, user_history=user_history,
    )
    spec.warm_cache(list(range(train.n_users)))
    results = [spec.recommend(uid) for uid in test_users]
    return results


def run_one(dataset_name, cfg, seed, device="cpu"):
    print(f"  seed={seed}")
    set_seed(seed)

    train, test, user_history, test_lookup = load_data(dataset_name, cfg)
    n_users, n_items = train.n_users, train.n_items
    test_users = get_test_users(test_lookup, cfg["n_test_users"], seed)

    # ---- MF baseline ----
    # MF trains on CPU (nn.Parameter.to(cuda) bug); LightGCN uses GPU
    print("    Training MF (cpu)...")
    mf = MatrixFactorizationRecommender(n_users=n_users, n_items=n_items,
                                         embedding_dim=cfg["embedding_dim"], device="cpu")
    mf.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs_mf"])
    mf_item_embs = mf.get_all_item_embeddings()
    mf_cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=cfg["embedding_dim"])
    mf_cm.set_item_embeddings(mf_item_embs)
    mf_cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    mf_fresh = []
    for uid in test_users:
        exclude = user_history.get(uid)
        t0 = time.time()
        items = list(mf.recommend(uid, n=N_RECS, exclude_items=exclude))
        lat = (time.time() - t0) * 1000
        mf_fresh.append(SpeculativeResult(
            user_id=uid, items=items, accepted=False, acceptance_prob=0.0,
            accepted_cluster_id=None, accepted_cluster_rank=-1, latency_ms=lat, phase="fresh"
        ))
    mf_fresh_m = aggregate(mf_fresh, test_lookup, n_items)

    mf_spec = run_speculative(mf, mf_cm, mf_item_embs, test_users, user_history, train)
    mf_spec_m = aggregate(mf_spec, test_lookup, n_items)
    fn = mf_fresh_m["ndcg"]
    mf_spec_m["retention"] = mf_spec_m["ndcg"] / fn if fn > 0 else 0
    mf_spec_m["speedup"] = mf_fresh_m["mean_latency_ms"] / mf_spec_m["mean_latency_ms"]
    print(f"    MF  fresh={fn:.4f}, spec={mf_spec_m['ndcg']:.4f} "
          f"(ret={mf_spec_m['retention']:.0%}), acc={mf_spec_m['accept_rate']:.0%}, "
          f"MCG={mf_spec_m['mcg']:.0%}")

    # ---- LightGCN ----
    print("    Training LightGCN...")
    lgcn = LightGCNRecommender(n_users=n_users, n_items=n_items,
                                embedding_dim=cfg["embedding_dim"], n_layers=3, device=device)
    lgcn.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs_lgcn"])
    lgcn_cm, lgcn_item_embs = build_cm_from_model_embeddings(lgcn, train, cfg)

    lgcn_fresh = []
    for uid in test_users:
        exclude = user_history.get(uid)
        t0 = time.time()
        items = list(lgcn.recommend(uid, n=N_RECS, exclude_items=exclude))
        lat = (time.time() - t0) * 1000
        lgcn_fresh.append(SpeculativeResult(
            user_id=uid, items=items, accepted=False, acceptance_prob=0.0,
            accepted_cluster_id=None, accepted_cluster_rank=-1, latency_ms=lat, phase="fresh"
        ))
    lgcn_fresh_m = aggregate(lgcn_fresh, test_lookup, n_items)

    lgcn_spec = run_speculative(lgcn, lgcn_cm, lgcn_item_embs, test_users, user_history, train)
    lgcn_spec_m = aggregate(lgcn_spec, test_lookup, n_items)
    fn_lgcn = lgcn_fresh_m["ndcg"]
    lgcn_spec_m["retention"] = lgcn_spec_m["ndcg"] / fn_lgcn if fn_lgcn > 0 else 0
    lgcn_spec_m["speedup"] = lgcn_fresh_m["mean_latency_ms"] / lgcn_spec_m["mean_latency_ms"]
    print(f"    LGN fresh={fn_lgcn:.4f}, spec={lgcn_spec_m['ndcg']:.4f} "
          f"(ret={lgcn_spec_m['retention']:.0%}), acc={lgcn_spec_m['accept_rate']:.0%}, "
          f"MCG={lgcn_spec_m['mcg']:.0%}")

    return {
        "mf":   {"fresh": mf_fresh_m,   "spec": mf_spec_m},
        "lgcn": {"fresh": lgcn_fresh_m, "spec": lgcn_spec_m},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(exist_ok=True, parents=True)

    all_results = {}
    for ds in args.dataset:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n{'='*60}\nS13: LightGCN | {ds}\n{'='*60}")

        seed_results = []
        for seed in args.seeds:
            seed_results.append(run_one(ds, cfg, seed, args.device))

        def avg_m(model, method, key):
            return float(np.mean([r[model][method].get(key, 0) for r in seed_results]))

        averaged = {}
        for model in ["mf", "lgcn"]:
            for method in ["fresh", "spec"]:
                key = f"{model}_{method}"
                all_k = list(seed_results[0][model][method].keys())
                averaged[key] = {k: avg_m(model, method, k) for k in all_k}
        all_results[ds] = averaged

        print(f"\n  {ds} Summary:")
        print(f"  {'Model/Method':<28} {'NDCG':>8} {'Ret':>7} {'Accept':>8} {'MCG':>6}")
        print("  " + "-"*65)
        for model, label in [("mf","MF"),("lgcn","LightGCN")]:
            fresh = averaged[f"{model}_fresh"]
            spec  = averaged[f"{model}_spec"]
            fn    = fresh["ndcg"]
            print(f"  {label+' Fresh':<28} {fn:>8.4f} {'—':>7} {'—':>8} {'—':>6}")
            print(f"  {label+' Spec K=3':<28} {spec['ndcg']:>8.4f} "
                  f"{spec['retention']:>7.1%} {spec['accept_rate']:>8.1%} {spec['mcg']:>6.1%}")

    out = results_dir / "s13_lightgcn.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    # MCG comparison figure
    if all_results:
        datasets = list(all_results.keys())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(datasets))
        w = 0.3
        for ax, metric in zip(axes, ["mcg", "accept_rate"]):
            mf_vals   = [all_results[ds]["mf_spec"].get(metric, 0)   for ds in datasets]
            lgcn_vals = [all_results[ds]["lgcn_spec"].get(metric, 0) for ds in datasets]
            ax.bar(x - w/2, mf_vals,   w, label="MF")
            ax.bar(x + w/2, lgcn_vals, w, label="LightGCN")
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=15)
            ax.set_ylabel(metric); ax.legend()
            ax.set_title(f"S13: {metric} — MF vs LightGCN")
        plt.tight_layout()
        fig_path = figures_dir / "s13_lightgcn.pdf"
        plt.savefig(fig_path, bbox_inches="tight"); plt.close()
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
