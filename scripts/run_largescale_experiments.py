#!/usr/bin/env python3
"""
S11/S12 — Large-Scale Benchmark Experiments.

S11: MIND-large  (~94K users, 1.2M interactions after 5-core)
S12: Amazon Movies-and-TV  (sampled 1M interactions, ~200K users after 5-core)

Runs the standard S4 end-to-end comparison (Fresh / Naive / Speculative K=3)
on large-scale datasets to demonstrate RecCache scales beyond ML-1M.

Usage:
    conda activate reccache
    python scripts/run_largescale_experiments.py
    python scripts/run_largescale_experiments.py --dataset mind-large
    python scripts/run_largescale_experiments.py --dataset amazon-movies
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
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig, SpeculativeResult
from reccache.evaluation.metrics import RecommendationMetrics, SpeculativeMetrics, compute_coverage

EXTERNAL_DATA_DIR = Path.home() / "DataSets"

DATASET_CONFIGS = {
    "mind-large": {
        # 90M raw interactions. Early-stop parsing at 3M rows (train: 2.4M, dev: 0.6M)
        # Expected after 5-core: ~50K users, ~20K articles — sufficient for large-scale claim
        "max_samples": 3_000_000,
        "min_user": 5, "min_item": 5,
        "min_rating_gt": 0.5,
        "n_clusters": 100, "embedding_dim": 64, "epochs": 15,
        "n_test_users": 1000,
    },
    "amazon-movies": {
        # 17M raw; sample 2M gives ~3 per user → need lighter filter
        "max_samples": 2_000_000,
        "min_user": 3, "min_item": 5,
        "min_rating_gt": 4.0,
        "n_clusters": 100, "embedding_dim": 64, "epochs": 15,
        "n_test_users": 1000,
    },
}

N_RUNS = 3
TOP_K = 3
THRESHOLD = 0.35
N_RECS = 10


def set_seed(seed):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def load_data(dataset_name, cfg):
    loader = DataLoader("data", external_data_dir=str(EXTERNAL_DATA_DIR))
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

    return train, val, test, dict(user_history), test_lookup


def get_test_users(test_lookup, n, seed):
    rng = np.random.default_rng(seed)
    uids = [u for u, items in test_lookup.items() if len(items) > 0]
    if len(uids) > n:
        uids = rng.choice(uids, size=n, replace=False).tolist()
    return uids


def evaluate_fresh(model, test_users, test_lookup, user_history, item_embs, n_recs=10):
    results = []
    for uid in test_users:
        exclude = user_history.get(uid)
        t0 = time.time()
        items = list(model.recommend(uid, n=n_recs, exclude_items=exclude))
        lat = (time.time() - t0) * 1000
        results.append(SpeculativeResult(
            user_id=uid, items=items, accepted=False, acceptance_prob=0.0,
            accepted_cluster_id=None, accepted_cluster_rank=-1, latency_ms=lat, phase="fresh"
        ))
    return aggregate(results, test_lookup, item_embs, n_recs)


def aggregate(results, test_lookup, item_embs, n_recs=10):
    ndcgs, hrs, lats, accepts, ranks = [], [], [], [], []
    for r in results:
        gt = set(test_lookup.get(r.user_id, []))
        if not gt:
            continue
        ndcgs.append(RecommendationMetrics.ndcg_at_k(r.items, gt, k=n_recs))
        hrs.append(RecommendationMetrics.hit_rate(r.items, gt, k=n_recs))
        lats.append(r.latency_ms)
        accepts.append(int(r.accepted))
        if r.accepted:
            ranks.append(r.accepted_cluster_rank)

    accept_rate = float(np.mean(accepts)) if accepts else 0.0
    mcg = float(np.mean([v > 0 for v in ranks])) if ranks else 0.0
    mean_lat = float(np.mean(lats)) if lats else 1.0
    cov = compute_coverage({r.user_id: r.items for r in results}, item_embs.shape[0])

    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hr": float(np.mean(hrs)) if hrs else 0.0,
        "accept_rate": accept_rate,
        "mean_latency_ms": mean_lat,
        "mcg": mcg,
        "coverage": cov,
        "n": len(ndcgs),
    }


def run_one_seed(dataset_name, cfg, seed):
    print(f"  seed={seed}")
    set_seed(seed)

    train, val, test, user_history, test_lookup = load_data(dataset_name, cfg)
    print(f"    {train.n_users} users, {train.n_items} items, "
          f"{len(train.user_ids)} train interactions, {len(test_lookup)} GT users")

    # Train MF
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items,
        embedding_dim=cfg["embedding_dim"]
    )
    model.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"])
    item_embs = model.get_all_item_embeddings()

    # Cluster manager
    cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=cfg["embedding_dim"])
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    test_users = get_test_users(test_lookup, cfg["n_test_users"], seed)

    # Fresh
    fresh = evaluate_fresh(model, test_users, test_lookup, user_history, item_embs)
    fresh_ndcg = fresh["ndcg"]
    print(f"    Fresh  NDCG={fresh_ndcg:.4f}, lat={fresh['mean_latency_ms']:.2f}ms")

    # Speculative K=3
    config = SpeculativeConfig(top_k_clusters=TOP_K, acceptance_threshold=THRESHOLD,
                               n_recs=N_RECS, use_pool_retrieval=False)
    criterion = ScoreRatioAcceptanceCriterion(threshold=THRESHOLD)
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm, acceptance_criterion=criterion,
        config=config, item_embeddings=item_embs, user_history=user_history,
    )
    spec.warm_cache(list(range(train.n_users)))

    spec_results = [spec.recommend(uid) for uid in test_users]
    spec_m = aggregate(spec_results, test_lookup, item_embs)
    spec_m["retention"] = spec_m["ndcg"] / fresh_ndcg if fresh_ndcg > 0 else 0
    # Empirical speedup
    fresh_lat = fresh["mean_latency_ms"]
    spec_lat = spec_m["mean_latency_ms"]
    spec_m["speedup"] = fresh_lat / spec_lat if spec_lat > 0 else 1.0
    print(f"    Spec   NDCG={spec_m['ndcg']:.4f} "
          f"(ret={spec_m['retention']:.0%}), accept={spec_m['accept_rate']:.0%}, "
          f"MCG={spec_m['mcg']:.0%}, speedup={spec_m['speedup']:.1f}x")

    # Spec+Pool
    config_pool = SpeculativeConfig(top_k_clusters=TOP_K, acceptance_threshold=THRESHOLD,
                                    n_recs=N_RECS, use_pool_retrieval=True, pool_size=200)
    spec_pool = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=THRESHOLD),
        config=config_pool, item_embeddings=item_embs, user_history=user_history,
    )
    spec_pool.warm_cache(list(range(train.n_users)))
    pool_results = [spec_pool.recommend(uid) for uid in test_users]
    pool_m = aggregate(pool_results, test_lookup, item_embs)
    pool_m["retention"] = pool_m["ndcg"] / fresh_ndcg if fresh_ndcg > 0 else 0
    pool_m["speedup"] = fresh_lat / pool_m["mean_latency_ms"] if pool_m["mean_latency_ms"] > 0 else 1.0
    print(f"    Pool   NDCG={pool_m['ndcg']:.4f} "
          f"(ret={pool_m['retention']:.0%}), accept={pool_m['accept_rate']:.0%}, "
          f"cov={pool_m['coverage']:.3f}")

    return {"fresh": fresh, "spec_k3": spec_m, "spec_pool": pool_m,
            "n_users": train.n_users, "n_items": train.n_items}


def avg_seeds(seed_results):
    keys_fresh = list(seed_results[0]["fresh"].keys())
    keys_spec  = list(seed_results[0]["spec_k3"].keys())
    def avg(method, key):
        vals = [r[method].get(key, 0) for r in seed_results]
        return float(np.mean(vals))
    return {
        "fresh":     {k: avg("fresh", k)     for k in keys_fresh},
        "spec_k3":   {k: avg("spec_k3", k)   for k in keys_spec},
        "spec_pool": {k: avg("spec_pool", k)  for k in keys_spec},
        "n_users": seed_results[0]["n_users"],
        "n_items":  seed_results[0]["n_items"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+",
                        default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(exist_ok=True, parents=True)

    all_results = {}
    for ds in args.dataset:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        seed_results = []
        for seed in args.seeds:
            r = run_one_seed(ds, cfg, seed)
            seed_results.append(r)

        avg = avg_seeds(seed_results)
        all_results[ds] = avg

        fn = avg["fresh"]["ndcg"]
        print(f"\n  Summary ({ds}, {len(args.seeds)} seeds avg):")
        print(f"  {'Method':<20} {'NDCG':>8} {'Retain':>8} {'Accept':>8} {'MCG':>6} {'Speedup':>8}")
        print("  " + "-"*65)
        for method, label in [("fresh","Fresh"),("spec_k3","Spec K=3"),("spec_pool","Spec+Pool")]:
            m = avg[method]
            ret  = m["ndcg"]/fn if fn>0 and method!="fresh" else 1.0
            acc  = m.get("accept_rate",0) if method!="fresh" else float("nan")
            mcg  = m.get("mcg",0) if method!="fresh" else float("nan")
            spd  = m.get("speedup",1) if method!="fresh" else 1.0
            print(f"  {label:<20} {m['ndcg']:>8.4f} {ret:>8.1%} {acc:>8.1%} {mcg:>6.1%} {spd:>8.1f}x")

    out = results_dir / "s11_s12_largescale.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    # Simple figure
    if len(all_results) > 0:
        datasets = list(all_results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (metric, label) in zip(axes, [("ndcg","NDCG@10"),("accept_rate","Accept Rate"),("mcg","MCG")]):
            for ds in datasets:
                r = all_results[ds]
                fn = r["fresh"]["ndcg"]
                xs, ys = ["Fresh","Spec K=3","Spec+Pool"], []
                for m in ["fresh","spec_k3","spec_pool"]:
                    ys.append(r[m].get(metric,0) if metric!="ndcg" else r[m]["ndcg"])
                ax.plot(xs, ys, marker="o", label=ds)
            ax.set_title(label); ax.legend(fontsize=8)
        plt.tight_layout()
        fig_path = figures_dir / "s11_s12_largescale.pdf"
        plt.savefig(fig_path, bbox_inches="tight"); plt.close()
        print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
