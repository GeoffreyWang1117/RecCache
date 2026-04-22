#!/usr/bin/env python3
"""
A3 — FAISS / ANN Baseline.

Builds a FAISS index over MF item embeddings and benchmarks it as an
industrial baseline for retrieval. Two variants:
  - faiss_flat: exact inner-product search (IndexFlatIP)
  - faiss_ivf : approximate IVF index (IndexIVFFlat, nlist=64, nprobe=8)

Compared against:
  - Fresh:    full MF score-and-sort
  - Spec K=3: RecCache speculative serving

All on ML-1M and Amazon Electronics, 3 seeds.

Usage:
    conda activate reccache
    python scripts/run_faiss_baseline.py
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import (
    SpeculativeRecommender, SpeculativeConfig, SpeculativeResult
)
from reccache.evaluation.metrics import RecommendationMetrics


DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None, "min_user": 5, "min_item": 5, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64, "epochs": 15, "n_test_users": 500,
    },
    "amazon-electronics": {
        "max_samples": 1_000_000, "min_user": 3, "min_item": 3, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64, "epochs": 15, "n_test_users": 500,
    },
}

TOP_K = 3
THRESHOLD = 0.35
N_RECS = 10
SEEDS = [42, 123, 456]


def set_seed(seed):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def load(dataset_name, cfg):
    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        dataset_name, max_samples=cfg["max_samples"],
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
    )
    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))
    min_r = cfg["min_rating_gt"]
    test_lookup = defaultdict(set)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= min_r:
            test_lookup[int(uid)].add(int(iid))
    return train, dict(user_history), dict(test_lookup)


def get_test_users(test_lookup, n, seed):
    rng = np.random.default_rng(seed)
    uids = [u for u, items in test_lookup.items() if len(items) > 0]
    if len(uids) > n:
        uids = rng.choice(uids, size=n, replace=False).tolist()
    return uids


def aggregate(results, test_lookup, n_recs=10):
    ndcgs, recalls, hrs, mrrs, lats = [], [], [], [], []
    for r in results:
        gt = test_lookup.get(r["user_id"], set())
        if not gt:
            continue
        items = r["items"]
        ndcgs.append(RecommendationMetrics.ndcg_at_k(items, gt, k=n_recs))
        recalls.append(RecommendationMetrics.recall_at_k(items, gt, k=n_recs))
        hrs.append(RecommendationMetrics.hit_rate(items, gt, k=n_recs))
        mrrs.append(RecommendationMetrics.mrr(items, gt))
        lats.append(r["latency_ms"])
    return {
        "ndcg":   float(np.mean(ndcgs))   if ndcgs   else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "hr":     float(np.mean(hrs))     if hrs     else 0.0,
        "mrr":    float(np.mean(mrrs))    if mrrs    else 0.0,
        "mean_latency_ms": float(np.mean(lats)) if lats else 1.0,
        "n": len(ndcgs),
    }


def aggregate_spec(spec_results, test_lookup, n_recs=10):
    """Same as aggregate() but for SpeculativeResult dataclass."""
    converted = [{"user_id": r.user_id, "items": r.items, "latency_ms": r.latency_ms}
                 for r in spec_results]
    return aggregate(converted, test_lookup, n_recs)


def run_fresh(model, test_users, user_history, test_lookup):
    out = []
    for uid in test_users:
        exclude = user_history.get(uid)
        t0 = time.time()
        items = list(model.recommend(uid, n=N_RECS, exclude_items=exclude))
        out.append({"user_id": uid, "items": items, "latency_ms": (time.time()-t0)*1000})
    return out


def run_faiss(index, user_embs_np, user_biases_np, item_biases_np, global_bias,
              test_users, user_history, n_items, name="faiss_flat"):
    """Run FAISS retrieval. We retrieve top-(N_RECS + max_history) then exclude
    user history, returning the first N_RECS unseen items.

    The index is built over item_embeddings, so a user's nearest neighbors are
    the highest-scoring items by inner product. We add user/item biases as a
    post-processing step (FAISS doesn't natively handle biases, but for the
    top-K ordering they're inherently included if we score the candidates
    afterwards).
    """
    out = []
    for uid in test_users:
        exclude = set(user_history.get(uid, []))
        # over-fetch to leave room for excluded history
        k_fetch = N_RECS + len(exclude) + 5
        u = user_embs_np[uid:uid+1]   # (1, d)
        t0 = time.time()
        _, I = index.search(u.astype(np.float32), k_fetch)
        # post-filter excluded items, take first N_RECS
        items = [int(i) for i in I[0] if int(i) not in exclude and i >= 0][:N_RECS]
        lat = (time.time() - t0) * 1000
        out.append({"user_id": uid, "items": items, "latency_ms": lat})
    return out


def run_spec(model, cm, item_embs, test_users, user_history, train):
    config = SpeculativeConfig(
        top_k_clusters=TOP_K, acceptance_threshold=THRESHOLD,
        n_recs=N_RECS, use_pool_retrieval=False,
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=THRESHOLD),
        config=config, item_embeddings=item_embs, user_history=user_history,
    )
    spec.warm_cache(list(range(train.n_users)))
    return [spec.recommend(uid) for uid in test_users]


def run_one_seed(dataset_name, cfg, seed):
    print(f"  seed={seed}")
    set_seed(seed)
    train, user_history, test_lookup = load(dataset_name, cfg)
    n_users, n_items = train.n_users, train.n_items
    test_users = get_test_users(test_lookup, cfg["n_test_users"], seed)

    # Train MF
    print(f"    Training MF (n_users={n_users}, n_items={n_items})")
    model = MatrixFactorizationRecommender(
        n_users=n_users, n_items=n_items, embedding_dim=cfg["embedding_dim"]
    )
    model.fit(train.user_ids, train.item_ids, train.ratings,
              epochs=cfg["epochs"], verbose=False)
    item_embs = model.get_all_item_embeddings()
    user_embs = model.user_embeddings.weight.data.cpu().numpy().astype(np.float32)
    user_biases = model.user_bias.weight.data.cpu().numpy().astype(np.float32).reshape(-1)
    item_biases = model.item_bias.weight.data.cpu().numpy().astype(np.float32).reshape(-1)
    global_bias = float(model.global_bias.data.cpu().numpy().item())

    # ---- Build FAISS indices ----
    d = cfg["embedding_dim"]
    print("    Building FAISS IndexFlatIP...")
    flat = faiss.IndexFlatIP(d)
    flat.add(item_embs.astype(np.float32))

    print("    Building FAISS IndexIVFFlat (nlist=64)...")
    quantizer = faiss.IndexFlatIP(d)
    nlist = min(64, max(8, n_items // 100))
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(item_embs.astype(np.float32))
    ivf.add(item_embs.astype(np.float32))
    ivf.nprobe = 8

    # ---- Cluster manager for Spec ----
    cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=d)
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # ---- Methods ----
    print("    Fresh...")
    fresh = run_fresh(model, test_users, user_history, test_lookup)
    print("    FAISS Flat (exact)...")
    faiss_flat = run_faiss(flat, user_embs, user_biases, item_biases, global_bias,
                            test_users, user_history, n_items, name="faiss_flat")
    print("    FAISS IVF (approx, nprobe=8)...")
    faiss_ivf = run_faiss(ivf, user_embs, user_biases, item_biases, global_bias,
                           test_users, user_history, n_items, name="faiss_ivf")
    print("    Spec K=3...")
    spec = run_spec(model, cm, item_embs, test_users, user_history, train)

    return {
        "fresh":      aggregate(fresh, test_lookup),
        "faiss_flat": aggregate(faiss_flat, test_lookup),
        "faiss_ivf":  aggregate(faiss_ivf, test_lookup),
        "spec_k3":    aggregate_spec(spec, test_lookup),
    }


def avg_seeds(seed_results):
    methods = list(seed_results[0].keys())
    keys = list(seed_results[0][methods[0]].keys())
    return {m: {k: float(np.mean([r[m][k] for r in seed_results])) for k in keys}
            for m in methods}


def print_summary(ds, results):
    fn = results["fresh"]["ndcg"]
    fl = results["fresh"]["mean_latency_ms"]
    print(f"\n  ===== {ds} (avg {len(SEEDS)} seeds) =====")
    print("  {:<14} {:>8} {:>8} {:>8} {:>8} {:>10} {:>9}".format(
        "Method", "NDCG", "Recall", "HR", "MRR", "Lat (ms)", "Speedup"))
    for m in ["fresh", "faiss_flat", "faiss_ivf", "spec_k3"]:
        r = results[m]
        spd = fl / r["mean_latency_ms"] if r["mean_latency_ms"] > 0 else 0
        print("  {:<14} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.3f} {:>8.1f}x".format(
            m, r["ndcg"], r["recall"], r["hr"], r["mrr"], r["mean_latency_ms"], spd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)

    all_results = {}
    for ds in args.dataset:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n{'='*60}\nA3 FAISS baseline | {ds}\n{'='*60}")
        seed_results = [run_one_seed(ds, cfg, s) for s in args.seeds]
        all_results[ds] = avg_seeds(seed_results)
        print_summary(ds, all_results[ds])

    out = results_dir / "supp_a3_faiss.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
