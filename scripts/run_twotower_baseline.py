#!/usr/bin/env python3
"""
Two-Tower Retrieve-and-Rerank Baseline.

Standard industry pipeline: a lightweight retrieval model (MF d=16/32)
generates top-100 candidates via dot-product, then the full model (MF d=64)
re-ranks to top-10. This is the "same-cost" baseline a reviewer would expect
when comparing against speculative serving.

Compared against Fresh (d=64 full scoring) and Spec K=3.

Usage:
    conda activate reccache
    python scripts/run_twotower_baseline.py
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import RecommendationMetrics

N_RECS = 10
SEEDS = [42, 123, 456]

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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    test_lookup = defaultdict(set)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= cfg["min_rating_gt"]:
            test_lookup[int(uid)].add(int(iid))
    return train, dict(user_history), dict(test_lookup)


def get_test_users(test_lookup, n, seed):
    rng = np.random.default_rng(seed)
    uids = [u for u, items in test_lookup.items() if len(items) > 0]
    if len(uids) > n:
        uids = rng.choice(uids, size=n, replace=False).tolist()
    return uids


class TwoTowerReranker:
    """Lightweight retrieval (d_small) + full re-rank (d_full)."""

    def __init__(self, retrieval_model, rerank_model, n_candidates=100, n_recs=10):
        self.retrieval = retrieval_model
        self.reranker = rerank_model
        self.n_candidates = n_candidates
        self.n_recs = n_recs

        # Pre-extract retrieval embeddings for fast dot-product
        self.ret_item_embs = retrieval_model.get_all_item_embeddings().astype(np.float32)
        self.ret_user_embs = retrieval_model.user_embeddings.weight.data.cpu().numpy().astype(np.float32)

    def recommend(self, uid, exclude=None):
        exclude_set = set(exclude) if exclude else set()

        # Stage 1: Lightweight retrieval (dot product with small embeddings)
        u_emb = self.ret_user_embs[uid]
        scores = self.ret_item_embs @ u_emb  # (n_items,)

        # Exclude history
        for i in exclude_set:
            if i < len(scores):
                scores[i] = -np.inf

        # Top-N candidates from retrieval model
        candidates = np.argsort(-scores)[:self.n_candidates].tolist()

        # Stage 2: Full model re-rank on candidates only
        with torch.no_grad():
            u_t = torch.tensor([uid], dtype=torch.long)
            c_t = torch.tensor(candidates, dtype=torch.long)
            u_emb_full = self.reranker.user_embeddings(u_t)  # (1, d_full)
            c_emb_full = self.reranker.item_embeddings(c_t)  # (n_cand, d_full)
            u_bias = self.reranker.user_bias(u_t)
            c_bias = self.reranker.item_bias(c_t).squeeze(1)
            rerank_scores = (u_emb_full @ c_emb_full.T).squeeze(0) + u_bias.squeeze() + c_bias + self.reranker.global_bias
            rerank_scores = rerank_scores.cpu().numpy()

        top_idx = np.argsort(-rerank_scores)[:self.n_recs]
        return [candidates[i] for i in top_idx]


def aggregate(results, test_lookup):
    ndcgs, recalls, hrs, mrrs, lats = [], [], [], [], []
    for r in results:
        gt = test_lookup.get(r["uid"], set())
        if not gt:
            continue
        ndcgs.append(RecommendationMetrics.ndcg_at_k(r["items"], gt, N_RECS))
        recalls.append(RecommendationMetrics.recall_at_k(r["items"], gt, N_RECS))
        hrs.append(RecommendationMetrics.hit_rate(r["items"], gt, N_RECS))
        mrrs.append(RecommendationMetrics.mrr(r["items"], gt))
        lats.append(r["lat"])
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "hr": float(np.mean(hrs)) if hrs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "latency_ms": float(np.mean(lats)) if lats else 0.0,
        "n": len(ndcgs),
    }


def run_one_seed(dataset_name, cfg, seed):
    print(f"  seed={seed}")
    set_seed(seed)
    train, user_history, test_lookup = load(dataset_name, cfg)
    n_users, n_items = train.n_users, train.n_items
    test_users = get_test_users(test_lookup, cfg["n_test_users"], seed)

    # Full model (target, d=64)
    full = MatrixFactorizationRecommender(n_users=n_users, n_items=n_items, embedding_dim=64)
    full.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"], verbose=False)
    item_embs = full.get_all_item_embeddings()

    # Retrieval models (drafts)
    ret16 = MatrixFactorizationRecommender(n_users=n_users, n_items=n_items, embedding_dim=16)
    ret16.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"], verbose=False)

    ret32 = MatrixFactorizationRecommender(n_users=n_users, n_items=n_items, embedding_dim=32)
    ret32.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"], verbose=False)

    # Two-tower pipelines
    tt16 = TwoTowerReranker(ret16, full, n_candidates=100, n_recs=N_RECS)
    tt32 = TwoTowerReranker(ret32, full, n_candidates=100, n_recs=N_RECS)

    # Spec K=3
    cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=64)
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
    spec_cfg = SpeculativeConfig(top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS)
    spec = SpeculativeRecommender(
        recommender=full, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
        config=spec_cfg, item_embeddings=item_embs, user_history=user_history,
    )
    spec.warm_cache(list(range(n_users)))

    # Run all methods
    methods = {}
    for label, fn in [
        ("Fresh", lambda uid: list(full.recommend(uid, n=N_RECS, exclude_items=user_history.get(uid)))),
        ("TwoTower-16", lambda uid: tt16.recommend(uid, user_history.get(uid))),
        ("TwoTower-32", lambda uid: tt32.recommend(uid, user_history.get(uid))),
        ("Spec K=3", lambda uid: spec.recommend(uid).items),
    ]:
        results = []
        for uid in test_users:
            t0 = time.time()
            items = fn(uid)
            lat = (time.time() - t0) * 1000
            results.append({"uid": uid, "items": items, "lat": lat})
        methods[label] = aggregate(results, test_lookup)

    return methods


def main():
    print("Two-Tower Retrieve-and-Rerank Baseline")

    all_results = {}
    for ds in DATASET_CONFIGS:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n{'='*50}\n{ds}\n{'='*50}")
        seed_results = [run_one_seed(ds, cfg, s) for s in SEEDS]

        # Average across seeds
        methods = list(seed_results[0].keys())
        keys = list(seed_results[0][methods[0]].keys())
        avg = {m: {k: float(np.mean([r[m][k] for r in seed_results])) for k in keys}
               for m in methods}
        all_results[ds] = avg

        fn = avg["Fresh"]["ndcg"]
        fl = avg["Fresh"]["latency_ms"]
        print(f"\n  {'Method':<16} {'NDCG':>8} {'Ret%':>7} {'Recall':>8} {'HR':>8} "
              f"{'Lat(ms)':>8} {'Spd':>6}")
        print("  " + "-"*65)
        for m in methods:
            r = avg[m]
            ret = r["ndcg"]/fn if fn > 0 else 0
            spd = fl/r["latency_ms"] if r["latency_ms"] > 0 else 0
            print(f"  {m:<16} {r['ndcg']:>8.4f} {ret:>7.0%} {r['recall']:>8.4f} "
                  f"{r['hr']:>8.3f} {r['latency_ms']:>8.3f} {spd:>5.1f}x")

    out = Path("results/supp_twotower_baseline.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
