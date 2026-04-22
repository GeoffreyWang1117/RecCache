#!/usr/bin/env python3
"""
Online Serving Replay Experiment.

Simulates a production recommendation serving pipeline on ML-1M using
temporal ordering of test interactions. At each timestep a user arrives,
and the system must serve recommendations using one of several policies.

Policies compared:
  - Fresh:       full MF inference every request
  - Popularity:  global top-N popular items (no personalization)
  - FAISS Flat:  exact ANN retrieval on item embeddings
  - Naive Cache: nearest-cluster cache, no acceptance test
  - Spec K=3:    speculative serving (our method)
  - Spec+Pool:   speculative + embedding pool

Metrics (rolling 200-request windows):
  - CTR proxy:   fraction of recommended items in user's future test interactions
  - NDCG@10, Recall@10, HR@10
  - Novelty:     mean(-log2(popularity(item))) of recommended items
  - Latency (ms)

Outputs: results/supp_online_serving_replay.json
         paper/figures/supp_online_serving.pdf

Usage:
    conda activate reccache
    python scripts/run_online_serving_replay.py
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict, Counter

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
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import RecommendationMetrics

N_RECS = 10
WINDOW = 200
SEED = 42


def novelty_score(items, item_pop, n_interactions):
    """Mean self-information: -log2(pop(item)/N). Higher = more novel."""
    scores = []
    for i in items:
        p = item_pop.get(i, 1) / n_interactions
        scores.append(-np.log2(p + 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def ctr_proxy(items, future_items):
    """Fraction of recommended items that user will interact with in test."""
    if not future_items or not items:
        return 0.0
    hits = sum(1 for i in items if i in future_items)
    return hits / len(items)


def build_temporal_stream(test_data, gt, timestamps=None):
    """Build test user stream in temporal order."""
    # Use interaction order in test set as temporal proxy
    seen = set()
    stream = []
    for idx in range(len(test_data.user_ids)):
        uid = int(test_data.user_ids[idx])
        if uid not in seen and uid in gt:
            stream.append(uid)
            seen.add(uid)
    return stream


class PopularityBaseline:
    """Serve globally most popular items to everyone."""
    def __init__(self, item_pop, n_items, n_recs=10):
        sorted_items = sorted(range(n_items), key=lambda i: -item_pop.get(i, 0))
        self.top_items = sorted_items[:n_recs * 5]  # over-fetch for exclusion
        self.n_recs = n_recs

    def recommend(self, uid, exclude=None):
        exclude = set(exclude) if exclude else set()
        return [i for i in self.top_items if i not in exclude][:self.n_recs]


class FAISSBaseline:
    """FAISS IndexFlatIP retrieval."""
    def __init__(self, item_embs, user_embs, n_recs=10):
        d = item_embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(item_embs.astype(np.float32))
        self.user_embs = user_embs.astype(np.float32)
        self.n_recs = n_recs

    def recommend(self, uid, exclude=None):
        exclude = set(exclude) if exclude else set()
        k = self.n_recs + len(exclude) + 5
        u = self.user_embs[uid:uid+1]
        _, I = self.index.search(u, k)
        return [int(i) for i in I[0] if int(i) not in exclude and i >= 0][:self.n_recs]


def run_policy(name, recommend_fn, stream, gt, user_history, item_pop, n_ints):
    """Run a single policy over the temporal stream."""
    ndcgs, recalls, hrs, ctrs, novs, lats = [], [], [], [], [], []
    for uid in stream:
        exclude = user_history.get(uid)
        future = gt.get(uid, set())
        t0 = time.time()
        items = recommend_fn(uid, exclude)
        lat = (time.time() - t0) * 1000

        if not items:
            items = []
        ndcgs.append(RecommendationMetrics.ndcg_at_k(items, future, N_RECS))
        recalls.append(RecommendationMetrics.recall_at_k(items, future, N_RECS))
        hrs.append(RecommendationMetrics.hit_rate(items, future, N_RECS))
        ctrs.append(ctr_proxy(items, future))
        novs.append(novelty_score(items, item_pop, n_ints))
        lats.append(lat)

    # Rolling windows
    windows = []
    for start in range(0, len(stream), WINDOW):
        end = min(start + WINDOW, len(stream))
        if end - start < WINDOW // 2:
            break
        s = slice(start, end)
        windows.append({
            "start": start, "end": end,
            "ndcg": float(np.mean(ndcgs[s])),
            "recall": float(np.mean(recalls[s])),
            "hr": float(np.mean(hrs[s])),
            "ctr": float(np.mean(ctrs[s])),
            "novelty": float(np.mean(novs[s])),
            "latency": float(np.mean(lats[s])),
        })

    overall = {
        "ndcg": float(np.mean(ndcgs)),
        "recall": float(np.mean(recalls)),
        "hr": float(np.mean(hrs)),
        "ctr": float(np.mean(ctrs)),
        "novelty": float(np.mean(novs)),
        "latency_ms": float(np.mean(lats)),
        "n": len(stream),
    }
    return {"overall": overall, "windows": windows}


def main():
    np.random.seed(SEED)
    import torch; torch.manual_seed(SEED)
    print("Online Serving Replay | ML-1M (temporal order)")

    loader = DataLoader("data")
    train, val, test = loader.load_dataset("ml-1m")
    n_users, n_items = train.n_users, train.n_items

    gt = defaultdict(set)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= 4.0:
            gt[int(uid)].add(int(iid))
    gt = dict(gt)

    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))
    user_history = dict(user_history)

    item_pop = Counter(int(i) for i in train.item_ids)
    n_ints = len(train.item_ids)

    stream = build_temporal_stream(test, gt)
    print(f"  {len(stream)} users in temporal stream")

    # Train MF
    print("  Training MF...")
    model = MatrixFactorizationRecommender(
        n_users=n_users, n_items=n_items, embedding_dim=64
    )
    model.fit(train.user_ids, train.item_ids, train.ratings, epochs=15, verbose=False)
    item_embs = model.get_all_item_embeddings()
    user_embs = model.user_embeddings.weight.data.cpu().numpy()

    # Cluster manager
    cm = UserClusterManager(n_clusters=50, embedding_dim=64)
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # ---- Build all policies ----
    pop = PopularityBaseline(item_pop, n_items, N_RECS)
    faiss_bl = FAISSBaseline(item_embs, user_embs, N_RECS)

    spec_config = SpeculativeConfig(
        top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS, use_pool_retrieval=False
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
        config=spec_config, item_embeddings=item_embs, user_history=user_history,
    )
    spec.warm_cache(list(range(n_users)))

    pool_config = SpeculativeConfig(
        top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS,
        use_pool_retrieval=True, pool_size=200,
    )
    spec_pool = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
        config=pool_config, item_embeddings=item_embs, user_history=user_history,
    )
    spec_pool.warm_cache(list(range(n_users)))

    naive_config = SpeculativeConfig(
        top_k_clusters=1, acceptance_threshold=0.0, n_recs=N_RECS, use_pool_retrieval=False,
    )
    naive = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.0),
        config=naive_config, item_embeddings=item_embs, user_history=user_history,
    )
    naive.warm_cache(list(range(n_users)))

    # ---- Run all policies ----
    policies = {
        "Fresh": lambda uid, exc: list(model.recommend(uid, n=N_RECS, exclude_items=exc)),
        "Popularity": lambda uid, exc: pop.recommend(uid, exc),
        "FAISS": lambda uid, exc: faiss_bl.recommend(uid, exc),
        "Naive": lambda uid, exc: naive.recommend(uid).items,
        "Spec K=3": lambda uid, exc: spec.recommend(uid).items,
        "Spec+Pool": lambda uid, exc: spec_pool.recommend(uid).items,
    }

    results = {}
    for name, fn in policies.items():
        print(f"  Running {name}...")
        results[name] = run_policy(name, fn, stream, gt, user_history, item_pop, n_ints)
        o = results[name]["overall"]
        print(f"    NDCG={o['ndcg']:.4f} CTR={o['ctr']:.4f} Nov={o['novelty']:.1f} "
              f"Lat={o['latency_ms']:.3f}ms")

    # Summary table
    print(f"\n  {'Policy':<14} {'NDCG':>8} {'Recall':>8} {'HR':>8} {'CTR':>8} "
          f"{'Novel':>8} {'Lat(ms)':>8}")
    print("  " + "-" * 70)
    for name, r in results.items():
        o = r["overall"]
        print(f"  {name:<14} {o['ndcg']:>8.4f} {o['recall']:>8.4f} {o['hr']:>8.3f} "
              f"{o['ctr']:>8.4f} {o['novelty']:>8.1f} {o['latency_ms']:>8.3f}")

    # Save
    out = Path("results/supp_online_serving_replay.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")

    # Figure: rolling CTR + NDCG + novelty across policies
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"Fresh": "#2c3e50", "Popularity": "#95a5a6", "FAISS": "#e67e22",
              "Naive": "#c0392b", "Spec K=3": "#2980b9", "Spec+Pool": "#27ae60"}

    for ax, (metric, label) in zip(axes, [("ctr","CTR Proxy"),("ndcg","NDCG@10"),("novelty","Novelty")]):
        for name, r in results.items():
            ws = r["windows"]
            xs = [(w["start"]+w["end"])/2 for w in ws]
            ys = [w[metric] for w in ws]
            ax.plot(xs, ys, "-", color=colors.get(name,"gray"), label=name, alpha=0.8)
        ax.set_xlabel("Request #")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.2)

    axes[-1].legend(fontsize=7, loc="best")
    plt.tight_layout()
    fig_path = Path("paper/figures/supp_online_serving.pdf")
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
