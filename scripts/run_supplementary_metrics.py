#!/usr/bin/env python3
"""
A1 + A2 — Multi-metric and Popularity-bias Supplementary Experiments.

Re-runs S4 baseline pipeline (Fresh / Naive cache / Spec K=3 / Spec+Pool) on
ML-1M and Amazon Electronics, computing:
  - A1: Recall@10, HR@10, MRR, MAP, NDCG@10, ILD@10
  - A2: Coverage, Gini, Entropy of recommendation frequency
        Per-bucket NDCG (head 20% / torso 60% / tail 20% by training frequency)

Outputs: results/supp_a1_a2_multimetric.json
         paper/figures/supp_popularity_bias.pdf

Usage:
    conda activate reccache
    python scripts/run_supplementary_metrics.py
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
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
from reccache.evaluation.metrics import (
    RecommendationMetrics, compute_coverage, compute_ild
)


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

    # Item training-frequency for popularity-bias buckets
    item_freq = Counter(int(i) for i in train.item_ids)
    return train, dict(user_history), dict(test_lookup), item_freq


def get_test_users(test_lookup, n, seed):
    rng = np.random.default_rng(seed)
    uids = [u for u, items in test_lookup.items() if len(items) > 0]
    if len(uids) > n:
        uids = rng.choice(uids, size=n, replace=False).tolist()
    return uids


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (0=equal, 1=concentrated)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v) - (n + 1) * v.sum()) / (n * v.sum()))


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def assign_buckets(item_freq: Counter, n_items: int):
    """Return dict item_id -> 'head'/'torso'/'tail' by frequency rank.

    head: top 20% most-popular items (by frequency)
    torso: middle 60%
    tail: bottom 20%
    """
    items_sorted = sorted(range(n_items), key=lambda i: -item_freq.get(i, 0))
    head_cut = int(0.2 * n_items)
    torso_cut = int(0.8 * n_items)
    bucket = {}
    for rank, iid in enumerate(items_sorted):
        if rank < head_cut:
            bucket[iid] = "head"
        elif rank < torso_cut:
            bucket[iid] = "torso"
        else:
            bucket[iid] = "tail"
    return bucket


def aggregate_full(results, test_lookup, item_embs, item_buckets, n_items):
    """Compute full set of metrics including popularity bucket distribution."""
    ndcgs, recalls, hrs, mrrs, maps_, ilds = [], [], [], [], [], []
    rec_freq = Counter()       # global item-recommendation frequency
    bucket_share = Counter()   # head/torso/tail share over all positions

    for r in results:
        gt = test_lookup.get(r.user_id, set())
        if not gt:
            continue
        m = RecommendationMetrics.compute_all(r.items, gt, k=N_RECS)
        ndcgs.append(m.ndcg_at_k)
        recalls.append(m.recall_at_k)
        hrs.append(m.hit_rate)
        mrrs.append(m.mrr)
        maps_.append(m.map_score)
        ilds.append(compute_ild(r.items, item_embs))
        for it in r.items[:N_RECS]:
            rec_freq[it] += 1
            bucket_share[item_buckets.get(int(it), "tail")] += 1

    coverage = len(rec_freq) / n_items if n_items > 0 else 0.0

    # Recommendation-frequency Gini and entropy over the *catalog*
    freq_vec = np.zeros(n_items, dtype=np.float64)
    for it, c in rec_freq.items():
        if 0 <= it < n_items:
            freq_vec[it] = c
    g = gini(freq_vec)
    p = freq_vec / freq_vec.sum() if freq_vec.sum() > 0 else freq_vec
    h = entropy(p)
    h_norm = h / np.log(n_items) if n_items > 1 else 0.0  # normalised 0..1

    total_pos = sum(bucket_share.values()) or 1
    bucket_pct = {b: bucket_share.get(b, 0) / total_pos for b in ["head", "torso", "tail"]}

    return {
        "ndcg":     float(np.mean(ndcgs))   if ndcgs   else 0.0,
        "recall":   float(np.mean(recalls)) if recalls else 0.0,
        "hr":       float(np.mean(hrs))     if hrs     else 0.0,
        "mrr":      float(np.mean(mrrs))    if mrrs    else 0.0,
        "map":      float(np.mean(maps_))   if maps_   else 0.0,
        "ild":      float(np.mean(ilds))    if ilds    else 0.0,
        "coverage": coverage,
        "gini":     g,
        "entropy_norm": h_norm,
        "head_share":  bucket_pct["head"],
        "torso_share": bucket_pct["torso"],
        "tail_share":  bucket_pct["tail"],
        "n": len(ndcgs),
    }


def run_fresh(model, test_users, user_history):
    out = []
    for uid in test_users:
        exclude = user_history.get(uid)
        t0 = time.time()
        items = list(model.recommend(uid, n=N_RECS, exclude_items=exclude))
        out.append(SpeculativeResult(
            user_id=uid, items=items, accepted=False, acceptance_prob=0.0,
            accepted_cluster_id=None, accepted_cluster_rank=-1,
            latency_ms=(time.time()-t0)*1000, phase="fresh",
        ))
    return out


def run_spec(model, cm, item_embs, test_users, user_history, train, use_pool=False):
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
    return [spec.recommend(uid) for uid in test_users]


def run_naive(model, cm, item_embs, test_users, user_history, train):
    """Naive cache: always serve nearest cluster's cached recs (no acceptance test)."""
    config = SpeculativeConfig(
        top_k_clusters=1, acceptance_threshold=0.0,  # accept everything
        n_recs=N_RECS, use_pool_retrieval=False,
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.0),
        config=config, item_embeddings=item_embs, user_history=user_history,
    )
    spec.warm_cache(list(range(train.n_users)))
    return [spec.recommend(uid) for uid in test_users]


def run_one_seed(dataset_name, cfg, seed):
    print(f"  seed={seed}")
    set_seed(seed)
    train, user_history, test_lookup, item_freq = load(dataset_name, cfg)
    n_items = train.n_items
    item_buckets = assign_buckets(item_freq, n_items)
    test_users = get_test_users(test_lookup, cfg["n_test_users"], seed)

    print(f"    Training MF (n_users={train.n_users}, n_items={n_items})")
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=n_items, embedding_dim=cfg["embedding_dim"]
    )
    model.fit(train.user_ids, train.item_ids, train.ratings,
              epochs=cfg["epochs"], verbose=False)
    item_embs = model.get_all_item_embeddings()

    cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=cfg["embedding_dim"])
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    print("    Fresh...")
    fresh = run_fresh(model, test_users, user_history)
    print("    Naive cache (top-1, no accept test)...")
    naive = run_naive(model, cm, item_embs, test_users, user_history, train)
    print("    Spec K=3...")
    spec = run_spec(model, cm, item_embs, test_users, user_history, train, use_pool=False)
    print("    Spec+Pool...")
    pool = run_spec(model, cm, item_embs, test_users, user_history, train, use_pool=True)

    return {
        "fresh":     aggregate_full(fresh, test_lookup, item_embs, item_buckets, n_items),
        "naive":     aggregate_full(naive, test_lookup, item_embs, item_buckets, n_items),
        "spec_k3":   aggregate_full(spec,  test_lookup, item_embs, item_buckets, n_items),
        "spec_pool": aggregate_full(pool,  test_lookup, item_embs, item_buckets, n_items),
    }


def avg_seeds(seed_results):
    methods = list(seed_results[0].keys())
    keys = list(seed_results[0][methods[0]].keys())
    avg = {}
    for m in methods:
        avg[m] = {}
        for k in keys:
            avg[m][k] = float(np.mean([r[m][k] for r in seed_results]))
    return avg


def print_summary(ds, results):
    fn = results["fresh"]["ndcg"]
    print(f"\n  ===== {ds} (avg {len(SEEDS)} seeds) =====")
    cols = ["NDCG", "Recall", "HR", "MRR", "MAP", "ILD", "Cov", "Gini", "Tail%"]
    print("  {:<12}".format("Method") + " ".join(f"{c:>7}" for c in cols))
    for m in ["fresh", "naive", "spec_k3", "spec_pool"]:
        r = results[m]
        row = [r["ndcg"], r["recall"], r["hr"], r["mrr"], r["map"],
               r["ild"], r["coverage"], r["gini"], r["tail_share"]]
        print("  {:<12}".format(m) + " ".join(f"{v:>7.4f}" for v in row))


def make_figure(all_results, fig_path):
    """Bar chart: head/torso/tail recommendation share for each method × dataset."""
    datasets = list(all_results.keys())
    methods = ["fresh", "naive", "spec_k3", "spec_pool"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        x = np.arange(len(methods))
        head  = [all_results[ds][m]["head_share"]  for m in methods]
        torso = [all_results[ds][m]["torso_share"] for m in methods]
        tail  = [all_results[ds][m]["tail_share"]  for m in methods]
        ax.bar(x, head, label="head 20%", color="#4C72B0")
        ax.bar(x, torso, bottom=head, label="torso 60%", color="#DD8452")
        ax.bar(x, tail, bottom=np.array(head)+np.array(torso),
               label="tail 20%", color="#55A868")
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(f"{ds}: rec composition by item popularity")
        ax.set_ylabel("Fraction of recommendations")
    axes[0][-1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(exist_ok=True, parents=True)

    all_results = {}
    for ds in args.dataset:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n{'='*60}\nA1+A2 supplementary | {ds}\n{'='*60}")
        seed_results = [run_one_seed(ds, cfg, s) for s in args.seeds]
        all_results[ds] = avg_seeds(seed_results)
        print_summary(ds, all_results[ds])

    out = results_dir / "supp_a1_a2_multimetric.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    fig_path = figures_dir / "supp_popularity_bias.pdf"
    make_figure(all_results, fig_path)
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
