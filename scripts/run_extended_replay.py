#!/usr/bin/env python3
"""
B2 — Extended Log Replay Simulation.

Extends S6 to 3 random orderings of all ML-1M test users with GT (1188 users).
Each ordering is a full "replay" of the recommendation serving pipeline.
Reports rolling 100-request windows of NDCG, Recall@10, HR@10, Accept, MCG.

Key claim: metrics are stable regardless of request ordering.

Outputs: results/supp_b2_extended_replay.json
         paper/figures/supp_extended_replay.pdf

Usage:
    conda activate reccache
    python scripts/run_extended_replay.py
"""

import sys
import json
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
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import RecommendationMetrics

N_RECS = 10
WINDOW = 100
SEEDS = [42, 123, 456]


def run_replay(spec, users, gt, window=WINDOW):
    """Run a single replay pass and return per-window rolling metrics."""
    ndcgs, recalls, hrs, accepts, mcgs = [], [], [], [], []
    for uid in users:
        sr = spec.recommend(uid)
        relevant = gt.get(uid, set())
        if relevant:
            ndcgs.append(RecommendationMetrics.ndcg_at_k(sr.items, relevant, N_RECS))
            recalls.append(RecommendationMetrics.recall_at_k(sr.items, relevant, N_RECS))
            hrs.append(RecommendationMetrics.hit_rate(sr.items, relevant, N_RECS))
        else:
            ndcgs.append(np.nan)
            recalls.append(np.nan)
            hrs.append(np.nan)
        accepts.append(int(sr.accepted))
        mcgs.append(1 if (sr.accepted and sr.accepted_cluster_rank > 0) else 0)

    # Rolling windows
    windows = []
    for start in range(0, len(users), window):
        end = min(start + window, len(users))
        if end - start < window // 2:
            break
        s = slice(start, end)
        valid_ndcg   = [v for v in ndcgs[s]   if not np.isnan(v)]
        valid_recall = [v for v in recalls[s]  if not np.isnan(v)]
        valid_hr     = [v for v in hrs[s]      if not np.isnan(v)]
        windows.append({
            "start": start, "end": end,
            "ndcg":        float(np.mean(valid_ndcg))   if valid_ndcg   else 0.0,
            "recall":      float(np.mean(valid_recall)) if valid_recall else 0.0,
            "hr":          float(np.mean(valid_hr))     if valid_hr     else 0.0,
            "accept_rate": float(np.mean(accepts[s])),
            "mcg":         float(np.mean(mcgs[s])),
        })

    valid_all = [v for v in ndcgs if not np.isnan(v)]
    valid_rec = [v for v in recalls if not np.isnan(v)]
    valid_hrs = [v for v in hrs if not np.isnan(v)]
    overall = {
        "ndcg":        float(np.mean(valid_all)) if valid_all else 0.0,
        "recall":      float(np.mean(valid_rec)) if valid_rec else 0.0,
        "hr":          float(np.mean(valid_hrs)) if valid_hrs else 0.0,
        "accept_rate": float(np.mean(accepts)),
        "mcg":         float(np.mean(mcgs)),
        "n": len(users),
    }
    return {"overall": overall, "windows": windows}


def main():
    print("B2: Extended log replay simulation | ML-1M")
    np.random.seed(42)

    loader = DataLoader("data")
    train, val, test = loader.load_dataset("ml-1m")
    gt = defaultdict(set)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= 4.0:
            gt[int(uid)].add(int(iid))
    gt = dict(gt)

    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))

    # Use same test user set as S6: users with ground truth, in test order
    seen = set()
    all_test_users = []
    for uid in test.user_ids:
        u = int(uid)
        if u not in seen and u in gt:
            all_test_users.append(u)
            seen.add(u)
    print(f"  {len(all_test_users)} test users with GT")

    # Train MF
    import torch; torch.manual_seed(42)
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=64
    )
    model.fit(train.user_ids, train.item_ids, train.ratings, epochs=15, verbose=False)
    item_embs = model.get_all_item_embeddings()

    cm = UserClusterManager(n_clusters=50, embedding_dim=64)
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    config = SpeculativeConfig(top_k_clusters=3, acceptance_threshold=0.35, n_recs=N_RECS)
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
        config=config, item_embeddings=item_embs, user_history=dict(user_history),
    )
    spec.warm_cache(list(range(train.n_users)))

    # Run 3 replays with different orderings
    all_runs = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(all_test_users).tolist()
        print(f"  Replay seed={seed}, {len(order)} requests...")
        result = run_replay(spec, order, gt)
        all_runs[f"seed_{seed}"] = result
        o = result["overall"]
        print(f"    NDCG={o['ndcg']:.4f}, Recall={o['recall']:.4f}, HR={o['hr']:.3f}, "
              f"Accept={o['accept_rate']:.1%}, MCG={o['mcg']:.1%}")

    # Cross-seed stability: std of overall metrics across orderings
    metrics = ["ndcg", "recall", "hr", "accept_rate", "mcg"]
    stability = {}
    for m in metrics:
        vals = [all_runs[k]["overall"][m] for k in all_runs]
        stability[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    print(f"\n  Cross-seed stability (3 orderings):")
    for m in metrics:
        print(f"    {m:>12}: {stability[m]['mean']:.4f} ± {stability[m]['std']:.4f}")

    # Save
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
    out = results_dir / "supp_b2_extended_replay.json"
    with open(out, "w") as f:
        json.dump({"runs": all_runs, "stability": stability}, f, indent=2)
    print(f"\nResults saved: {out}")

    # Figure: rolling metrics across all 3 orderings (band plot)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    metric_labels = [("ndcg", "NDCG@10"), ("recall", "Recall@10"),
                     ("accept_rate", "Accept Rate"), ("mcg", "MCG")]

    for ax, (metric, label) in zip(axes, metric_labels):
        for seed_key, color in zip(all_runs, ["#4C72B0", "#DD8452", "#55A868"]):
            ws = all_runs[seed_key]["windows"]
            xs = [(w["start"] + w["end"]) / 2 for w in ws]
            ys = [w[metric] for w in ws]
            ax.plot(xs, ys, "-", alpha=0.6, color=color,
                    label=seed_key.replace("seed_", "order "))
        ax.axhline(stability[metric]["mean"], ls="--", color="gray", alpha=0.5)
        ax.set_xlabel("Request #")
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n(μ={stability[metric]['mean']:.3f} ± {stability[metric]['std']:.4f})")
        if metric in ("accept_rate", "mcg"):
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

    axes[-1].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    fig_path = Path("paper/figures/supp_extended_replay.pdf")
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
