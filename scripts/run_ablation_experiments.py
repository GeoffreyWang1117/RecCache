#!/usr/bin/env python3
"""
Supplemental ablation experiments for RecSys 2026.

  A1 — Temperature sensitivity (score-ratio criterion)
  A2 — Cold-start vs active user breakdown
  A3 — Latency breakdown (per-phase timing)
  A4 — Cluster count sensitivity
"""

import sys
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
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import (
    RecommendationMetrics, SpeculativeMetrics, compute_ild, compute_coverage,
)

N_RUNS = 3
N_TEST = 500


def set_seed(s):
    np.random.seed(s)


def build_gt(test, min_r=4.0):
    gt = defaultdict(set)
    for u, i, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= min_r:
            gt[int(u)].add(int(i))
    return dict(gt)


def build_int_counts(train):
    c = defaultdict(int)
    for u in train.user_ids:
        c[int(u)] += 1
    return dict(c)


_cache = {}


def load_all(name, cfg):
    if name in _cache:
        return _cache[name]
    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        name, min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"], max_samples=cfg["max_samples"],
    )
    gt = build_gt(test, cfg["min_rating_gt"])
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items, embedding_dim=64,
    )
    model.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"], verbose=True)
    item_embs = model.get_all_item_embeddings()
    cm = UserClusterManager(
        n_clusters=min(cfg["n_clusters"], train.n_users // 2),
        embedding_dim=item_embs.shape[1], n_items=len(item_embs),
    )
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
    int_counts = build_int_counts(train)
    _cache[name] = (train, test, gt, model, item_embs, cm, int_counts)
    return _cache[name]


CONFIGS = {
    "ml-1m": {"max_samples": None, "min_user": 5, "min_item": 5,
              "min_rating_gt": 4.0, "n_clusters": 50, "epochs": 15},
    "amazon-electronics": {"max_samples": 1000000, "min_user": 3, "min_item": 3,
                           "min_rating_gt": 4.0, "n_clusters": 50, "epochs": 15},
}


def evaluate(model, cm, criterion, gt, users, item_embs, int_counts,
             k=3, thr=0.35, n_recs=10, warm_users=None, use_pool=False, pool_size=200):
    config = SpeculativeConfig(
        top_k_clusters=k, acceptance_threshold=thr, n_recs=n_recs,
        use_pool_retrieval=use_pool, pool_size=pool_size,
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=criterion, config=config, item_embeddings=item_embs,
    )
    if warm_users:
        spec.warm_cache(warm_users)

    results, recs, ndcgs = [], {}, {}
    for uid in users:
        if uid not in gt:
            continue
        sr = spec.recommend(uid)
        results.append(sr)
        recs[uid] = sr.items
        ndcgs[uid] = RecommendationMetrics.ndcg_at_k(sr.items, gt[uid], n_recs)

    if not results:
        return {"ndcg": 0, "accept_rate": 0, "coverage": 0, "mcg": 0, "speedup": 1}

    return {
        "ndcg": float(np.mean(list(ndcgs.values()))),
        "accept_rate": SpeculativeMetrics.acceptance_rate(results),
        "coverage": compute_coverage(recs, item_embs.shape[0]),
        "mcg": SpeculativeMetrics.multi_cluster_gain(results),
        "speedup": SpeculativeMetrics.speedup_estimate(results),
        "user_ndcgs": ndcgs,
        "results": results,
    }


def multi_seed(fn):
    all_m = defaultdict(list)
    for ri in range(N_RUNS):
        set_seed(42 + ri)
        m = fn(ri)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                all_m[k].append(v)
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_m.items()}


# ===========================================================================
# A1: Temperature sensitivity
# ===========================================================================
def run_a1():
    print(f"\n{'='*60}\nA1: Temperature Sensitivity\n{'='*60}")
    temps = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    results = {}
    for ds in ["ml-1m", "amazon-electronics"]:
        print(f"\n--- {ds} ---")
        train, test, gt, model, embs, cm, ic = load_all(ds, CONFIGS[ds])
        users = list(gt.keys())[:N_TEST]
        warm = list(gt.keys())
        ds_r = {}
        for t in temps:
            print(f"  T={t}", flush=True)
            def fn(ri, _t=t):
                np.random.shuffle(users)
                crit = ScoreRatioAcceptanceCriterion(threshold=0.35, temperature=_t)
                return evaluate(model, cm, crit, gt, users, embs, ic, warm_users=warm)
            ds_r[f"T={t}"] = multi_seed(fn)
            r = ds_r[f"T={t}"]
            print(f"    NDCG={r['ndcg']['mean']:.4f} Acc={r['accept_rate']['mean']:.3f} MCG={r['mcg']['mean']:.3f}")
        results[ds] = ds_r
    return results


# ===========================================================================
# A2: Cold-start vs active user analysis
# ===========================================================================
def run_a2():
    print(f"\n{'='*60}\nA2: Cold-start vs Active User Analysis\n{'='*60}")
    results = {}
    for ds in ["ml-1m", "amazon-electronics"]:
        print(f"\n--- {ds} ---")
        train, test, gt, model, embs, cm, ic = load_all(ds, CONFIGS[ds])
        users = list(gt.keys())[:N_TEST]
        warm = list(gt.keys())

        # Split users by interaction count
        cold = [u for u in users if ic.get(u, 0) < 10]
        medium = [u for u in users if 10 <= ic.get(u, 0) < 50]
        active = [u for u in users if ic.get(u, 0) >= 50]

        print(f"  Cold(<10): {len(cold)}, Medium(10-50): {len(medium)}, Active(>=50): {len(active)}")

        ds_r = {}
        for label, group in [("cold", cold), ("medium", medium), ("active", active)]:
            if len(group) < 5:
                print(f"  {label}: too few users, skip")
                continue
            print(f"  {label} ({len(group)} users)", flush=True)

            for mode_label, use_pool in [("static", False), ("pool", True)]:
                def fn(ri, _g=group, _p=use_pool):
                    crit = ScoreRatioAcceptanceCriterion(threshold=0.35)
                    return evaluate(model, cm, crit, gt, _g, embs, ic, warm_users=warm,
                                    use_pool=_p, pool_size=200)
                key = f"{label}_{mode_label}"
                ds_r[key] = multi_seed(fn)
                r = ds_r[key]
                print(f"    {key}: NDCG={r['ndcg']['mean']:.4f} Acc={r['accept_rate']['mean']:.3f} Cov={r['coverage']['mean']:.4f}")

        results[ds] = ds_r
    return results


# ===========================================================================
# A3: Latency breakdown
# ===========================================================================
def run_a3():
    print(f"\n{'='*60}\nA3: Latency Breakdown\n{'='*60}")
    results = {}
    for ds in ["ml-1m"]:
        print(f"\n--- {ds} ---")
        train, test, gt, model, embs, cm, ic = load_all(ds, CONFIGS[ds])
        users = list(gt.keys())[:200]
        warm = list(gt.keys())

        for mode_label, use_pool in [("static", False), ("pool", True)]:
            config = SpeculativeConfig(
                top_k_clusters=3, acceptance_threshold=0.35, n_recs=10,
                use_pool_retrieval=use_pool, pool_size=200,
            )
            crit = ScoreRatioAcceptanceCriterion(threshold=0.35)
            spec = SpeculativeRecommender(
                recommender=model, cluster_manager=cm,
                acceptance_criterion=crit, config=config, item_embeddings=embs,
            )
            spec.warm_cache(warm)

            # Measure per-phase timing
            cluster_times, verify_times, total_times = [], [], []
            for uid in users:
                if uid not in gt:
                    continue
                # Phase 1: cluster lookup
                t0 = time.perf_counter()
                candidates = cm.get_nearest_clusters(uid, top_k=3)
                user_emb = cm.get_user_embedding(uid)
                t1 = time.perf_counter()
                cluster_times.append((t1 - t0) * 1000)

                # Phase 2+3: full pipeline timing
                t2 = time.perf_counter()
                sr = spec.recommend(uid)
                t3 = time.perf_counter()
                total_times.append((t3 - t2) * 1000)
                verify_times.append((t3 - t2) * 1000 - (t1 - t0) * 1000)

            print(f"  {mode_label}:")
            print(f"    Cluster lookup: {np.mean(cluster_times):.3f}ms ± {np.std(cluster_times):.3f}")
            print(f"    Verify+serve:   {np.mean(verify_times):.3f}ms ± {np.std(verify_times):.3f}")
            print(f"    Total:          {np.mean(total_times):.3f}ms ± {np.std(total_times):.3f}")
            results[f"{ds}_{mode_label}"] = {
                "cluster_ms": {"mean": float(np.mean(cluster_times)), "std": float(np.std(cluster_times))},
                "verify_ms": {"mean": float(np.mean(verify_times)), "std": float(np.std(verify_times))},
                "total_ms": {"mean": float(np.mean(total_times)), "std": float(np.std(total_times))},
            }
    return results


# ===========================================================================
# A4: Cluster count sensitivity
# ===========================================================================
def run_a4():
    print(f"\n{'='*60}\nA4: Cluster Count Sensitivity\n{'='*60}")
    n_clusters_list = [10, 25, 50, 100, 200]
    results = {}
    for ds in ["ml-1m", "amazon-electronics"]:
        print(f"\n--- {ds} ---")
        cfg = CONFIGS[ds]
        loader = DataLoader("data")
        train, val, test = loader.load_dataset(
            ds, min_user_interactions=cfg["min_user"],
            min_item_interactions=cfg["min_item"], max_samples=cfg["max_samples"],
        )
        gt = build_gt(test, cfg["min_rating_gt"])
        ic = build_int_counts(train)

        # Reuse model from cache if available
        if ds in _cache:
            _, _, _, model, embs, _, _ = _cache[ds]
        else:
            model = MatrixFactorizationRecommender(n_users=train.n_users, n_items=train.n_items, embedding_dim=64)
            model.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"], verbose=True)
            embs = model.get_all_item_embeddings()

        users = list(gt.keys())[:N_TEST]
        warm = list(gt.keys())
        ds_r = {}

        for nc in n_clusters_list:
            if nc > train.n_users // 2:
                continue
            print(f"  n_clusters={nc}", flush=True)

            def fn(ri, _nc=nc):
                np.random.shuffle(users)
                _cm = UserClusterManager(n_clusters=_nc, embedding_dim=embs.shape[1], n_items=len(embs))
                _cm.set_item_embeddings(embs)
                _cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)
                crit = ScoreRatioAcceptanceCriterion(threshold=0.35)
                # Build speculative recommender manually
                config = SpeculativeConfig(top_k_clusters=3, acceptance_threshold=0.35, n_recs=10)
                spec = SpeculativeRecommender(
                    recommender=model, cluster_manager=_cm,
                    acceptance_criterion=crit, config=config, item_embeddings=embs,
                )
                spec.warm_cache(warm)
                res, recs, ndcgs = [], {}, {}
                for uid in users:
                    if uid not in gt: continue
                    sr = spec.recommend(uid)
                    res.append(sr)
                    recs[uid] = sr.items
                    ndcgs[uid] = RecommendationMetrics.ndcg_at_k(sr.items, gt[uid], 10)
                if not res:
                    return {"ndcg": 0, "accept_rate": 0, "coverage": 0, "mcg": 0}
                return {
                    "ndcg": float(np.mean(list(ndcgs.values()))),
                    "accept_rate": SpeculativeMetrics.acceptance_rate(res),
                    "coverage": compute_coverage(recs, embs.shape[0]),
                    "mcg": SpeculativeMetrics.multi_cluster_gain(res),
                }

            ds_r[f"K={nc}"] = multi_seed(fn)
            r = ds_r[f"K={nc}"]
            print(f"    NDCG={r['ndcg']['mean']:.4f} Acc={r['accept_rate']['mean']:.3f} MCG={r['mcg']['mean']:.3f} Cov={r['coverage']['mean']:.4f}")

        results[ds] = ds_r
    return results


# ===========================================================================
# Figure generation
# ===========================================================================
def gen_figures(all_results, fig_dir="paper/figures"):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # A1: Temperature plot
    if "A1" in all_results:
        for ds, ds_data in all_results["A1"].items():
            temps, ndcgs, accs, mcgs = [], [], [], []
            for key in sorted(ds_data.keys(), key=lambda x: float(x.split("=")[1])):
                temps.append(float(key.split("=")[1]))
                ndcgs.append(ds_data[key]["ndcg"]["mean"])
                accs.append(ds_data[key]["accept_rate"]["mean"])
                mcgs.append(ds_data[key]["mcg"]["mean"])

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(temps, accs, "o-", color="#1f77b4", linewidth=2, label="Accept Rate")
            ax1.plot(temps, mcgs, "s--", color="#2ca02c", linewidth=2, label="MCG")
            ax1.set_xlabel("Temperature T", fontsize=11)
            ax1.set_ylabel("Rate", fontsize=11)
            ax1.set_xscale("log")

            ax2 = ax1.twinx()
            ax2.plot(temps, ndcgs, "D:", color="#d62728", linewidth=2, label="NDCG")
            ax2.set_ylabel("NDCG@10", fontsize=11, color="#d62728")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
            ax1.set_title(f"Temperature Sensitivity ({ds})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / f"a1_temperature_{ds}.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"a1_temperature_{ds}.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: a1_temperature_{ds}.pdf")

    # A2: Cold vs Active bar chart
    if "A2" in all_results:
        for ds, ds_data in all_results["A2"].items():
            groups = ["cold", "medium", "active"]
            modes = ["static", "pool"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            x = np.arange(len(groups))
            w = 0.35
            for i, mode in enumerate(modes):
                ndcgs = [ds_data.get(f"{g}_{mode}", {}).get("ndcg", {}).get("mean", 0) for g in groups]
                covs = [ds_data.get(f"{g}_{mode}", {}).get("coverage", {}).get("mean", 0) for g in groups]
                color = "#1f77b4" if mode == "static" else "#2ca02c"
                axes[0].bar(x + i * w - w/2, ndcgs, w, label=mode.capitalize(), color=color, alpha=0.85, edgecolor="black", linewidth=0.3)
                axes[1].bar(x + i * w - w/2, covs, w, label=mode.capitalize(), color=color, alpha=0.85, edgecolor="black", linewidth=0.3)

            for ax, ylabel, title in [(axes[0], "NDCG@10", "Quality"), (axes[1], "Coverage", "Coverage")]:
                ax.set_xticks(x)
                ax.set_xticklabels(["Cold\n(<10)", "Medium\n(10-50)", "Active\n(>=50)"])
                ax.set_ylabel(ylabel)
                ax.set_title(title, fontweight="bold")
                ax.legend(fontsize=8)

            plt.suptitle(f"User Activity Breakdown ({ds})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / f"a2_user_groups_{ds}.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"a2_user_groups_{ds}.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: a2_user_groups_{ds}.pdf")

    # A4: Cluster count
    if "A4" in all_results:
        for ds, ds_data in all_results["A4"].items():
            ks, ndcgs, accs, covs = [], [], [], []
            for key in sorted(ds_data.keys(), key=lambda x: int(x.split("=")[1])):
                ks.append(int(key.split("=")[1]))
                ndcgs.append(ds_data[key]["ndcg"]["mean"])
                accs.append(ds_data[key]["accept_rate"]["mean"])
                covs.append(ds_data[key]["coverage"]["mean"])

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(ks, accs, "o-", color="#1f77b4", linewidth=2, label="Accept Rate")
            ax1.plot(ks, covs, "s--", color="#2ca02c", linewidth=2, label="Coverage")
            ax1.set_xlabel("Number of Clusters", fontsize=11)
            ax1.set_ylabel("Rate", fontsize=11)

            ax2 = ax1.twinx()
            ax2.plot(ks, ndcgs, "D:", color="#d62728", linewidth=2, label="NDCG")
            ax2.set_ylabel("NDCG@10", fontsize=11, color="#d62728")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
            ax1.set_title(f"Cluster Count Sensitivity ({ds})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / f"a4_clusters_{ds}.pdf", bbox_inches="tight", dpi=300)
            plt.savefig(fig_dir / f"a4_clusters_{ds}.png", bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  Saved: a4_clusters_{ds}.pdf")


# ===========================================================================
def main():
    print("=" * 60)
    print("RecCache — Supplemental Ablation Experiments")
    print("=" * 60, flush=True)

    all_results = {}
    t0 = time.time()

    for name, fn in [("A1", run_a1), ("A2", run_a2), ("A3", run_a3), ("A4", run_a4)]:
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    print("\nGenerating figures...")
    gen_figures(all_results)

    # Save
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    # Remove non-serializable fields
    clean = {}
    for group, gdata in all_results.items():
        clean[group] = {}
        for ds, ds_data in gdata.items():
            if isinstance(ds_data, dict):
                clean[group][ds] = {}
                for method, mdata in ds_data.items():
                    if isinstance(mdata, dict):
                        clean[group][ds][method] = {
                            k: v for k, v in mdata.items()
                            if k not in ("user_ndcgs", "results")
                        }
                    else:
                        clean[group][ds][method] = mdata
            else:
                clean[group][ds] = ds_data

    out = Path("results/ablation_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(convert(clean), f, indent=2)
    print(f"Results saved to {out}")
    print("Ablation experiments complete!")


if __name__ == "__main__":
    main()
