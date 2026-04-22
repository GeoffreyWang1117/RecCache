#!/usr/bin/env python3
"""
S15 — GPU Latency Benchmark.

Measures per-request inference latency for Fresh vs Speculative serving
across catalog sizes on CPU and GPU. Demonstrates that RecCache cluster-cache
acceptance is nearly device-agnostic (constant small lookup) while fresh
inference scales linearly with catalog size on both CPU/GPU.

Key question: does the speedup ratio hold (or improve) on GPU?

Usage:
    conda activate reccache
    python scripts/run_gpu_latency.py
    python scripts/run_gpu_latency.py --device cuda
    python scripts/run_gpu_latency.py --device cpu --n-items 1000 10000 100000
"""

import sys
import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig


N_WARMUP = 20
N_TRIALS = 200
TOP_K = 3
THRESHOLD = 0.35
N_RECS = 10


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class GPUScorer:
    """Wraps MF embeddings as plain tensors for GPU-native scoring.

    Avoids the nn.Parameter.to(cuda) non-leaf issue by holding the
    trained weights as regular torch.Tensor on the target device.
    """

    def __init__(self, mf: MatrixFactorizationRecommender, device: str = "cpu",
                 max_catalog: int = None):
        self.device = device
        self.n_items = mf.n_items
        self.embedding_dim = mf.embedding_dim

        with torch.no_grad():
            self.user_emb = mf.user_embeddings.weight.data.clone().to(device)  # (n_users, d)
            self.user_bias = mf.user_bias.weight.data.clone().to(device)       # (n_users, 1)
            self.global_bias = mf.global_bias.data.clone().to(device)         # (1,)

            real_item_emb = mf.item_embeddings.weight.data.clone()  # (n_items, d)
            real_item_bias = mf.item_bias.weight.data.clone()       # (n_items, 1)

            # If max_catalog > actual catalog, tile real embeddings to desired size.
            # This ensures matmul benchmarks reflect actual computation at larger scales.
            if max_catalog and max_catalog > self.n_items:
                repeats = (max_catalog + self.n_items - 1) // self.n_items
                real_item_emb = real_item_emb.repeat(repeats, 1)[:max_catalog]
                real_item_bias = real_item_bias.repeat(repeats, 1)[:max_catalog]

            self.item_emb = real_item_emb.to(device)    # (N, d)
            self.item_bias = real_item_bias.to(device)  # (N, 1)
            self._full_catalog = len(self.item_emb)

    def score_all_items(self, user_id: int, n_items_override: int = None) -> np.ndarray:
        """Score all items for a user. n_items_override limits catalog for benchmark."""
        n = min(n_items_override or self.n_items, self._full_catalog)
        with torch.no_grad():
            u_emb = self.user_emb[user_id]                       # (d,)
            i_emb = self.item_emb[:n]                            # (n, d)
            u_b   = self.user_bias[user_id]                      # (1,)
            i_b   = self.item_bias[:n].squeeze(1)                # (n,)
            scores = i_emb @ u_emb + u_b.squeeze() + i_b + self.global_bias.squeeze()
            if self.device != "cpu":
                torch.cuda.synchronize()
            return scores.cpu().numpy()

    def recommend(self, user_id: int, n: int = 10, n_items_override: int = None,
                  exclude: list = None) -> list:
        scores = self.score_all_items(user_id, n_items_override)
        if exclude:
            for i in exclude:
                if i < len(scores):
                    scores[i] = -np.inf
        return np.argsort(-scores)[:n].tolist()


def measure_fresh_latency(scorer: GPUScorer, user_ids: list,
                          n_items_override: int, n_trials: int = N_TRIALS):
    """Measure mean per-request fresh inference latency (ms)."""
    # Warmup
    for uid in user_ids[:N_WARMUP]:
        scorer.score_all_items(uid, n_items_override)

    lats = []
    for i in range(n_trials):
        uid = user_ids[i % len(user_ids)]
        t0 = time.perf_counter()
        scorer.recommend(uid, n=N_RECS, n_items_override=n_items_override)
        lats.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(lats)), float(np.std(lats))


def measure_spec_latency(spec: SpeculativeRecommender, user_ids: list,
                         n_trials: int = N_TRIALS):
    """Measure mean per-request speculative serving latency (ms)."""
    # Warmup
    for uid in user_ids[:N_WARMUP]:
        spec.recommend(uid)

    lats = []
    accepts = []
    for i in range(n_trials):
        uid = user_ids[i % len(user_ids)]
        res = spec.recommend(uid)
        lats.append(res.latency_ms)
        accepts.append(int(res.accepted))
    return float(np.mean(lats)), float(np.std(lats)), float(np.mean(accepts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-items", nargs="+", type=int,
                        default=[1000, 3706, 10000, 50000, 100000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)
    figures_dir = Path(args.figures_dir); figures_dir.mkdir(exist_ok=True, parents=True)

    cuda_avail = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_avail:
        print("CUDA not available — falling back to CPU")
        args.device = "cpu"

    print(f"S15: GPU Latency Benchmark | device={args.device} | "
          f"catalog_sizes={args.n_items}")

    all_seed_results = []

    for seed in args.seeds:
        print(f"\n  seed={seed}")
        set_seed(seed)

        loader = DataLoader("data")
        train, val, test = loader.load_dataset(args.dataset)
        n_users_real = train.n_users
        n_items_real = train.n_items
        print(f"    {n_users_real} users, {n_items_real} items")

        # Train MF on CPU (avoid nn.Parameter.to(cuda) non-leaf issue)
        mf = MatrixFactorizationRecommender(
            n_users=n_users_real, n_items=n_items_real, embedding_dim=64, device="cpu"
        )
        mf.fit(train.user_ids, train.item_ids, train.ratings, epochs=10, verbose=False)

        # Cluster manager (CPU — used by speculative, not in hot path)
        item_embs_np = mf.get_all_item_embeddings()
        cm = UserClusterManager(n_clusters=50, embedding_dim=64)
        cm.set_item_embeddings(item_embs_np)
        cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

        from collections import defaultdict
        user_history = defaultdict(list)
        for uid, iid in zip(train.user_ids, train.item_ids):
            user_history[int(uid)].append(int(iid))

        # Pick test users that have history
        rng = np.random.default_rng(seed)
        test_users = rng.choice(n_users_real, size=min(500, n_users_real),
                                replace=False).tolist()

        # Build GPU scorer (wraps trained weights as plain tensors on target device).
        # max_catalog ensures item_emb is pre-tiled to cover all benchmark sizes.
        max_cat = max(args.n_items)
        scorer_cpu = GPUScorer(mf, device="cpu",    max_catalog=max_cat)
        scorer_dev = GPUScorer(mf, device=args.device, max_catalog=max_cat)

        # Speculative recommender (runs on CPU — measures cache lookup only)
        config = SpeculativeConfig(top_k_clusters=TOP_K, acceptance_threshold=THRESHOLD,
                                   n_recs=N_RECS, use_pool_retrieval=False)
        spec = SpeculativeRecommender(
            recommender=mf, cluster_manager=cm,
            acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=THRESHOLD),
            config=config, item_embeddings=item_embs_np, user_history=dict(user_history),
        )
        spec.warm_cache(list(range(n_users_real)))
        spec_lat, spec_std, accept_rate = measure_spec_latency(spec, test_users)
        print(f"    Spec lat={spec_lat:.2f}±{spec_std:.2f}ms, accept={accept_rate:.0%}")

        seed_entry = {
            "spec_lat": spec_lat, "spec_std": spec_std, "accept_rate": accept_rate,
            "fresh_cpu": {}, "fresh_gpu": {},
        }

        for n_items in args.n_items:
            # CPU fresh
            lat_cpu, std_cpu = measure_fresh_latency(scorer_cpu, test_users, n_items)
            seed_entry["fresh_cpu"][str(n_items)] = {"lat": lat_cpu, "std": std_cpu}

            if args.device != "cpu":
                lat_gpu, std_gpu = measure_fresh_latency(scorer_dev, test_users, n_items)
                seed_entry["fresh_gpu"][str(n_items)] = {"lat": lat_gpu, "std": std_gpu}
                speedup_cpu = lat_cpu / spec_lat
                speedup_gpu = lat_gpu / spec_lat
                print(f"    n_items={n_items:>7,}: "
                      f"CPU={lat_cpu:.2f}ms (x{speedup_cpu:.1f}), "
                      f"GPU={lat_gpu:.2f}ms (x{speedup_gpu:.1f})")
            else:
                speedup = lat_cpu / spec_lat
                print(f"    n_items={n_items:>7,}: CPU={lat_cpu:.2f}±{std_cpu:.2f}ms "
                      f"(spec speedup x{speedup:.1f})")

        all_seed_results.append(seed_entry)

    # Average across seeds
    def avg_field(field):
        return float(np.mean([r[field] for r in all_seed_results]))

    avg_spec_lat = avg_field("spec_lat")
    avg_fresh_cpu = {}
    avg_fresh_gpu = {}
    for n in args.n_items:
        key = str(n)
        avg_fresh_cpu[key] = float(np.mean([r["fresh_cpu"].get(key, {}).get("lat", 0)
                                            for r in all_seed_results]))
        if args.device != "cpu":
            avg_fresh_gpu[key] = float(np.mean([r["fresh_gpu"].get(key, {}).get("lat", 0)
                                                for r in all_seed_results]))

    avg_accept = avg_field("accept_rate")

    print(f"\n  Summary (avg {len(args.seeds)} seeds):")
    print(f"  Spec  latency = {avg_spec_lat:.2f}ms (accept={avg_accept:.0%})")
    print(f"  {'n_items':<12} {'CPU lat':>10} {'CPU speedup':>12}", end="")
    if args.device != "cpu":
        print(f" {'GPU lat':>10} {'GPU speedup':>12}")
    else:
        print()
    for n in args.n_items:
        key = str(n)
        lat_c = avg_fresh_cpu[key]
        su_c  = lat_c / avg_spec_lat if avg_spec_lat > 0 else 0
        if args.device != "cpu":
            lat_g = avg_fresh_gpu.get(key, 0)
            su_g  = lat_g / avg_spec_lat if avg_spec_lat > 0 else 0
            print(f"  {n:<12,} {lat_c:>10.2f}ms {su_c:>11.1f}x {lat_g:>10.2f}ms {su_g:>11.1f}x")
        else:
            print(f"  {n:<12,} {lat_c:>10.2f}ms {su_c:>11.1f}x")

    results = {
        "device": args.device,
        "seeds": args.seeds,
        "n_items_tested": args.n_items,
        "avg_spec_lat_ms": avg_spec_lat,
        "avg_accept_rate": avg_accept,
        "avg_fresh_cpu_ms": avg_fresh_cpu,
        "avg_fresh_gpu_ms": avg_fresh_gpu,
    }
    out = results_dir / "s15_gpu_latency.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")

    # Figure: latency vs catalog size
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n_vals = args.n_items
    cpu_lats = [avg_fresh_cpu[str(n)] for n in n_vals]
    spec_line = [avg_spec_lat] * len(n_vals)

    ax = axes[0]
    ax.plot(n_vals, cpu_lats, "o-", label="Fresh (CPU)", color="tab:blue")
    if args.device != "cpu" and avg_fresh_gpu:
        gpu_lats = [avg_fresh_gpu.get(str(n), 0) for n in n_vals]
        ax.plot(n_vals, gpu_lats, "s-", label="Fresh (GPU)", color="tab:orange")
    ax.axhline(avg_spec_lat, linestyle="--", color="tab:green", label="Speculative (cache)")
    ax.set_xlabel("Catalog size (items)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xscale("log")
    ax.set_title("S15: Inference Latency vs Catalog Size")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    cpu_speedups = [c / avg_spec_lat for c in cpu_lats]
    ax.plot(n_vals, cpu_speedups, "o-", label="Speedup vs Fresh CPU", color="tab:blue")
    if args.device != "cpu" and avg_fresh_gpu:
        gpu_lats = [avg_fresh_gpu.get(str(n), 0) for n in n_vals]
        gpu_speedups = [g / avg_spec_lat for g in gpu_lats]
        ax.plot(n_vals, gpu_speedups, "s-", label="Speedup vs Fresh GPU", color="tab:orange")
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xlabel("Catalog size (items)")
    ax.set_ylabel("Speedup (fresh / speculative)")
    ax.set_xscale("log")
    ax.set_title("S15: Speculative Speedup vs Catalog Size")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = figures_dir / "s15_gpu_latency.pdf"
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
