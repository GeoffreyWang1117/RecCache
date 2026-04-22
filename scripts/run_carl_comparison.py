#!/usr/bin/env python3
"""
CARL Comparison Experiment: RL-based caching vs Speculative Serving.

Implements a simplified CARL-like RL caching baseline (DQN policy that
decides serve-cached vs compute-fresh per user) and compares against
our training-free speculative serving framework.

CARL (WWW 2024) uses RL to jointly optimize real-time and cached recs.
Since no public code exists, we implement the core mechanism: a DQN
policy network that observes (user_embedding, cluster_distance,
cache_age, request_count) and outputs serve_cached/compute_fresh.

Usage:
    conda activate reccache
    python scripts/run_carl_comparison.py
    python scripts/run_carl_comparison.py --dataset ml-1m
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict, deque
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.evaluation.metrics import (
    RecommendationMetrics,
    SpeculativeMetrics,
    compute_coverage,
)


# =========================================================================
# CARL-like DQN caching policy
# =========================================================================
class CacheDecisionNetwork(nn.Module):
    """DQN for cache serve/compute decision."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 actions: serve_cached, compute_fresh
        )

    def forward(self, state):
        return self.net(state)


class CARLBaseline:
    """
    Simplified CARL-like RL caching policy.

    State: [user_embedding(d), cluster_distance, cache_age_normalized,
            user_request_count_normalized]
    Action: 0=serve_cached, 1=compute_fresh
    Reward: alpha * quality + (1-alpha) * speedup_indicator
    """

    def __init__(
        self,
        recommender,
        cluster_manager,
        item_embeddings,
        embedding_dim=64,
        n_recs=10,
        alpha=0.5,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        device="cpu",
        user_history=None,
    ):
        self.recommender = recommender
        self.cluster_manager = cluster_manager
        self.item_embeddings = item_embeddings
        self.n_recs = n_recs
        self.alpha = alpha
        self.user_history = user_history or {}
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device

        # State dimension: embedding_dim + 3 (dist, age, count)
        self.state_dim = embedding_dim + 3
        self.policy_net = CacheDecisionNetwork(self.state_dim).to(device)
        self.target_net = CacheDecisionNetwork(self.state_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

        # Cluster caches: cluster_id -> list of item_ids
        self._cluster_cache = {}
        self._cache_timestamps = {}
        self._request_counts = defaultdict(int)
        self._current_step = 0

    def warm_cache(self, user_ids):
        """Warm cluster caches using training users."""
        cluster_recs = defaultdict(list)
        for uid in user_ids:
            cid = self.cluster_manager.get_user_cluster(uid).cluster_id
            if cid is not None:
                exclude = self.user_history.get(uid)
                recs = self.recommender.recommend(uid, n=self.n_recs, exclude_items=exclude)
                cluster_recs[cid].extend(recs)

        for cid, items in cluster_recs.items():
            # Use most common items as cluster cache
            from collections import Counter
            counts = Counter(items)
            self._cluster_cache[cid] = [
                item for item, _ in counts.most_common(self.n_recs)
            ]
            self._cache_timestamps[cid] = self._current_step

    def _get_state(self, user_id):
        """Build state vector for DQN."""
        user_emb = self.cluster_manager.get_user_embedding(user_id)
        if user_emb is None:
            user_emb = np.zeros(self.state_dim - 3)

        info = self.cluster_manager.get_user_cluster(user_id)
        cid = info.cluster_id
        dist = info.distance_to_center
        cache_age = (self._current_step - self._cache_timestamps.get(cid, 0)) / max(self._current_step, 1)

        req_count = self._request_counts[user_id] / 100.0  # normalize

        state = np.concatenate([
            user_emb[:self.state_dim - 3],
            [dist, cache_age, req_count]
        ]).astype(np.float32)
        return state

    def _select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(1).item()

    def _compute_reward(self, action, user_id, ground_truth):
        """Compute reward: alpha*quality + (1-alpha)*speed."""
        cid = self.cluster_manager.get_user_cluster(user_id).cluster_id

        if action == 0:  # serve cached
            if cid is not None and cid in self._cluster_cache:
                recs = self._cluster_cache[cid]
            else:
                recs = []
            speed_reward = 1.0  # fast
        else:  # compute fresh
            exclude = self.user_history.get(user_id)
            recs = self.recommender.recommend(user_id, n=self.n_recs, exclude_items=exclude)
            speed_reward = 0.0  # slow

        # Quality reward
        gt = ground_truth.get(user_id, set())
        ndcg = RecommendationMetrics.ndcg_at_k(recs, gt, self.n_recs) if gt else 0.0

        reward = self.alpha * ndcg + (1 - self.alpha) * speed_reward
        return reward, recs

    def _update_network(self):
        """Train DQN from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_policy(self, train_user_ids, ground_truth, n_episodes=5, verbose=True):
        """Train the RL caching policy."""
        t0 = time.time()
        total_rewards = []

        for episode in range(n_episodes):
            episode_reward = 0
            random.shuffle(train_user_ids)

            for uid in train_user_ids[:1000]:  # limit per episode
                self._current_step += 1
                self._request_counts[uid] += 1

                state = self._get_state(uid)
                action = self._select_action(state, training=True)
                reward, recs = self._compute_reward(action, uid, ground_truth)

                # Next state (next user)
                next_uid = random.choice(train_user_ids)
                next_state = self._get_state(next_uid)

                self.replay_buffer.append((state, action, reward, next_state, 0.0))
                self._update_network()

                episode_reward += reward

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Update target network
            if episode % 2 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            avg_reward = episode_reward / min(len(train_user_ids), 1000)
            total_rewards.append(avg_reward)

            if verbose:
                print(f"    Episode {episode+1}/{n_episodes}: "
                      f"avg_reward={avg_reward:.4f}, epsilon={self.epsilon:.3f}")

        train_time = time.time() - t0
        return {"train_time": train_time, "rewards": total_rewards}

    def recommend(self, user_id):
        """Serve recommendation using trained policy."""
        self._current_step += 1
        self._request_counts[user_id] += 1

        state = self._get_state(user_id)
        action = self._select_action(state, training=False)

        cid = self.cluster_manager.get_user_cluster(user_id).cluster_id

        if action == 0 and cid is not None and cid in self._cluster_cache:
            # Serve cached
            recs = self._cluster_cache[cid]
            accepted = True
            speedup = 50.0
        else:
            # Compute fresh
            exclude = self.user_history.get(user_id)
            recs = self.recommender.recommend(user_id, n=self.n_recs, exclude_items=exclude)
            accepted = False
            speedup = 1.0

        return recs, accepted, speedup


# =========================================================================
# Experiment logic
# =========================================================================
DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None,
        "min_user": 5, "min_item": 5,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
    "amazon-electronics": {
        "max_samples": 1000000,
        "min_user": 3, "min_item": 3,
        "implicit": False, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64,
        "epochs": 15,
    },
}

N_RUNS = 3
N_TEST_USERS = 500
N_RECS = 10


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


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


def evaluate_carl(carl, ground_truth, test_users, n_items):
    """Evaluate CARL baseline."""
    recommendations = {}
    user_ndcgs = {}
    accepted_count = 0

    for uid in test_users:
        if uid not in ground_truth:
            continue
        recs, accepted, speedup = carl.recommend(uid)
        recommendations[uid] = recs
        user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(recs, ground_truth[uid], N_RECS)
        if accepted:
            accepted_count += 1

    n_eval = len(user_ndcgs)
    if n_eval == 0:
        return {"ndcg": 0.0, "accept_rate": 0.0, "coverage": 0.0, "speedup": 1.0}

    accept_rate = accepted_count / n_eval
    speedup = 1.0 / (accept_rate * (1/50.0) + (1 - accept_rate) * 1.0) if n_eval > 0 else 1.0

    return {
        "ndcg": float(np.mean(list(user_ndcgs.values()))),
        "accept_rate": accept_rate,
        "coverage": compute_coverage(recommendations, n_items),
        "speedup": speedup,
    }


def evaluate_speculative(recommender, cluster_manager, ground_truth,
                         test_users, item_embeddings, use_pool=False,
                         user_history=None, fresh_latency_ms=0.0):
    """Evaluate our speculative serving."""
    criterion = ScoreRatioAcceptanceCriterion(threshold=0.35)
    config = SpeculativeConfig(
        top_k_clusters=3,
        acceptance_threshold=0.35,
        n_recs=N_RECS,
        use_pool_retrieval=use_pool,
        pool_size=200,
    )
    spec = SpeculativeRecommender(
        recommender=recommender,
        cluster_manager=cluster_manager,
        acceptance_criterion=criterion,
        config=config,
        item_embeddings=item_embeddings,
        user_history=user_history,
    )

    all_users = list(set(int(u) for u in cluster_manager._user_embeddings.keys()))
    spec.warm_cache(all_users)

    results = []
    recommendations = {}
    user_ndcgs = {}

    for uid in test_users:
        if uid not in ground_truth:
            continue
        sr = spec.recommend(uid)
        results.append(sr)
        recommendations[uid] = sr.items
        user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(
            sr.items, ground_truth[uid], N_RECS
        )

    if not results:
        return {"ndcg": 0.0, "accept_rate": 0.0, "coverage": 0.0,
                "speedup": 1.0, "mcg": 0.0}

    return {
        "ndcg": float(np.mean(list(user_ndcgs.values()))),
        "accept_rate": SpeculativeMetrics.acceptance_rate(results),
        "coverage": compute_coverage(recommendations, item_embeddings.shape[0]),
        "speedup": SpeculativeMetrics.speedup_estimate(results, fresh_latency_ms=fresh_latency_ms),
        "mcg": SpeculativeMetrics.multi_cluster_gain(results),
    }


def evaluate_fresh(model, ground_truth, test_users, item_embeddings,
                   user_history=None):
    recommendations = {}
    user_ndcgs = {}
    for uid in test_users:
        if uid not in ground_truth:
            continue
        exclude = user_history.get(uid) if user_history else None
        recs = list(model.recommend(uid, n=N_RECS, exclude_items=exclude))
        recommendations[uid] = recs
        user_ndcgs[uid] = RecommendationMetrics.ndcg_at_k(recs, ground_truth[uid], N_RECS)

    if not user_ndcgs:
        return {"ndcg": 0.0, "accept_rate": 0.0, "coverage": 0.0, "speedup": 1.0}

    return {
        "ndcg": float(np.mean(list(user_ndcgs.values()))),
        "accept_rate": 0.0,
        "coverage": compute_coverage(recommendations, item_embeddings.shape[0]),
        "speedup": 1.0,
    }


def run_multi_seed(eval_fn, n_runs=N_RUNS):
    run_metrics = defaultdict(list)
    for run_i in range(n_runs):
        set_seed(42 + run_i)
        m = eval_fn(run_i)
        for metric, val in m.items():
            run_metrics[metric].append(val)
    return {
        metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for metric, vals in run_metrics.items()
    }


def run_experiment(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"\n{'='*70}")
    print(f"  CARL Comparison — {dataset_name}")
    print(f"{'='*70}")

    loader = DataLoader("data")
    train, val, test = loader.load_dataset(
        dataset_name,
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
        max_samples=cfg["max_samples"],
    )
    gt = build_ground_truth(test, min_rating=cfg["min_rating_gt"])
    test_users = [u for u in set(test.user_ids.tolist()) if u in gt][:N_TEST_USERS]
    train_user_list = list(set(train.user_ids.tolist()))

    user_hist = build_user_history(train)

    print(f"  {train.n_users} users, {train.n_items} items, "
          f"{len(gt)} GT users, {len(test_users)} test users")

    # Train base model
    print(f"\n  Training base MF model...", flush=True)
    model = MatrixFactorizationRecommender(
        n_users=train.n_users, n_items=train.n_items,
        embedding_dim=cfg["embedding_dim"],
    )
    model.fit(train.user_ids, train.item_ids, train.ratings,
              epochs=cfg["epochs"], verbose=True)
    item_embs = model.get_all_item_embeddings()

    # Measure fresh latency empirically
    _latencies = []
    for uid in test_users[:50]:
        if uid not in gt:
            continue
        exclude = user_hist.get(uid)
        _t0 = time.perf_counter()
        model.recommend(uid, n=N_RECS, exclude_items=exclude)
        _latencies.append((time.perf_counter() - _t0) * 1000)
    fresh_latency_ms = float(np.mean(_latencies)) if _latencies else 0.3

    # Build cluster manager
    n_clusters = min(cfg["n_clusters"], train.n_users // 2)
    cm = UserClusterManager(
        n_clusters=n_clusters,
        embedding_dim=item_embs.shape[1],
        n_items=len(item_embs),
    )
    cm.set_item_embeddings(item_embs)
    cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    results = {}

    # 1. Fresh baseline
    print(f"\n  Evaluating Fresh (upper bound)...", flush=True)
    results["Fresh"] = run_multi_seed(
        lambda run_i: evaluate_fresh(model, gt, test_users, item_embs,
                                     user_history=user_hist)
    )

    # 2. CARL-like RL baseline
    print(f"\n  Training CARL-like RL policy...", flush=True)
    carl_train_times = []
    carl_results_list = defaultdict(list)

    for run_i in range(N_RUNS):
        set_seed(42 + run_i)
        carl = CARLBaseline(
            recommender=model,
            cluster_manager=cm,
            item_embeddings=item_embs,
            embedding_dim=cfg["embedding_dim"],
            n_recs=N_RECS,
            alpha=0.5,
            user_history=user_hist,
        )
        carl.warm_cache(train_user_list[:500])

        train_info = carl.train_policy(
            train_user_list, gt, n_episodes=5, verbose=(run_i == 0)
        )
        carl_train_times.append(train_info["train_time"])

        metrics = evaluate_carl(carl, gt, test_users, train.n_items)
        for k, v in metrics.items():
            carl_results_list[k].append(v)

    results["CARL-like"] = {
        metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for metric, vals in carl_results_list.items()
    }
    results["CARL-like"]["_train_time_s"] = {
        "mean": float(np.mean(carl_train_times)),
        "std": float(np.std(carl_train_times)),
    }

    # 3. Speculative (ours, static)
    print(f"\n  Evaluating Speculative (ours, static)...", flush=True)
    results["Spec (ours)"] = run_multi_seed(
        lambda run_i: evaluate_speculative(
            model, cm, gt, test_users, item_embs, use_pool=False,
            user_history=user_hist, fresh_latency_ms=fresh_latency_ms,
        )
    )

    # 4. Speculative + Pool (ours)
    print(f"\n  Evaluating Speculative + Pool (ours)...", flush=True)
    results["Spec+Pool (ours)"] = run_multi_seed(
        lambda run_i: evaluate_speculative(
            model, cm, gt, test_users, item_embs, use_pool=True,
            user_history=user_hist, fresh_latency_ms=fresh_latency_ms,
        )
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS — {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'NDCG':>8} {'Accept':>8} {'Speedup':>8} {'Cov':>8}")
    print(f"  {'-'*52}")
    for method in ["Fresh", "CARL-like", "Spec (ours)", "Spec+Pool (ours)"]:
        m = results[method]
        ndcg_str = f"{m['ndcg']['mean']:.4f}±{m['ndcg']['std']:.4f}"
        accept = m['accept_rate']['mean']
        speedup = m['speedup']['mean']
        cov = m['coverage']['mean']
        print(f"  {method:<20} {ndcg_str:>16} {accept:7.1%} {speedup:7.1f}x {cov:8.4f}")

    if "_train_time_s" in results["CARL-like"]:
        carl_t = results["CARL-like"]["_train_time_s"]["mean"]
        print(f"\n  CARL RL training time: {carl_t:.1f}s")
        print(f"  Speculative training time: 0s (training-free)")

    return results


def make_comparison_figure(all_results, output_dir):
    """Bar chart comparing CARL vs Speculative."""
    for dataset, data in all_results.items():
        methods = ["Fresh", "CARL-like", "Spec (ours)", "Spec+Pool (ours)"]
        x = np.arange(len(methods))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # NDCG
        ndcgs = [data[m]["ndcg"]["mean"] for m in methods]
        stds = [data[m]["ndcg"]["std"] for m in methods]
        colors = ["#999999", "#e74c3c", "#3498db", "#2ecc71"]
        axes[0].bar(x, ndcgs, yerr=stds, color=colors, capsize=4, alpha=0.85)
        axes[0].set_ylabel("NDCG@10")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        axes[0].set_title("Quality")
        axes[0].grid(axis="y", alpha=0.3)

        # Acceptance Rate
        accepts = [data[m]["accept_rate"]["mean"] for m in methods]
        axes[1].bar(x, accepts, color=colors, alpha=0.85)
        axes[1].set_ylabel("Acceptance Rate")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        axes[1].set_title("Cache Utilization")
        axes[1].grid(axis="y", alpha=0.3)

        # Coverage
        covs = [data[m]["coverage"]["mean"] for m in methods]
        axes[2].bar(x, covs, color=colors, alpha=0.85)
        axes[2].set_ylabel("Coverage")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        axes[2].set_title("Catalog Coverage")
        axes[2].grid(axis="y", alpha=0.3)

        plt.suptitle(f"CARL vs Speculative Serving — {dataset}", fontsize=12)
        plt.tight_layout()

        for ext in ["pdf", "png"]:
            fig.savefig(output_dir / f"carl_comparison_{dataset}.{ext}",
                        dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+",
                        default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ds in args.dataset:
        all_results[ds] = run_experiment(ds)

    out_path = results_dir / "carl_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    make_comparison_figure(all_results, figures_dir)
    print(f"Figures saved to paper/figures/carl_comparison_*.{{pdf,png}}")


if __name__ == "__main__":
    main()
