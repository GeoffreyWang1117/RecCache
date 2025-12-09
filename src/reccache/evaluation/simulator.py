"""Online traffic simulation for cache evaluation."""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from reccache.cache.manager import CacheManager, RecommendationRequest, CacheAwareRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.evaluation.metrics import RecommendationMetrics, CacheEvaluator, compute_list_similarity
from reccache.models.reranker import LightweightReranker


logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for traffic simulation."""

    n_requests: int = 10000
    n_warmup_requests: int = 1000

    # Traffic pattern
    user_distribution: str = "zipf"  # "uniform", "zipf", "clustered"
    zipf_alpha: float = 1.2

    # Temporal patterns
    time_window_hours: int = 24
    peak_hours: List[int] = field(default_factory=lambda: [12, 18, 20])

    # Evaluation
    eval_sample_rate: float = 0.1  # Fraction of requests to evaluate quality
    k: int = 10  # Cutoff for metrics


@dataclass
class SimulationResult:
    """Results from simulation run."""

    # Cache performance
    hit_rate: float
    local_hit_rate: float
    redis_hit_rate: float
    miss_rate: float

    # Latency
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Quality
    avg_ndcg: float
    ndcg_degradation: float  # vs. fresh recommendations
    avg_list_similarity: float

    # Cost
    compute_saved_pct: float
    estimated_cost_reduction: float

    # Details
    n_requests: int
    duration_seconds: float
    requests_per_second: float


class OnlineSimulator:
    """
    Simulate online recommendation traffic with caching.

    Features:
    - Realistic traffic patterns (Zipf distribution)
    - Temporal patterns (peak hours)
    - Quality tracking
    - A/B comparison (cached vs. fresh)
    """

    def __init__(
        self,
        recommender,
        cache_manager: CacheManager,
        cluster_manager: Optional[UserClusterManager] = None,
        reranker: Optional[LightweightReranker] = None,
        config: Optional[SimulationConfig] = None,
    ):
        self.recommender = recommender
        self.cache_manager = cache_manager
        self.cluster_manager = cluster_manager
        self.reranker = reranker
        self.config = config or SimulationConfig()

        # Wrapped recommender with caching
        self.cached_recommender = CacheAwareRecommender(
            recommender=recommender,
            cache_manager=cache_manager,
            reranker=reranker,
        )

        # State
        self._user_history: Dict[int, List[int]] = defaultdict(list)
        self._request_log: List[Dict] = []

        # For evaluation
        self._cache_evaluator = CacheEvaluator(k=self.config.k)

    def set_ground_truth(self, ground_truth: Dict[int, set]):
        """Set ground truth relevant items for evaluation."""
        self._ground_truth = ground_truth

    def generate_traffic(
        self,
        n_users: int,
        n_items: int,
        n_requests: int,
    ) -> Generator[Tuple[int, Optional[np.ndarray], float], None, None]:
        """
        Generate synthetic traffic.

        Yields:
            (user_id, context_features, timestamp)
        """
        # Generate user request probabilities based on distribution
        if self.config.user_distribution == "zipf":
            probs = np.array([1.0 / (i + 1) ** self.config.zipf_alpha for i in range(n_users)])
            probs /= probs.sum()
        elif self.config.user_distribution == "clustered":
            # More requests from certain user clusters
            probs = np.random.dirichlet(np.ones(n_users) * 0.5)
        else:  # uniform
            probs = np.ones(n_users) / n_users

        # Generate timestamps
        base_time = time.time()
        time_window = self.config.time_window_hours * 3600

        for i in range(n_requests):
            # Sample user
            user_id = np.random.choice(n_users, p=probs)

            # Generate timestamp with peak hours
            progress = i / n_requests
            hour = int((progress * 24) % 24)

            # Boost probability during peak hours
            if hour in self.config.peak_hours:
                time_offset = progress * time_window + np.random.normal(0, 300)
            else:
                time_offset = progress * time_window + np.random.normal(0, 600)

            timestamp = base_time + time_offset

            # Generate context features
            context = self._generate_context(timestamp)

            yield user_id, context, timestamp

    def _generate_context(self, timestamp: float) -> np.ndarray:
        """Generate context features from timestamp."""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)

        hour = dt.hour
        day = dt.weekday()

        return np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 7),
            np.cos(2 * np.pi * day / 7),
            1.0 if day >= 5 else 0.0,
        ], dtype=np.float32)

    def run_simulation(
        self,
        n_users: int,
        n_items: int,
        ground_truth: Optional[Dict[int, set]] = None,
        verbose: bool = True,
    ) -> SimulationResult:
        """
        Run full simulation.

        Args:
            n_users: Number of users
            n_items: Number of items
            ground_truth: Optional ground truth for quality evaluation
            verbose: Print progress

        Returns:
            SimulationResult with all metrics
        """
        if ground_truth:
            self._ground_truth = ground_truth

        n_total = self.config.n_requests + self.config.n_warmup_requests

        latencies = []
        ndcg_scores = []
        list_similarities = []
        cache_hits = {"local": 0, "redis": 0, "miss": 0}

        start_time = time.time()

        traffic_gen = self.generate_traffic(n_users, n_items, n_total)

        for i, (user_id, context, timestamp) in enumerate(traffic_gen):
            is_warmup = i < self.config.n_warmup_requests

            # Get recommendations
            req_start = time.time()
            recs, metadata = self.cached_recommender.recommend(
                user_id=user_id,
                n=20,
                context_features=context,
                timestamp=timestamp,
            )
            latency = (time.time() - req_start) * 1000

            if not is_warmup:
                latencies.append(latency)

                # Track cache level
                if metadata.get("cache_hit"):
                    if metadata.get("cache_level") == "local":
                        cache_hits["local"] += 1
                    else:
                        cache_hits["redis"] += 1
                else:
                    cache_hits["miss"] += 1

                # Quality evaluation (sample)
                if (
                    hasattr(self, "_ground_truth")
                    and user_id in self._ground_truth
                    and np.random.random() < self.config.eval_sample_rate
                ):
                    relevant = self._ground_truth[user_id]
                    ndcg = RecommendationMetrics.ndcg_at_k(recs, relevant, self.config.k)
                    ndcg_scores.append(ndcg)

                    # Compare with fresh if cache hit
                    if metadata.get("cache_hit"):
                        fresh_recs = self.recommender.recommend(user_id, n=20)
                        sim = compute_list_similarity(recs, fresh_recs, self.config.k)
                        list_similarities.append(sim["overlap"])

                        fresh_ndcg = RecommendationMetrics.ndcg_at_k(
                            fresh_recs, relevant, self.config.k
                        )
                        self._cache_evaluator.add_comparison(
                            user_id=user_id,
                            cached_recs=recs,
                            fresh_recs=fresh_recs,
                            relevant_items=relevant,
                            cache_hit=True,
                            latency_cached_ms=latency,
                            latency_fresh_ms=metadata.get("compute_time_ms", 50),
                        )

            # Log request
            self._request_log.append({
                "user_id": user_id,
                "timestamp": timestamp,
                "latency_ms": latency,
                "cache_hit": metadata.get("cache_hit", False),
                "cache_level": metadata.get("cache_level", "miss"),
            })

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rps = (i + 1) / elapsed
                print(f"Progress: {i + 1}/{n_total} requests ({rps:.1f} req/s)")

        duration = time.time() - start_time
        n_eval = self.config.n_requests

        # Compute latency percentiles
        latencies = np.array(latencies)

        # Cache metrics
        total_hits = cache_hits["local"] + cache_hits["redis"]
        hit_rate = total_hits / n_eval if n_eval > 0 else 0

        # Quality metrics
        cache_metrics = self._cache_evaluator.compute_metrics()

        return SimulationResult(
            hit_rate=hit_rate,
            local_hit_rate=cache_hits["local"] / n_eval if n_eval > 0 else 0,
            redis_hit_rate=cache_hits["redis"] / n_eval if n_eval > 0 else 0,
            miss_rate=cache_hits["miss"] / n_eval if n_eval > 0 else 0,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            avg_ndcg=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            ndcg_degradation=cache_metrics.quality_loss,
            avg_list_similarity=float(np.mean(list_similarities)) if list_similarities else 1.0,
            compute_saved_pct=hit_rate * 100,
            estimated_cost_reduction=hit_rate * 0.8,  # Assume 80% cost is compute
            n_requests=n_eval,
            duration_seconds=duration,
            requests_per_second=n_eval / duration if duration > 0 else 0,
        )

    def run_ab_test(
        self,
        n_users: int,
        n_items: int,
        n_requests: int,
        ground_truth: Dict[int, set],
    ) -> Dict:
        """
        Run A/B test comparing cached vs. fresh recommendations.

        Returns detailed comparison metrics.
        """
        self._ground_truth = ground_truth

        results_cached = []
        results_fresh = []

        traffic_gen = self.generate_traffic(n_users, n_items, n_requests * 2)

        for i, (user_id, context, timestamp) in enumerate(traffic_gen):
            if user_id not in ground_truth:
                continue

            relevant = ground_truth[user_id]

            # Alternate between cached and fresh
            if i % 2 == 0:
                # Cached path
                start = time.time()
                recs, metadata = self.cached_recommender.recommend(
                    user_id=user_id,
                    n=20,
                    context_features=context,
                    timestamp=timestamp,
                )
                latency = (time.time() - start) * 1000

                ndcg = RecommendationMetrics.ndcg_at_k(recs, relevant, self.config.k)
                results_cached.append({
                    "ndcg": ndcg,
                    "latency_ms": latency,
                    "cache_hit": metadata.get("cache_hit", False),
                })
            else:
                # Fresh path (bypass cache)
                start = time.time()
                recs = self.recommender.recommend(user_id, n=20)
                latency = (time.time() - start) * 1000

                ndcg = RecommendationMetrics.ndcg_at_k(recs, relevant, self.config.k)
                results_fresh.append({
                    "ndcg": ndcg,
                    "latency_ms": latency,
                })

        # Aggregate results
        cached_ndcgs = [r["ndcg"] for r in results_cached]
        fresh_ndcgs = [r["ndcg"] for r in results_fresh]
        cached_latencies = [r["latency_ms"] for r in results_cached]
        fresh_latencies = [r["latency_ms"] for r in results_fresh]

        return {
            "cached": {
                "avg_ndcg": np.mean(cached_ndcgs),
                "std_ndcg": np.std(cached_ndcgs),
                "avg_latency_ms": np.mean(cached_latencies),
                "p50_latency_ms": np.percentile(cached_latencies, 50),
                "p95_latency_ms": np.percentile(cached_latencies, 95),
                "hit_rate": np.mean([r["cache_hit"] for r in results_cached]),
                "n_samples": len(results_cached),
            },
            "fresh": {
                "avg_ndcg": np.mean(fresh_ndcgs),
                "std_ndcg": np.std(fresh_ndcgs),
                "avg_latency_ms": np.mean(fresh_latencies),
                "p50_latency_ms": np.percentile(fresh_latencies, 50),
                "p95_latency_ms": np.percentile(fresh_latencies, 95),
                "n_samples": len(results_fresh),
            },
            "comparison": {
                "ndcg_diff": np.mean(cached_ndcgs) - np.mean(fresh_ndcgs),
                "ndcg_ratio": np.mean(cached_ndcgs) / np.mean(fresh_ndcgs) if np.mean(fresh_ndcgs) > 0 else 1.0,
                "latency_reduction_pct": (
                    (np.mean(fresh_latencies) - np.mean(cached_latencies))
                    / np.mean(fresh_latencies) * 100
                    if np.mean(fresh_latencies) > 0 else 0
                ),
            },
        }

    def get_request_log(self) -> List[Dict]:
        """Get detailed request log."""
        return self._request_log

    def clear(self):
        """Clear simulation state."""
        self._request_log = []
        self._cache_evaluator.clear()
        self._user_history.clear()


def analyze_cluster_effectiveness(
    cluster_manager: UserClusterManager,
    recommender,
    ground_truth: Dict[int, set],
    k: int = 10,
    n_samples: int = 100,
) -> Dict:
    """
    Analyze how well clustering captures user similarity.

    Returns metrics on intra-cluster recommendation similarity.
    """
    stats = cluster_manager.get_statistics()
    cluster_sizes = stats["cluster_sizes"]

    results = []

    for cluster_id in range(len(cluster_sizes)):
        if cluster_sizes[cluster_id] < 2:
            continue

        # Get users in this cluster
        users = cluster_manager.get_cluster_users(cluster_id)
        if len(users) < 2:
            continue

        # Sample pairs
        n_pairs = min(n_samples, len(users) * (len(users) - 1) // 2)
        pairs = []
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                pairs.append((users[i], users[j]))
        pairs = pairs[:n_pairs]

        # Compute recommendation similarity for pairs
        similarities = []
        for u1, u2 in pairs:
            recs1 = recommender.recommend(u1, n=k * 2)
            recs2 = recommender.recommend(u2, n=k * 2)
            sim = compute_list_similarity(recs1, recs2, k)
            similarities.append(sim["overlap"])

        if similarities:
            results.append({
                "cluster_id": cluster_id,
                "cluster_size": cluster_sizes[cluster_id],
                "avg_similarity": np.mean(similarities),
                "std_similarity": np.std(similarities),
                "n_pairs": len(pairs),
            })

    return {
        "cluster_analysis": results,
        "overall_avg_similarity": np.mean([r["avg_similarity"] for r in results]) if results else 0,
        "n_clusters_analyzed": len(results),
    }
