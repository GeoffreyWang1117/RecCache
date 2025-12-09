"""Cache warming strategies for RecCache."""

import logging
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
import heapq

logger = logging.getLogger(__name__)


@dataclass
class WarmingPlan:
    """Plan for cache warming."""

    requests: List[Tuple[int, Optional[np.ndarray]]]  # (user_id, context)
    priority_scores: List[float]
    estimated_hit_rate_improvement: float
    n_requests: int


class CacheWarmer:
    """
    Intelligent cache warming strategies.

    Strategies:
    1. Popular users: Warm cache for most active users
    2. Cluster representatives: Warm cache for cluster centers
    3. Temporal: Warm cache based on expected traffic patterns
    4. Hybrid: Combine multiple strategies
    """

    def __init__(
        self,
        cache_manager,
        recommender,
        cluster_manager=None,
    ):
        self.cache_manager = cache_manager
        self.recommender = recommender
        self.cluster_manager = cluster_manager

        # Historical data for planning
        self._user_request_counts: Dict[int, int] = {}
        self._temporal_patterns: Dict[int, List[int]] = {}  # hour -> user_ids

    def record_request(self, user_id: int, hour: int = None):
        """Record a request for warming strategy learning."""
        self._user_request_counts[user_id] = self._user_request_counts.get(user_id, 0) + 1

        if hour is not None:
            if hour not in self._temporal_patterns:
                self._temporal_patterns[hour] = []
            self._temporal_patterns[hour].append(user_id)

    def plan_popular_users(
        self,
        n_users: int = 100,
        n_recommendations: int = 20,
    ) -> WarmingPlan:
        """
        Plan warming for most popular users.

        Returns:
            WarmingPlan with user requests sorted by popularity
        """
        if not self._user_request_counts:
            logger.warning("No request history for popular user warming")
            return WarmingPlan([], [], 0.0, 0)

        # Sort by request count
        sorted_users = sorted(
            self._user_request_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n_users]

        total_requests = sum(self._user_request_counts.values())
        covered_requests = sum(count for _, count in sorted_users)
        estimated_hit_rate = covered_requests / total_requests if total_requests > 0 else 0

        requests = [(user_id, None) for user_id, _ in sorted_users]
        priority_scores = [count / total_requests for _, count in sorted_users]

        return WarmingPlan(
            requests=requests,
            priority_scores=priority_scores,
            estimated_hit_rate_improvement=estimated_hit_rate,
            n_requests=len(requests),
        )

    def plan_cluster_representatives(
        self,
        n_per_cluster: int = 5,
    ) -> WarmingPlan:
        """
        Plan warming for cluster representative users.

        For each cluster, selects users closest to the cluster center.
        """
        if self.cluster_manager is None:
            logger.warning("Cluster manager not available")
            return WarmingPlan([], [], 0.0, 0)

        requests = []
        priority_scores = []

        stats = self.cluster_manager.get_statistics()
        n_clusters = stats["n_clusters"]

        for cluster_id in range(n_clusters):
            # Get users in cluster sorted by distance to center
            users = self.cluster_manager.get_cluster_users(cluster_id)
            if not users:
                continue

            # Sort by distance to center
            user_distances = []
            for user_id in users:
                info = self.cluster_manager.get_user_cluster(user_id)
                user_distances.append((user_id, info.distance_to_center))

            user_distances.sort(key=lambda x: x[1])

            # Select closest users
            for user_id, distance in user_distances[:n_per_cluster]:
                requests.append((user_id, None))
                # Priority inversely proportional to distance
                priority = 1.0 / (1.0 + distance)
                priority_scores.append(priority)

        # Estimate hit rate based on cluster coverage
        cluster_sizes = stats["cluster_sizes"]
        avg_cluster_size = np.mean(cluster_sizes)
        estimated_hit_rate = min(1.0, n_per_cluster / avg_cluster_size) if avg_cluster_size > 0 else 0

        return WarmingPlan(
            requests=requests,
            priority_scores=priority_scores,
            estimated_hit_rate_improvement=estimated_hit_rate,
            n_requests=len(requests),
        )

    def plan_temporal(
        self,
        target_hour: int,
        n_users: int = 100,
    ) -> WarmingPlan:
        """
        Plan warming based on temporal patterns.

        Warms cache for users who typically make requests at the target hour.
        """
        if target_hour not in self._temporal_patterns:
            return WarmingPlan([], [], 0.0, 0)

        # Count requests per user at this hour
        user_counts = Counter(self._temporal_patterns[target_hour])

        # Get top users
        top_users = user_counts.most_common(n_users)

        total_at_hour = len(self._temporal_patterns[target_hour])
        covered = sum(count for _, count in top_users)
        estimated_hit_rate = covered / total_at_hour if total_at_hour > 0 else 0

        requests = [(user_id, None) for user_id, _ in top_users]
        priority_scores = [count / total_at_hour for _, count in top_users]

        return WarmingPlan(
            requests=requests,
            priority_scores=priority_scores,
            estimated_hit_rate_improvement=estimated_hit_rate,
            n_requests=len(requests),
        )

    def plan_hybrid(
        self,
        n_users: int = 200,
        popular_weight: float = 0.4,
        cluster_weight: float = 0.4,
        temporal_weight: float = 0.2,
        target_hour: Optional[int] = None,
    ) -> WarmingPlan:
        """
        Hybrid warming strategy combining multiple approaches.
        """
        # Get individual plans
        popular_plan = self.plan_popular_users(n_users)
        cluster_plan = self.plan_cluster_representatives(n_per_cluster=5)

        temporal_plan = WarmingPlan([], [], 0.0, 0)
        if target_hour is not None:
            temporal_plan = self.plan_temporal(target_hour, n_users // 2)

        # Combine with weights
        user_scores: Dict[int, float] = {}

        # Add popular scores
        for (user_id, _), score in zip(popular_plan.requests, popular_plan.priority_scores):
            user_scores[user_id] = user_scores.get(user_id, 0) + popular_weight * score

        # Add cluster scores
        for (user_id, _), score in zip(cluster_plan.requests, cluster_plan.priority_scores):
            user_scores[user_id] = user_scores.get(user_id, 0) + cluster_weight * score

        # Add temporal scores
        for (user_id, _), score in zip(temporal_plan.requests, temporal_plan.priority_scores):
            user_scores[user_id] = user_scores.get(user_id, 0) + temporal_weight * score

        # Sort by combined score
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:n_users]

        requests = [(user_id, None) for user_id, _ in sorted_users]
        priority_scores = [score for _, score in sorted_users]

        # Estimate combined hit rate
        estimated_hit_rate = (
            popular_weight * popular_plan.estimated_hit_rate_improvement +
            cluster_weight * cluster_plan.estimated_hit_rate_improvement +
            temporal_weight * temporal_plan.estimated_hit_rate_improvement
        )

        return WarmingPlan(
            requests=requests,
            priority_scores=priority_scores,
            estimated_hit_rate_improvement=estimated_hit_rate,
            n_requests=len(requests),
        )

    def execute_warming(
        self,
        plan: WarmingPlan,
        n_recommendations: int = 20,
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        """
        Execute a warming plan.

        Args:
            plan: Warming plan to execute
            n_recommendations: Number of recommendations per user
            batch_size: Batch size for processing
            progress_callback: Callback(completed, total) for progress updates

        Returns:
            Statistics about the warming operation
        """
        from reccache.cache.manager import RecommendationRequest

        n_total = len(plan.requests)
        n_completed = 0
        n_errors = 0

        logger.info(f"Starting cache warming with {n_total} requests")

        for i in range(0, n_total, batch_size):
            batch = plan.requests[i:i + batch_size]

            for user_id, context in batch:
                try:
                    # Generate recommendations
                    recs = self.recommender.recommend(user_id, n=n_recommendations)

                    # Store in cache
                    request = RecommendationRequest(
                        user_id=user_id,
                        context_features=context,
                        n_recommendations=n_recommendations,
                    )
                    self.cache_manager.put(request, recs)

                    n_completed += 1

                except Exception as e:
                    logger.error(f"Error warming cache for user {user_id}: {e}")
                    n_errors += 1

            if progress_callback:
                progress_callback(n_completed, n_total)

        logger.info(f"Cache warming complete: {n_completed}/{n_total} successful, {n_errors} errors")

        return {
            "n_total": n_total,
            "n_completed": n_completed,
            "n_errors": n_errors,
            "estimated_hit_rate": plan.estimated_hit_rate_improvement,
        }


class IncrementalWarmer:
    """
    Incremental cache warmer that runs in the background.

    Continuously warms cache based on observed patterns.
    """

    def __init__(
        self,
        cache_warmer: CacheWarmer,
        update_interval: int = 1000,  # requests
        warmup_batch_size: int = 10,
    ):
        self.cache_warmer = cache_warmer
        self.update_interval = update_interval
        self.warmup_batch_size = warmup_batch_size

        self._request_count = 0
        self._last_warmup = 0

    def on_request(self, user_id: int, hour: int = None):
        """Call on each request to update patterns and trigger warming."""
        self.cache_warmer.record_request(user_id, hour)
        self._request_count += 1

        # Check if we should trigger warming
        if self._request_count - self._last_warmup >= self.update_interval:
            self._trigger_warmup()
            self._last_warmup = self._request_count

    def _trigger_warmup(self):
        """Trigger incremental cache warming."""
        # Use hybrid strategy with small batch
        plan = self.cache_warmer.plan_hybrid(n_users=self.warmup_batch_size)

        if plan.n_requests > 0:
            self.cache_warmer.execute_warming(plan)


def create_warmup_schedule(
    hours: List[int] = None,
    strategies: List[str] = None,
) -> List[Dict]:
    """
    Create a scheduled warmup plan.

    Args:
        hours: Hours of day to run warmup (0-23)
        strategies: List of strategies to use

    Returns:
        Schedule as list of {hour, strategy} dicts
    """
    if hours is None:
        # Default: warm before peak hours
        hours = [6, 11, 17]  # Before morning, lunch, evening peaks

    if strategies is None:
        strategies = ["hybrid"]

    schedule = []
    for hour in hours:
        for strategy in strategies:
            schedule.append({
                "hour": hour,
                "strategy": strategy,
                "config": {
                    "n_users": 200 if strategy == "hybrid" else 100,
                },
            })

    return schedule
