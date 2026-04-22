"""Evaluation metrics for recommendation quality and cache performance."""

import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class RankingMetrics:
    """Container for ranking metrics."""

    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    hit_rate: float
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision


@dataclass
class CacheMetrics:
    """Container for cache performance metrics."""

    hit_rate: float
    miss_rate: float
    latency_reduction: float  # Percentage reduction in latency
    quality_loss: float  # Average NDCG degradation
    cost_savings: float  # Percentage of compute saved


class RecommendationMetrics:
    """
    Compute recommendation quality metrics.

    Supports:
    - Precision/Recall @ K
    - NDCG @ K
    - Hit Rate
    - MRR (Mean Reciprocal Rank)
    - MAP (Mean Average Precision)
    """

    @staticmethod
    def precision_at_k(
        recommended: List[int],
        relevant: Set[int],
        k: int,
    ) -> float:
        """
        Precision @ K: fraction of top-K recommendations that are relevant.
        """
        if k <= 0:
            return 0.0

        recommended_k = recommended[:k]
        n_relevant = sum(1 for item in recommended_k if item in relevant)
        return n_relevant / k

    @staticmethod
    def recall_at_k(
        recommended: List[int],
        relevant: Set[int],
        k: int,
    ) -> float:
        """
        Recall @ K: fraction of relevant items in top-K.
        """
        if not relevant or k <= 0:
            return 0.0

        recommended_k = recommended[:k]
        n_relevant = sum(1 for item in recommended_k if item in relevant)
        return n_relevant / len(relevant)

    @staticmethod
    def ndcg_at_k(
        recommended: List[int],
        relevant: Set[int],
        k: int,
        relevance_scores: Optional[Dict[int, float]] = None,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain @ K.

        Args:
            recommended: Ordered list of recommended items
            relevant: Set of relevant items
            k: Cutoff
            relevance_scores: Optional dict of item -> relevance score
        """
        if k <= 0 or not relevant:
            return 0.0

        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                if relevance_scores:
                    rel = relevance_scores.get(item, 1.0)
                else:
                    rel = 1.0
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG
        if relevance_scores:
            ideal_rels = sorted(
                [relevance_scores.get(item, 1.0) for item in relevant],
                reverse=True,
            )[:k]
        else:
            ideal_rels = [1.0] * min(k, len(relevant))

        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate(
        recommended: List[int],
        relevant: Set[int],
        k: int,
    ) -> float:
        """
        Hit Rate @ K: 1 if any relevant item in top-K, else 0.
        """
        recommended_k = set(recommended[:k])
        return 1.0 if recommended_k & relevant else 0.0

    @staticmethod
    def mrr(
        recommended: List[int],
        relevant: Set[int],
    ) -> float:
        """
        Mean Reciprocal Rank: 1/rank of first relevant item.
        """
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def average_precision(
        recommended: List[int],
        relevant: Set[int],
    ) -> float:
        """
        Average Precision: average of precision @ each relevant position.
        """
        if not relevant:
            return 0.0

        precisions = []
        n_relevant = 0

        for i, item in enumerate(recommended):
            if item in relevant:
                n_relevant += 1
                precisions.append(n_relevant / (i + 1))

        return np.mean(precisions) if precisions else 0.0

    @classmethod
    def compute_all(
        cls,
        recommended: List[int],
        relevant: Set[int],
        k: int = 10,
        relevance_scores: Optional[Dict[int, float]] = None,
    ) -> RankingMetrics:
        """Compute all ranking metrics."""
        return RankingMetrics(
            precision_at_k=cls.precision_at_k(recommended, relevant, k),
            recall_at_k=cls.recall_at_k(recommended, relevant, k),
            ndcg_at_k=cls.ndcg_at_k(recommended, relevant, k, relevance_scores),
            hit_rate=cls.hit_rate(recommended, relevant, k),
            mrr=cls.mrr(recommended, relevant),
            map_score=cls.average_precision(recommended, relevant),
        )

    @classmethod
    def evaluate_recommendations(
        cls,
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for all users.

        Args:
            recommendations: Dict of user_id -> recommended items
            ground_truth: Dict of user_id -> relevant items
            k: Cutoff for metrics

        Returns:
            Averaged metrics across all users
        """
        all_metrics = []

        for user_id, recs in recommendations.items():
            if user_id not in ground_truth:
                continue

            relevant = ground_truth[user_id]
            if not relevant:
                continue

            metrics = cls.compute_all(recs, relevant, k)
            all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Average across users
        return {
            "precision@k": np.mean([m.precision_at_k for m in all_metrics]),
            "recall@k": np.mean([m.recall_at_k for m in all_metrics]),
            "ndcg@k": np.mean([m.ndcg_at_k for m in all_metrics]),
            "hit_rate": np.mean([m.hit_rate for m in all_metrics]),
            "mrr": np.mean([m.mrr for m in all_metrics]),
            "map": np.mean([m.map_score for m in all_metrics]),
            "n_users": len(all_metrics),
        }


class CacheEvaluator:
    """
    Evaluate cache performance and quality tradeoffs.

    Compares cached vs. fresh recommendations.
    """

    def __init__(self, k: int = 10):
        self.k = k
        self._results: List[Dict] = []

    def add_comparison(
        self,
        user_id: int,
        cached_recs: List[int],
        fresh_recs: List[int],
        relevant_items: Set[int],
        cache_hit: bool,
        latency_cached_ms: float,
        latency_fresh_ms: float,
    ):
        """
        Add a single comparison result.

        Args:
            user_id: User ID
            cached_recs: Recommendations from cache
            fresh_recs: Fresh recommendations
            relevant_items: Ground truth relevant items
            cache_hit: Whether cache was hit
            latency_cached_ms: Latency with cache
            latency_fresh_ms: Latency without cache
        """
        # Compute quality metrics
        cached_ndcg = RecommendationMetrics.ndcg_at_k(
            cached_recs, relevant_items, self.k
        )
        fresh_ndcg = RecommendationMetrics.ndcg_at_k(
            fresh_recs, relevant_items, self.k
        )

        quality_ratio = cached_ndcg / fresh_ndcg if fresh_ndcg > 0 else 1.0

        self._results.append({
            "user_id": user_id,
            "cache_hit": cache_hit,
            "cached_ndcg": cached_ndcg,
            "fresh_ndcg": fresh_ndcg,
            "quality_ratio": quality_ratio,
            "latency_cached_ms": latency_cached_ms,
            "latency_fresh_ms": latency_fresh_ms,
        })

    def compute_metrics(self) -> CacheMetrics:
        """Compute aggregate cache metrics."""
        if not self._results:
            return CacheMetrics(
                hit_rate=0.0,
                miss_rate=1.0,
                latency_reduction=0.0,
                quality_loss=0.0,
                cost_savings=0.0,
            )

        hits = sum(1 for r in self._results if r["cache_hit"])
        total = len(self._results)

        hit_rate = hits / total
        miss_rate = 1 - hit_rate

        # Latency analysis (only for hits)
        hit_results = [r for r in self._results if r["cache_hit"]]
        if hit_results:
            avg_cached_latency = np.mean([r["latency_cached_ms"] for r in hit_results])
            avg_fresh_latency = np.mean([r["latency_fresh_ms"] for r in hit_results])
            latency_reduction = (
                (avg_fresh_latency - avg_cached_latency) / avg_fresh_latency
                if avg_fresh_latency > 0 else 0.0
            )
        else:
            latency_reduction = 0.0

        # Quality analysis (only for hits)
        if hit_results:
            quality_ratios = [r["quality_ratio"] for r in hit_results]
            quality_loss = 1 - np.mean(quality_ratios)
        else:
            quality_loss = 0.0

        # Cost savings (compute avoided due to cache hits)
        cost_savings = hit_rate

        return CacheMetrics(
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            latency_reduction=latency_reduction,
            quality_loss=quality_loss,
            cost_savings=cost_savings,
        )

    def get_detailed_results(self) -> List[Dict]:
        """Get detailed per-request results."""
        return self._results

    def get_quality_by_distance(
        self, distance_buckets: List[float] = [0.5, 1.0, 2.0]
    ) -> Dict[str, Dict]:
        """
        Analyze quality by user distance to cluster center.

        Requires results to have 'distance' field.
        """
        # This requires distance information to be stored
        # Placeholder for now
        return {}

    def clear(self):
        """Clear accumulated results."""
        self._results = []


def compute_ild(recommendations: List[int], item_embeddings: np.ndarray) -> float:
    """
    Compute Intra-List Diversity: average pairwise cosine distance among recommended items.

    Args:
        recommendations: List of recommended item IDs
        item_embeddings: Item embedding matrix (n_items, dim)

    Returns:
        Average pairwise cosine distance (0 = identical, 2 = opposite)
    """
    valid = [r for r in recommendations if r < len(item_embeddings)]
    if len(valid) < 2:
        return 0.0

    embs = item_embeddings[valid]
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs_norm = embs / norms

    # Cosine similarity matrix
    sim_matrix = embs_norm @ embs_norm.T
    n = len(valid)

    # Average pairwise cosine distance (upper triangle only)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += 1.0 - sim_matrix[i, j]
            count += 1

    return total_dist / count if count > 0 else 0.0


def compute_coverage(recommendations: Dict[int, List[int]], total_items: int) -> float:
    """
    Compute catalog coverage: fraction of unique items recommended across all users.

    Args:
        recommendations: Dict of user_id -> recommended items
        total_items: Total number of items in the catalog

    Returns:
        Fraction of catalog covered (0-1)
    """
    if total_items == 0:
        return 0.0

    all_items = set()
    for recs in recommendations.values():
        all_items.update(recs)

    return len(all_items) / total_items


def compute_tail_user_ndcg(
    user_results: Dict[int, float],
    interaction_counts: Dict[int, int],
    threshold: int = 5,
) -> float:
    """
    Compute average NDCG for tail (cold/sparse) users with <= threshold interactions.

    Args:
        user_results: Dict of user_id -> NDCG score
        interaction_counts: Dict of user_id -> number of training interactions
        threshold: Maximum interaction count to be considered a tail user

    Returns:
        Average NDCG for tail users
    """
    tail_ndcgs = []
    for user_id, ndcg in user_results.items():
        if interaction_counts.get(user_id, 0) <= threshold:
            tail_ndcgs.append(ndcg)

    return float(np.mean(tail_ndcgs)) if tail_ndcgs else 0.0


class SpeculativeMetrics:
    """Metrics specific to speculative recommendation serving."""

    @staticmethod
    def acceptance_rate(results: List) -> float:
        """Fraction of requests served from cache (accepted)."""
        if not results:
            return 0.0
        accepted = sum(1 for r in results if r.accepted)
        return accepted / len(results)

    @staticmethod
    def speedup_estimate(results: List, fresh_latency_ms: float = 0.0) -> float:
        """Empirical speedup from measured latencies.

        Args:
            results: List of SpeculativeResult objects with latency_ms.
            fresh_latency_ms: Measured mean latency of fresh computation.
                If 0 or not provided, measures from residual (non-accepted)
                results as proxy for fresh latency.
        """
        if not results:
            return 1.0

        spec_latencies = [r.latency_ms for r in results]
        avg_spec = float(np.mean(spec_latencies))

        if fresh_latency_ms > 0:
            baseline = fresh_latency_ms
        else:
            # Use residual (rejected -> fresh compute) latencies as proxy
            residual = [r.latency_ms for r in results if not r.accepted]
            if residual:
                baseline = float(np.mean(residual))
            else:
                # All accepted — measure from accepted latencies is not useful
                # Return ratio based on acceptance rate heuristic as fallback
                accept_rate = SpeculativeMetrics.acceptance_rate(results)
                return 1.0 / (1.0 - accept_rate + 1e-8) if accept_rate < 1.0 else float(len(results))

        return baseline / avg_spec if avg_spec > 0 else 1.0

    @staticmethod
    def multi_cluster_gain(results: List) -> float:
        """Fraction of accepted results where a non-nearest cluster was used.

        Measures how often the 2nd/3rd/... nearest cluster saved a reject.
        """
        accepted = [r for r in results if r.accepted]
        if not accepted:
            return 0.0
        non_nearest = sum(1 for r in accepted if r.accepted_cluster_rank > 0)
        return non_nearest / len(accepted)


def compute_list_similarity(list1: List[int], list2: List[int], k: int = 10) -> Dict[str, float]:
    """
    Compute similarity metrics between two recommendation lists.

    Returns:
        Dict with various similarity metrics
    """
    list1_k = list1[:k]
    list2_k = list2[:k]

    set1 = set(list1_k)
    set2 = set(list2_k)

    # Jaccard similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0.0

    # Overlap
    overlap = intersection / k if k > 0 else 0.0

    # Rank correlation (Kendall tau)
    common_items = set1 & set2
    if len(common_items) < 2:
        kendall_tau = 1.0  # Perfect if 0 or 1 common items
    else:
        # Compute concordant/discordant pairs
        items_list = list(common_items)
        rank1 = {item: list1_k.index(item) for item in items_list}
        rank2 = {item: list2_k.index(item) for item in items_list}

        concordant = 0
        discordant = 0
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                item_i, item_j = items_list[i], items_list[j]
                if (rank1[item_i] < rank1[item_j]) == (rank2[item_i] < rank2[item_j]):
                    concordant += 1
                else:
                    discordant += 1

        total_pairs = concordant + discordant
        kendall_tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0

    return {
        "jaccard": jaccard,
        "overlap": overlap,
        "kendall_tau": kendall_tau,
        "n_common": intersection,
    }
