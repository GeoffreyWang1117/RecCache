"""Tests for evaluation metrics."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reccache.evaluation.metrics import (
    RecommendationMetrics,
    CacheEvaluator,
    compute_list_similarity,
)


class TestRecommendationMetrics:
    """Tests for recommendation metrics."""

    def test_precision_at_k(self):
        """Test Precision@K."""
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        relevant = {1, 3, 5, 7, 9}

        # P@5 = 3/5 (items 1, 3, 5 are relevant)
        precision = RecommendationMetrics.precision_at_k(recommended, relevant, k=5)
        assert precision == 0.6

        # P@10 = 5/10
        precision = RecommendationMetrics.precision_at_k(recommended, relevant, k=10)
        assert precision == 0.5

    def test_recall_at_k(self):
        """Test Recall@K."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5, 7, 9}  # 5 relevant items

        # R@5 = 3/5 (found 1, 3, 5)
        recall = RecommendationMetrics.recall_at_k(recommended, relevant, k=5)
        assert recall == 0.6

    def test_ndcg_at_k(self):
        """Test NDCG@K."""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}

        # Perfect ranking
        ndcg = RecommendationMetrics.ndcg_at_k(recommended, relevant, k=3)
        assert ndcg == 1.0

        # Imperfect ranking
        recommended = [1, 4, 3]  # item 4 not relevant
        ndcg = RecommendationMetrics.ndcg_at_k(recommended, relevant, k=3)
        assert 0 < ndcg < 1.0

    def test_hit_rate(self):
        """Test Hit Rate@K."""
        relevant = {10, 20, 30}

        # Hit
        recommended = [1, 2, 10, 4, 5]
        hr = RecommendationMetrics.hit_rate(recommended, relevant, k=5)
        assert hr == 1.0

        # Miss
        recommended = [1, 2, 3, 4, 5]
        hr = RecommendationMetrics.hit_rate(recommended, relevant, k=5)
        assert hr == 0.0

    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        relevant = {3}

        recommended = [3, 1, 2]  # First position
        assert RecommendationMetrics.mrr(recommended, relevant) == 1.0

        recommended = [1, 3, 2]  # Second position
        assert RecommendationMetrics.mrr(recommended, relevant) == 0.5

        recommended = [1, 2, 3]  # Third position
        assert abs(RecommendationMetrics.mrr(recommended, relevant) - 1/3) < 1e-6

    def test_average_precision(self):
        """Test Average Precision."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}

        # AP = (1/1 + 2/3 + 3/5) / 3
        ap = RecommendationMetrics.average_precision(recommended, relevant)
        expected = (1.0 + 2/3 + 3/5) / 3
        assert abs(ap - expected) < 1e-6

    def test_compute_all(self):
        """Test computing all metrics at once."""
        recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        relevant = {1, 3, 5, 7, 9}

        metrics = RecommendationMetrics.compute_all(recommended, relevant, k=10)

        assert metrics.precision_at_k == 0.5
        assert metrics.recall_at_k == 1.0
        assert metrics.hit_rate == 1.0
        assert metrics.mrr == 1.0


class TestCacheEvaluator:
    """Tests for CacheEvaluator."""

    def test_cache_comparison(self):
        """Test comparing cached vs. fresh recommendations."""
        evaluator = CacheEvaluator(k=5)

        # Identical recommendations
        evaluator.add_comparison(
            user_id=1,
            cached_recs=[1, 2, 3, 4, 5],
            fresh_recs=[1, 2, 3, 4, 5],
            relevant_items={1, 2, 3},
            cache_hit=True,
            latency_cached_ms=5.0,
            latency_fresh_ms=50.0,
        )

        metrics = evaluator.compute_metrics()
        assert metrics.hit_rate == 1.0
        assert metrics.quality_loss < 0.01  # Near zero

    def test_quality_loss_calculation(self):
        """Test quality loss when cached differs from fresh."""
        evaluator = CacheEvaluator(k=5)

        # Different recommendations
        evaluator.add_comparison(
            user_id=1,
            cached_recs=[10, 11, 12, 13, 14],  # Different items
            fresh_recs=[1, 2, 3, 4, 5],
            relevant_items={1, 2, 3},
            cache_hit=True,
            latency_cached_ms=5.0,
            latency_fresh_ms=50.0,
        )

        metrics = evaluator.compute_metrics()
        assert metrics.quality_loss > 0  # Some quality lost


class TestListSimilarity:
    """Tests for list similarity computation."""

    def test_identical_lists(self):
        """Test identical lists have similarity 1.0."""
        list1 = [1, 2, 3, 4, 5]
        list2 = [1, 2, 3, 4, 5]

        sim = compute_list_similarity(list1, list2, k=5)
        assert sim["jaccard"] == 1.0
        assert sim["overlap"] == 1.0
        assert sim["kendall_tau"] == 1.0

    def test_completely_different_lists(self):
        """Test completely different lists have similarity 0."""
        list1 = [1, 2, 3, 4, 5]
        list2 = [6, 7, 8, 9, 10]

        sim = compute_list_similarity(list1, list2, k=5)
        assert sim["jaccard"] == 0.0
        assert sim["overlap"] == 0.0
        assert sim["n_common"] == 0

    def test_partial_overlap(self):
        """Test partially overlapping lists."""
        list1 = [1, 2, 3, 4, 5]
        list2 = [1, 2, 6, 7, 8]

        sim = compute_list_similarity(list1, list2, k=5)
        assert 0 < sim["jaccard"] < 1
        assert sim["overlap"] == 0.4  # 2 out of 5
        assert sim["n_common"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
