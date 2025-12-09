"""Tests for recommendation models."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker


class TestMatrixFactorizationRecommender:
    """Tests for MF recommender."""

    def test_training(self):
        """Test model training."""
        n_users, n_items = 100, 50

        recommender = MatrixFactorizationRecommender(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=32,
        )

        # Create fake data
        n_samples = 1000
        user_ids = np.random.randint(0, n_users, size=n_samples)
        item_ids = np.random.randint(0, n_items, size=n_samples)
        ratings = np.random.uniform(1, 5, size=n_samples).astype(np.float32)

        stats = recommender.fit(
            user_ids=user_ids,
            item_ids=item_ids,
            ratings=ratings,
            epochs=5,
            verbose=False,
        )

        assert "final_loss" in stats
        assert stats["final_loss"] < stats["losses"][0]  # Loss decreased

    def test_recommendation(self):
        """Test recommendation generation."""
        n_users, n_items = 50, 30

        recommender = MatrixFactorizationRecommender(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=16,
        )

        # Train
        user_ids = np.random.randint(0, n_users, size=500)
        item_ids = np.random.randint(0, n_items, size=500)
        ratings = np.random.uniform(1, 5, size=500).astype(np.float32)

        recommender.fit(user_ids, item_ids, ratings, epochs=3, verbose=False)

        # Get recommendations
        recs = recommender.recommend(user_id=0, n=10)

        assert len(recs) == 10
        assert len(set(recs)) == 10  # All unique
        assert all(0 <= r < n_items for r in recs)

    def test_recommendation_with_exclusion(self):
        """Test excluding items from recommendations."""
        recommender = MatrixFactorizationRecommender(
            n_users=20,
            n_items=50,
            embedding_dim=16,
        )

        user_ids = np.random.randint(0, 20, size=200)
        item_ids = np.random.randint(0, 50, size=200)
        ratings = np.random.uniform(1, 5, size=200).astype(np.float32)
        recommender.fit(user_ids, item_ids, ratings, epochs=3, verbose=False)

        exclude = [0, 1, 2, 3, 4]
        recs = recommender.recommend(user_id=0, n=10, exclude_items=exclude)

        assert not any(r in exclude for r in recs)

    def test_embeddings(self):
        """Test embedding extraction."""
        recommender = MatrixFactorizationRecommender(
            n_users=20,
            n_items=30,
            embedding_dim=16,
        )

        user_emb = recommender.get_user_embedding(0)
        item_emb = recommender.get_item_embedding(0)

        assert user_emb.shape == (16,)
        assert item_emb.shape == (16,)

        all_user_embs = recommender.get_all_user_embeddings()
        all_item_embs = recommender.get_all_item_embeddings()

        assert all_user_embs.shape == (20, 16)
        assert all_item_embs.shape == (30, 16)


class TestQualityPredictor:
    """Tests for QualityPredictor."""

    def test_heuristic_prediction(self):
        """Test prediction before training (heuristic mode)."""
        predictor = QualityPredictor(quality_threshold=0.1)

        # Close to cluster center = high quality
        pred = predictor.predict(
            distance_to_center=0.1,
            cluster_size=50,
            context_match_score=1.0,
        )
        assert pred.quality_score > 0.8
        assert pred.use_cache

        # Far from cluster center = low quality
        pred = predictor.predict(
            distance_to_center=2.0,
            cluster_size=50,
            context_match_score=1.0,
        )
        assert pred.quality_score < 0.8

    def test_training_data_collection(self):
        """Test adding training samples."""
        predictor = QualityPredictor()

        # Add samples
        for _ in range(100):
            predictor.add_training_sample(
                distance_to_center=np.random.uniform(0, 2),
                cluster_size=np.random.randint(10, 100),
                context_match_score=np.random.uniform(0.5, 1.0),
                time_since_cache=np.random.uniform(0, 10),
                actual_quality=np.random.uniform(0.5, 1.0),
            )

        assert len(predictor._training_samples) == 100

    def test_training(self):
        """Test model training."""
        predictor = QualityPredictor()

        # Generate training data
        for _ in range(200):
            dist = np.random.uniform(0, 2)
            # Quality correlates with distance (inverse)
            quality = max(0, 1.0 - dist * 0.4 + np.random.normal(0, 0.1))

            predictor.add_training_sample(
                distance_to_center=dist,
                cluster_size=50,
                context_match_score=1.0,
                time_since_cache=0,
                actual_quality=quality,
            )

        result = predictor.train(epochs=50, min_samples=100)
        assert result["status"] == "success"
        assert predictor._trained


class TestLightweightReranker:
    """Tests for LightweightReranker."""

    def test_basic_reranking(self):
        """Test basic reranking functionality."""
        reranker = LightweightReranker()

        items = list(range(20))
        result = reranker.rerank(user_id=0, items=items)

        assert len(result.items) == 20
        assert set(result.items) == set(items)  # Same items, different order

    def test_reranking_with_history(self):
        """Test reranking incorporates user history."""
        reranker = LightweightReranker(history_weight=0.5)

        # Set item embeddings
        item_embs = np.random.randn(50, 16).astype(np.float32)
        reranker.set_item_embeddings(item_embs)

        # Set user history
        reranker.set_user_history(user_id=0, history=[1, 2, 3, 4, 5])

        items = list(range(10, 30))
        result = reranker.rerank(user_id=0, items=items)

        # Items similar to history should be boosted
        assert len(result.items) == 20

    def test_diversity_promotion(self):
        """Test diversity through category-based reranking."""
        reranker = LightweightReranker(diversity_weight=0.3)

        # Set categories (items 0-9 = cat 0, 10-19 = cat 1, etc.)
        categories = np.array([i // 10 for i in range(50)])
        reranker.set_item_categories(categories)

        # All items from same category
        items = list(range(10))  # All cat 0
        result = reranker.rerank(user_id=0, items=items)

        # Should still work
        assert len(result.items) == 10

    def test_recency_penalty(self):
        """Test penalty for recently seen items."""
        reranker = LightweightReranker(recency_weight=0.3)

        # Set history with recent items
        reranker.set_user_history(user_id=0, history=[5, 6, 7, 8, 9])

        # Include recently seen items
        items = [5, 6, 7, 15, 16, 17, 25, 26, 27, 35]
        result = reranker.rerank(user_id=0, items=items)

        # Recently seen should be pushed down
        # Items 15, 16, 17, 25, 26, 27, 35 should rank higher
        assert len(result.items) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
