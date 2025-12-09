"""Tests for clustering components."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reccache.clustering.online_kmeans import OnlineKMeans
from reccache.clustering.user_cluster import UserClusterManager


class TestOnlineKMeans:
    """Tests for OnlineKMeans."""

    def test_initialization(self):
        """Test cluster initialization."""
        kmeans = OnlineKMeans(n_clusters=5, dim=32)

        data = np.random.randn(100, 32).astype(np.float32)
        kmeans.initialize(data)

        # Should have 5 centers
        centers = kmeans.get_all_centers()
        assert centers.shape == (5, 32)

    def test_prediction(self):
        """Test cluster assignment."""
        kmeans = OnlineKMeans(n_clusters=3, dim=16)

        # Create clearly separated clusters
        data = np.vstack([
            np.random.randn(50, 16) + np.array([5] * 16),
            np.random.randn(50, 16) + np.array([-5] * 16),
            np.random.randn(50, 16) + np.array([0] * 16),
        ]).astype(np.float32)

        kmeans.initialize(data)

        # Predict on new points
        test_point = np.array([5] * 16, dtype=np.float32)
        assignment = kmeans.predict(test_point)
        assert len(assignment) == 1
        assert 0 <= assignment[0] < 3

    def test_predict_with_distance(self):
        """Test prediction with distance computation."""
        kmeans = OnlineKMeans(n_clusters=3, dim=8)
        data = np.random.randn(50, 8).astype(np.float32)
        kmeans.initialize(data)

        test_points = np.random.randn(10, 8).astype(np.float32)
        assignments, distances = kmeans.predict_with_distance(test_points)

        assert len(assignments) == 10
        assert len(distances) == 10
        assert all(d >= 0 for d in distances)

    def test_partial_fit(self):
        """Test incremental update."""
        kmeans = OnlineKMeans(n_clusters=5, dim=16)

        # Initial batch
        data1 = np.random.randn(100, 16).astype(np.float32)
        kmeans.initialize(data1)

        centers_before = kmeans.get_all_centers().copy()

        # Update with new batch
        data2 = np.random.randn(50, 16).astype(np.float32)
        kmeans.partial_fit(data2)

        centers_after = kmeans.get_all_centers()

        # Centers should have changed
        assert not np.allclose(centers_before, centers_after)

    def test_cluster_info(self):
        """Test cluster information retrieval."""
        kmeans = OnlineKMeans(n_clusters=3, dim=8)
        data = np.random.randn(100, 8).astype(np.float32)
        kmeans.initialize(data)

        info = kmeans.get_cluster_info(0)
        assert info.center is not None
        assert info.count >= 0


class TestUserClusterManager:
    """Tests for UserClusterManager."""

    def test_user_cluster_assignment(self):
        """Test user cluster assignment."""
        manager = UserClusterManager(
            n_clusters=10,
            embedding_dim=32,
            n_items=100,
        )

        # Set item embeddings
        item_embs = np.random.randn(100, 32).astype(np.float32)
        manager.set_item_embeddings(item_embs)

        # Simulate user history
        manager.update_user_behavior(user_id=1, item_id=5, rating=5.0)
        manager.update_user_behavior(user_id=1, item_id=10, rating=4.0)

        # Get cluster
        cluster_info = manager.get_user_cluster(user_id=1)
        assert 0 <= cluster_info.cluster_id < 10
        assert cluster_info.distance_to_center >= 0

    def test_initialize_from_interactions(self):
        """Test initialization from interaction data."""
        manager = UserClusterManager(
            n_clusters=5,
            embedding_dim=16,
            n_items=50,
        )

        item_embs = np.random.randn(50, 16).astype(np.float32)
        manager.set_item_embeddings(item_embs)

        # Create fake interactions
        n_interactions = 500
        user_ids = np.random.randint(0, 20, size=n_interactions)
        item_ids = np.random.randint(0, 50, size=n_interactions)
        ratings = np.random.uniform(1, 5, size=n_interactions)

        manager.initialize_from_interactions(user_ids, item_ids, ratings)

        stats = manager.get_statistics()
        assert stats["n_users"] > 0
        assert stats["n_clusters"] == 5

    def test_similar_users(self):
        """Test finding similar users."""
        manager = UserClusterManager(
            n_clusters=5,
            embedding_dim=16,
            n_items=50,
        )

        item_embs = np.random.randn(50, 16).astype(np.float32)
        manager.set_item_embeddings(item_embs)

        # Create users with similar behavior
        for user_id in range(10):
            for item_id in range(5):
                manager.update_user_behavior(
                    user_id=user_id,
                    item_id=item_id + (user_id % 2) * 10,  # Two groups
                    rating=4.0,
                    update_cluster=False,
                )

        similar = manager.get_similar_users(user_id=0, top_k=3)
        assert len(similar) <= 3
        # User 0 should be similar to other even users


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
