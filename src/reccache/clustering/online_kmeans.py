"""Online K-Means clustering with incremental updates."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClusterStats:
    """Statistics for a single cluster."""

    center: np.ndarray
    count: int
    sum_squared_dist: float


class OnlineKMeans:
    """
    Online K-Means clustering with incremental updates.

    Supports:
    - Mini-batch updates
    - Decay factor for forgetting old patterns
    - Cluster splitting/merging
    """

    def __init__(
        self,
        n_clusters: int,
        dim: int,
        learning_rate: float = 0.1,
        decay: float = 0.99,
        min_cluster_size: int = 10,
        seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.dim = dim
        self.learning_rate = learning_rate
        self.decay = decay
        self.min_cluster_size = min_cluster_size

        np.random.seed(seed)

        # Initialize cluster centers randomly
        self.centers = np.random.randn(n_clusters, dim).astype(np.float32)
        self.centers /= np.linalg.norm(self.centers, axis=1, keepdims=True) + 1e-8

        # Cluster statistics
        self.counts = np.zeros(n_clusters, dtype=np.int64)
        self.sum_squared_distances = np.zeros(n_clusters, dtype=np.float32)

        self._initialized = False
        self._n_updates = 0

    def initialize(self, data: np.ndarray):
        """
        Initialize clusters using k-means++ on initial data batch.

        Args:
            data: Initial data points (n_samples, dim)
        """
        n_samples = len(data)
        if n_samples < self.n_clusters:
            # Not enough samples, use random initialization
            self.centers[:n_samples] = data
            self._initialized = True
            return

        # K-means++ initialization
        centers = []

        # First center: random
        idx = np.random.randint(n_samples)
        centers.append(data[idx])

        # Remaining centers: probability proportional to squared distance
        for _ in range(1, self.n_clusters):
            distances = np.array([
                min(np.sum((x - c) ** 2) for c in centers)
                for x in data
            ])
            total = distances.sum()
            if total == 0 or np.isnan(total):
                # Degenerate case: all points identical, use random selection
                idx = np.random.randint(n_samples)
            else:
                probs = distances / total
                idx = np.random.choice(n_samples, p=probs)
            centers.append(data[idx])

        self.centers = np.array(centers, dtype=np.float32)
        self._initialized = True

        # Initial assignment
        assignments = self.predict(data)
        for i in range(self.n_clusters):
            mask = assignments == i
            self.counts[i] = mask.sum()

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Assign data points to nearest cluster.

        Args:
            data: Data points (n_samples, dim) or (dim,)

        Returns:
            Cluster assignments
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Compute distances to all centers
        distances = self._compute_distances(data)
        return np.argmin(distances, axis=1)

    def predict_with_distance(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign data points to nearest cluster and return distances.

        Args:
            data: Data points (n_samples, dim) or (dim,)

        Returns:
            Tuple of (assignments, distances to assigned centers)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        distances = self._compute_distances(data)
        assignments = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(data)), assignments]

        return assignments, min_distances

    def _compute_distances(self, data: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances to all centers."""
        # (n_samples, dim), (n_clusters, dim) -> (n_samples, n_clusters)
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        data_sq = np.sum(data ** 2, axis=1, keepdims=True)
        center_sq = np.sum(self.centers ** 2, axis=1, keepdims=True).T
        cross = data @ self.centers.T

        return data_sq + center_sq - 2 * cross

    def partial_fit(self, data: np.ndarray):
        """
        Update clusters with a mini-batch of data.

        Args:
            data: Mini-batch of data points (n_samples, dim)
        """
        if not self._initialized:
            self.initialize(data)
            return

        # Apply decay to old counts
        self.counts = (self.counts * self.decay).astype(np.int64)

        # Assign points to clusters
        assignments = self.predict(data)

        # Update each cluster
        for k in range(self.n_clusters):
            mask = assignments == k
            if not mask.any():
                continue

            cluster_data = data[mask]
            n_points = len(cluster_data)

            # Update count
            self.counts[k] += n_points

            # Update center using learning rate
            new_center = cluster_data.mean(axis=0)
            lr = self.learning_rate * n_points / (self.counts[k] + 1)
            self.centers[k] = (1 - lr) * self.centers[k] + lr * new_center

            # Update squared distance
            dists = np.sum((cluster_data - self.centers[k]) ** 2, axis=1)
            self.sum_squared_distances[k] = (
                self.decay * self.sum_squared_distances[k] + dists.sum()
            )

        self._n_updates += 1

        # Periodically rebalance clusters
        if self._n_updates % 100 == 0:
            self._rebalance_clusters()

    def _rebalance_clusters(self):
        """Split large clusters and merge/reinitialize small ones."""
        # Find empty or very small clusters
        small_clusters = np.where(self.counts < self.min_cluster_size)[0]

        # Find large clusters (by variance)
        variances = self.sum_squared_distances / (self.counts + 1)
        large_clusters = np.argsort(variances)[-len(small_clusters):]

        # Reinitialize small clusters near large cluster centers
        for small_idx, large_idx in zip(small_clusters, large_clusters):
            if variances[large_idx] < 0.1:
                continue  # Don't split tight clusters

            # Add small perturbation to large cluster center
            noise = np.random.randn(self.dim) * 0.1
            self.centers[small_idx] = self.centers[large_idx] + noise
            self.counts[small_idx] = self.counts[large_idx] // 2
            self.counts[large_idx] //= 2

    def get_cluster_center(self, cluster_id: int) -> np.ndarray:
        """Get the center of a specific cluster."""
        return self.centers[cluster_id]

    def get_cluster_info(self, cluster_id: int) -> ClusterStats:
        """Get statistics for a specific cluster."""
        return ClusterStats(
            center=self.centers[cluster_id],
            count=int(self.counts[cluster_id]),
            sum_squared_dist=float(self.sum_squared_distances[cluster_id]),
        )

    def get_all_centers(self) -> np.ndarray:
        """Get all cluster centers."""
        return self.centers.copy()

    def save(self, path: str):
        """Save cluster state to file."""
        np.savez(
            path,
            centers=self.centers,
            counts=self.counts,
            sum_squared_distances=self.sum_squared_distances,
            n_updates=self._n_updates,
            initialized=self._initialized,
        )

    def load(self, path: str):
        """Load cluster state from file."""
        data = np.load(path)
        self.centers = data["centers"]
        self.counts = data["counts"]
        self.sum_squared_distances = data["sum_squared_distances"]
        self._n_updates = int(data["n_updates"])
        self._initialized = bool(data["initialized"])
