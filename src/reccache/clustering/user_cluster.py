"""User clustering manager for recommendation caching."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading

from reccache.clustering.online_kmeans import OnlineKMeans


@dataclass
class UserClusterInfo:
    """Information about a user's cluster assignment."""

    cluster_id: int
    distance_to_center: float
    cluster_size: int
    embedding: np.ndarray


class UserClusterManager:
    """
    Manages user clustering for cache key generation.

    Features:
    - Maintains user embeddings based on behavior
    - Online clustering with incremental updates
    - Thread-safe operations
    """

    def __init__(
        self,
        n_clusters: int = 100,
        embedding_dim: int = 64,
        n_items: int = 1000,
        update_interval: int = 1000,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.n_items = n_items
        self.update_interval = update_interval

        # Online clustering
        self.clusterer = OnlineKMeans(
            n_clusters=n_clusters,
            dim=embedding_dim,
            learning_rate=learning_rate,
            seed=seed,
        )

        # User embeddings: user_id -> embedding
        self._user_embeddings: Dict[int, np.ndarray] = {}

        # User behavior history for embedding updates
        self._user_history: Dict[int, List[Tuple[int, float]]] = {}  # item, rating

        # Item embeddings for computing user embeddings
        self._item_embeddings: Optional[np.ndarray] = None

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._n_requests = 0
        self._pending_updates: List[np.ndarray] = []

        np.random.seed(seed)

    def set_item_embeddings(self, embeddings: np.ndarray):
        """
        Set item embeddings from a trained recommender model.

        Args:
            embeddings: Item embedding matrix (n_items, embedding_dim)
        """
        with self._lock:
            self._item_embeddings = embeddings.astype(np.float32)
            self.n_items = len(embeddings)

    def get_user_cluster(self, user_id: int) -> UserClusterInfo:
        """
        Get cluster assignment for a user.

        Args:
            user_id: User identifier

        Returns:
            UserClusterInfo with cluster assignment and distance
        """
        with self._lock:
            embedding = self._get_or_create_embedding(user_id)

            cluster_ids, distances = self.clusterer.predict_with_distance(embedding)
            cluster_id = int(cluster_ids[0])
            distance = float(distances[0])

            cluster_info = self.clusterer.get_cluster_info(cluster_id)

            return UserClusterInfo(
                cluster_id=cluster_id,
                distance_to_center=distance,
                cluster_size=cluster_info.count,
                embedding=embedding,
            )

    def update_user_behavior(
        self,
        user_id: int,
        item_id: int,
        rating: float,
        update_cluster: bool = True,
    ):
        """
        Update user embedding based on new interaction.

        Args:
            user_id: User identifier
            item_id: Item that was interacted with
            rating: Rating or implicit signal
            update_cluster: Whether to trigger cluster update
        """
        with self._lock:
            # Update history
            if user_id not in self._user_history:
                self._user_history[user_id] = []
            self._user_history[user_id].append((item_id, rating))

            # Keep only recent history
            max_history = 100
            if len(self._user_history[user_id]) > max_history:
                self._user_history[user_id] = self._user_history[user_id][-max_history:]

            # Update embedding
            embedding = self._compute_user_embedding(user_id)
            self._user_embeddings[user_id] = embedding

            # Schedule cluster update
            self._pending_updates.append(embedding)
            self._n_requests += 1

            if update_cluster and self._n_requests % self.update_interval == 0:
                self._update_clusters()

    def _get_or_create_embedding(self, user_id: int) -> np.ndarray:
        """Get existing embedding or create a new one."""
        if user_id in self._user_embeddings:
            return self._user_embeddings[user_id]

        # Create new embedding
        if user_id in self._user_history and len(self._user_history[user_id]) > 0:
            embedding = self._compute_user_embedding(user_id)
        else:
            # Random initialization for new users
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding /= np.linalg.norm(embedding) + 1e-8

        self._user_embeddings[user_id] = embedding
        return embedding

    def _compute_user_embedding(self, user_id: int) -> np.ndarray:
        """Compute user embedding from behavior history."""
        history = self._user_history.get(user_id, [])

        if not history or self._item_embeddings is None:
            # Return existing or random embedding
            if user_id in self._user_embeddings:
                return self._user_embeddings[user_id]
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            return embedding / (np.linalg.norm(embedding) + 1e-8)

        # Weighted average of item embeddings
        items = []
        weights = []
        for item_id, rating in history:
            if item_id < len(self._item_embeddings):
                items.append(item_id)
                # Use rating as weight (normalize to [0, 1])
                weights.append((rating - 1) / 4)  # Assuming 1-5 scale

        if not items:
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            return embedding / (np.linalg.norm(embedding) + 1e-8)

        weights = np.array(weights, dtype=np.float32)
        weights = np.exp(weights) / np.exp(weights).sum()  # Softmax

        # Weighted sum of item embeddings
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        for item_id, weight in zip(items, weights):
            embedding += weight * self._item_embeddings[item_id]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding

    def _update_clusters(self):
        """Update clusters with pending embeddings."""
        if not self._pending_updates:
            return

        update_batch = np.array(self._pending_updates, dtype=np.float32)
        self.clusterer.partial_fit(update_batch)
        self._pending_updates = []

    def initialize_from_interactions(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
    ):
        """
        Initialize user embeddings and clusters from historical data.

        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            ratings: Array of ratings
        """
        # Build user histories
        for user_id, item_id, rating in zip(user_ids, item_ids, ratings):
            if user_id not in self._user_history:
                self._user_history[user_id] = []
            self._user_history[user_id].append((int(item_id), float(rating)))

        # Compute all user embeddings
        embeddings = []
        for user_id in self._user_history.keys():
            embedding = self._compute_user_embedding(user_id)
            self._user_embeddings[user_id] = embedding
            embeddings.append(embedding)

        # Initialize clusters
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.clusterer.initialize(embeddings_array)

    def get_cluster_users(self, cluster_id: int) -> List[int]:
        """Get all users assigned to a cluster."""
        users = []
        with self._lock:
            for user_id, embedding in self._user_embeddings.items():
                cluster = self.clusterer.predict(embedding)[0]
                if cluster == cluster_id:
                    users.append(user_id)
        return users

    def get_similar_users(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find most similar users based on embedding distance.

        Args:
            user_id: Reference user
            top_k: Number of similar users to return

        Returns:
            List of (user_id, similarity) tuples
        """
        with self._lock:
            if user_id not in self._user_embeddings:
                return []

            ref_embedding = self._user_embeddings[user_id]

            similarities = []
            for other_id, other_embedding in self._user_embeddings.items():
                if other_id == user_id:
                    continue
                # Cosine similarity
                sim = np.dot(ref_embedding, other_embedding)
                similarities.append((other_id, float(sim)))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    def get_cluster_center(self, cluster_id: int) -> np.ndarray:
        """Get the center embedding of a cluster."""
        return self.clusterer.get_cluster_center(cluster_id)

    def get_statistics(self) -> Dict:
        """Get clustering statistics."""
        with self._lock:
            cluster_sizes = []
            for k in range(self.n_clusters):
                info = self.clusterer.get_cluster_info(k)
                cluster_sizes.append(info.count)

            return {
                "n_users": len(self._user_embeddings),
                "n_clusters": self.n_clusters,
                "cluster_sizes": cluster_sizes,
                "total_requests": self._n_requests,
                "pending_updates": len(self._pending_updates),
            }

    def save(self, path: str):
        """Save cluster state."""
        import pickle

        with self._lock:
            state = {
                "user_embeddings": self._user_embeddings,
                "user_history": self._user_history,
                "item_embeddings": self._item_embeddings,
                "n_requests": self._n_requests,
            }
            with open(f"{path}_state.pkl", "wb") as f:
                pickle.dump(state, f)

            self.clusterer.save(f"{path}_clusters.npz")

    def load(self, path: str):
        """Load cluster state."""
        import pickle

        with self._lock:
            with open(f"{path}_state.pkl", "rb") as f:
                state = pickle.load(f)

            self._user_embeddings = state["user_embeddings"]
            self._user_history = state["user_history"]
            self._item_embeddings = state["item_embeddings"]
            self._n_requests = state["n_requests"]

            self.clusterer.load(f"{path}_clusters.npz")
