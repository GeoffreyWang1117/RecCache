"""Baseline recommendation models for comparison."""

import numpy as np
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm


class BaseRecommender(ABC):
    """Abstract base class for recommenders."""

    @abstractmethod
    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray, **kwargs) -> Dict:
        """Train the model."""
        pass

    @abstractmethod
    def recommend(self, user_id: int, n: int = 20, exclude_items: Optional[List[int]] = None) -> List[int]:
        """Get top-N recommendations for a user."""
        pass

    def recommend_batch(self, user_ids: List[int], n: int = 20) -> List[List[int]]:
        """Get recommendations for multiple users."""
        return [self.recommend(uid, n) for uid in user_ids]

    def get_all_item_embeddings(self) -> np.ndarray:
        """Get item embeddings if available."""
        raise NotImplementedError


class MostPopularRecommender(BaseRecommender):
    """
    Most Popular items recommender.

    Simple baseline that recommends most frequently interacted items.
    """

    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self.item_popularity: np.ndarray = None
        self.top_items: List[int] = []

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray, **kwargs) -> Dict:
        """Count item interactions."""
        item_counts = np.zeros(self.n_items)
        for item_id in item_ids:
            if item_id < self.n_items:
                item_counts[item_id] += 1

        self.item_popularity = item_counts / item_counts.sum()
        self.top_items = np.argsort(-item_counts).tolist()

        return {"n_interactions": len(user_ids)}

    def recommend(self, user_id: int, n: int = 20, exclude_items: Optional[List[int]] = None) -> List[int]:
        """Return most popular items."""
        exclude = set(exclude_items) if exclude_items else set()
        result = []
        for item in self.top_items:
            if item not in exclude:
                result.append(item)
                if len(result) >= n:
                    break
        return result

    def get_popularity_scores(self) -> np.ndarray:
        """Get item popularity scores."""
        return self.item_popularity


class ItemKNNRecommender(BaseRecommender):
    """
    Item-based K-Nearest Neighbors recommender.

    Uses cosine similarity between items.
    """

    def __init__(self, n_users: int, n_items: int, k: int = 50):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.user_item_matrix: csr_matrix = None
        self.item_similarity: np.ndarray = None
        self.user_history: Dict[int, List[int]] = defaultdict(list)

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray, **kwargs) -> Dict:
        """Build item-item similarity matrix."""
        verbose = kwargs.get("verbose", True)

        # Build user-item matrix
        self.user_item_matrix = csr_matrix(
            (ratings, (user_ids, item_ids)),
            shape=(self.n_users, self.n_items),
        )

        # Build user history
        for user_id, item_id in zip(user_ids, item_ids):
            self.user_history[int(user_id)].append(int(item_id))

        # Compute item-item similarity (cosine)
        if verbose:
            print("Computing item-item similarity...")

        item_matrix = self.user_item_matrix.T.tocsr()

        # Normalize
        norms = np.sqrt(np.array(item_matrix.power(2).sum(axis=1)).flatten())
        norms[norms == 0] = 1

        # Compute similarity in batches to save memory
        batch_size = 500
        n_batches = (self.n_items + batch_size - 1) // batch_size

        self.item_similarity = np.zeros((self.n_items, self.k), dtype=np.float32)
        self.item_neighbors = np.zeros((self.n_items, self.k), dtype=np.int32)

        iterator = tqdm(range(n_batches), desc="Building kNN") if verbose else range(n_batches)

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.n_items)

            batch_items = item_matrix[start_idx:end_idx]
            batch_norms = norms[start_idx:end_idx]

            # Compute similarity with all items
            sim = batch_items.dot(item_matrix.T).toarray()
            sim = sim / (batch_norms[:, None] * norms[None, :] + 1e-8)

            # Get top-k neighbors (excluding self)
            for i, row in enumerate(sim):
                item_id = start_idx + i
                row[item_id] = -1  # Exclude self
                top_k_idx = np.argsort(-row)[:self.k]
                self.item_neighbors[item_id] = top_k_idx
                self.item_similarity[item_id] = row[top_k_idx]

        return {"n_items": self.n_items, "k": self.k}

    def recommend(self, user_id: int, n: int = 20, exclude_items: Optional[List[int]] = None) -> List[int]:
        """Recommend items based on user history and item similarity."""
        if user_id not in self.user_history or not self.user_history[user_id]:
            # Cold start: return popular items
            return list(range(n))

        user_items = set(self.user_history[user_id])
        exclude = set(exclude_items) if exclude_items else set()
        exclude = exclude | user_items

        # Score all items based on similarity to user's history
        scores = np.zeros(self.n_items)
        for item_id in user_items:
            if item_id < self.n_items:
                neighbors = self.item_neighbors[item_id]
                sims = self.item_similarity[item_id]
                for neighbor, sim in zip(neighbors, sims):
                    if neighbor not in exclude:
                        scores[neighbor] += sim

        # Get top-N
        for item in exclude:
            if item < self.n_items:
                scores[item] = -np.inf

        top_items = np.argsort(-scores)[:n]
        return top_items.tolist()


class UserKNNRecommender(BaseRecommender):
    """
    User-based K-Nearest Neighbors recommender.

    Uses cosine similarity between users.
    """

    def __init__(self, n_users: int, n_items: int, k: int = 50):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.user_item_matrix: csr_matrix = None
        self.user_similarity: np.ndarray = None
        self.user_neighbors: np.ndarray = None

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray, **kwargs) -> Dict:
        """Build user-user similarity matrix."""
        verbose = kwargs.get("verbose", True)

        # Build user-item matrix
        self.user_item_matrix = csr_matrix(
            (ratings, (user_ids, item_ids)),
            shape=(self.n_users, self.n_items),
        )

        if verbose:
            print("Computing user-user similarity...")

        # Normalize
        norms = np.sqrt(np.array(self.user_item_matrix.power(2).sum(axis=1)).flatten())
        norms[norms == 0] = 1

        # Compute similarity in batches
        batch_size = 500
        n_batches = (self.n_users + batch_size - 1) // batch_size

        self.user_similarity = np.zeros((self.n_users, self.k), dtype=np.float32)
        self.user_neighbors = np.zeros((self.n_users, self.k), dtype=np.int32)

        iterator = tqdm(range(n_batches), desc="Building kNN") if verbose else range(n_batches)

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.n_users)

            batch_users = self.user_item_matrix[start_idx:end_idx]
            batch_norms = norms[start_idx:end_idx]

            sim = batch_users.dot(self.user_item_matrix.T).toarray()
            sim = sim / (batch_norms[:, None] * norms[None, :] + 1e-8)

            for i, row in enumerate(sim):
                user_id = start_idx + i
                row[user_id] = -1
                top_k_idx = np.argsort(-row)[:self.k]
                self.user_neighbors[user_id] = top_k_idx
                self.user_similarity[user_id] = row[top_k_idx]

        return {"n_users": self.n_users, "k": self.k}

    def recommend(self, user_id: int, n: int = 20, exclude_items: Optional[List[int]] = None) -> List[int]:
        """Recommend items based on similar users."""
        if user_id >= self.n_users:
            return list(range(n))

        neighbors = self.user_neighbors[user_id]
        sims = self.user_similarity[user_id]

        # Weighted average of neighbor ratings
        scores = np.zeros(self.n_items)
        weights = np.zeros(self.n_items)

        for neighbor, sim in zip(neighbors, sims):
            if sim > 0:
                neighbor_ratings = self.user_item_matrix[neighbor].toarray().flatten()
                scores += sim * neighbor_ratings
                weights += sim * (neighbor_ratings > 0)

        weights[weights == 0] = 1
        scores = scores / weights

        # Exclude items
        user_items = set(self.user_item_matrix[user_id].indices)
        exclude = set(exclude_items) if exclude_items else set()
        exclude = exclude | user_items

        for item in exclude:
            if item < self.n_items:
                scores[item] = -np.inf

        top_items = np.argsort(-scores)[:n]
        return top_items.tolist()


class BPRDataset(Dataset):
    """Dataset for BPR training with negative sampling."""

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, n_items: int, n_negatives: int = 1):
        self.users = user_ids
        self.pos_items = item_ids
        self.n_items = n_items
        self.n_negatives = n_negatives

        # Build user-item set for negative sampling
        self.user_items = defaultdict(set)
        for u, i in zip(user_ids, item_ids):
            self.user_items[int(u)].add(int(i))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        # Sample negative item
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.user_items[int(user)]:
            neg_item = np.random.randint(0, self.n_items)

        return user, pos_item, neg_item


class BPRMF(BaseRecommender):
    """
    Bayesian Personalized Ranking with Matrix Factorization.

    Optimizes AUC using pairwise ranking loss.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        device: str = "cpu",
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.device = device

        self.user_embeddings = nn.Embedding(n_users, embedding_dim).to(device)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim).to(device)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        self._trained = False

    def _score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Compute user-item scores."""
        user_emb = self.user_embeddings(users)
        item_emb = self.item_embeddings(items)
        return (user_emb * item_emb).sum(dim=-1)

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
        **kwargs
    ) -> Dict:
        """Train with BPR loss."""
        dataset = BPRDataset(user_ids, item_ids, self.n_items)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        params = list(self.user_embeddings.parameters()) + list(self.item_embeddings.parameters())
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        losses = []
        iterator = tqdm(range(epochs), desc="Training BPR-MF") if verbose else range(epochs)

        for epoch in iterator:
            epoch_loss = 0.0
            for users, pos_items, neg_items in loader:
                users = users.to(self.device).long()
                pos_items = pos_items.to(self.device).long()
                neg_items = neg_items.to(self.device).long()

                optimizer.zero_grad()

                pos_scores = self._score(users, pos_items)
                neg_scores = self._score(users, neg_items)

                # BPR loss: -log(sigmoid(pos - neg))
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if verbose:
                iterator.set_postfix({"loss": f"{avg_loss:.4f}"})

        self._trained = True
        return {"final_loss": losses[-1], "losses": losses}

    def recommend(self, user_id: int, n: int = 20, exclude_items: Optional[List[int]] = None) -> List[int]:
        """Get top-N recommendations."""
        with torch.no_grad():
            user_t = torch.tensor([user_id], dtype=torch.long).to(self.device)
            user_emb = self.user_embeddings(user_t)

            all_items = torch.arange(self.n_items, dtype=torch.long).to(self.device)
            item_emb = self.item_embeddings(all_items)

            scores = (user_emb @ item_emb.T).squeeze().cpu().numpy()

        if exclude_items:
            for item in exclude_items:
                if item < len(scores):
                    scores[item] = -np.inf

        return np.argsort(-scores)[:n].tolist()

    def get_all_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings."""
        with torch.no_grad():
            return self.item_embeddings.weight.cpu().numpy()


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.

    He et al., SIGIR 2020
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.adj_matrix = None

    def build_graph(self, user_ids: np.ndarray, item_ids: np.ndarray):
        """Build normalized adjacency matrix for graph convolution."""
        n_users = self.n_users
        n_items = self.n_items
        n_nodes = n_users + n_items

        # Build adjacency matrix
        user_np = np.array(user_ids)
        item_np = np.array(item_ids) + n_users  # Shift item IDs

        rows = np.concatenate([user_np, item_np])
        cols = np.concatenate([item_np, user_np])
        data = np.ones(len(rows))

        adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Normalize: D^(-0.5) * A * D^(-0.5)
        degrees = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degrees + 1e-8, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        d_mat = csr_matrix((d_inv_sqrt, (range(n_nodes), range(n_nodes))), shape=(n_nodes, n_nodes))
        norm_adj = d_mat @ adj @ d_mat

        # Convert to torch sparse tensor
        coo = norm_adj.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform graph convolution and return user/item embeddings."""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        embeddings_list = [all_embeddings]

        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Average embeddings from all layers
        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        user_emb = final_embeddings[:self.n_users]
        item_emb = final_embeddings[self.n_users:]

        return user_emb, item_emb


class LightGCNRecommender(BaseRecommender):
    """LightGCN recommender wrapper."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        device: str = "cpu",
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.device = device

        self.model = LightGCN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
        ).to(device)

        self._trained = False
        self._cached_user_emb = None
        self._cached_item_emb = None

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
        **kwargs
    ) -> Dict:
        """Train LightGCN with BPR loss."""
        # Build graph
        self.model.build_graph(user_ids, item_ids)
        self.model.adj_matrix = self.model.adj_matrix.to(self.device)

        # Build dataset
        dataset = BPRDataset(user_ids, item_ids, self.n_items)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        losses = []
        iterator = tqdm(range(epochs), desc="Training LightGCN") if verbose else range(epochs)

        for epoch in iterator:
            self.model.train()
            epoch_loss = 0.0

            for users, pos_items, neg_items in loader:
                users = users.to(self.device).long()
                pos_items = pos_items.to(self.device).long()
                neg_items = neg_items.to(self.device).long()

                optimizer.zero_grad()

                user_emb, item_emb = self.model()

                user_e = user_emb[users]
                pos_e = item_emb[pos_items]
                neg_e = item_emb[neg_items]

                pos_scores = (user_e * pos_e).sum(dim=-1)
                neg_scores = (user_e * neg_e).sum(dim=-1)

                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

                # L2 regularization on embeddings
                reg_loss = (
                    self.model.user_embedding(users).pow(2).sum() +
                    self.model.item_embedding(pos_items).pow(2).sum() +
                    self.model.item_embedding(neg_items).pow(2).sum()
                ) / batch_size * weight_decay

                total_loss = loss + reg_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if verbose:
                iterator.set_postfix({"loss": f"{avg_loss:.4f}"})

        self._trained = True

        # Cache embeddings for inference
        self.model.eval()
        with torch.no_grad():
            self._cached_user_emb, self._cached_item_emb = self.model()
            self._cached_user_emb = self._cached_user_emb.cpu().numpy()
            self._cached_item_emb = self._cached_item_emb.cpu().numpy()

        return {"final_loss": losses[-1], "losses": losses}

    def recommend(self, user_id: int, n: int = 20, exclude_items: Optional[List[int]] = None) -> List[int]:
        """Get top-N recommendations."""
        if self._cached_user_emb is None:
            raise RuntimeError("Model not trained yet")

        user_emb = self._cached_user_emb[user_id]
        scores = self._cached_item_emb @ user_emb

        if exclude_items:
            for item in exclude_items:
                if item < len(scores):
                    scores[item] = -np.inf

        return np.argsort(-scores)[:n].tolist()

    def get_all_item_embeddings(self) -> np.ndarray:
        """Get item embeddings after graph convolution."""
        if self._cached_item_emb is None:
            with torch.no_grad():
                _, item_emb = self.model()
            return item_emb.cpu().numpy()
        return self._cached_item_emb


class RandomRecommender(BaseRecommender):
    """Random recommendation baseline."""

    def __init__(self, n_users: int, n_items: int, seed: int = 42):
        self.n_users = n_users
        self.n_items = n_items
        self.rng = np.random.RandomState(seed)

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray, **kwargs) -> Dict:
        """No training needed."""
        return {}

    def recommend(self, user_id: int, n: int = 20, exclude_items: Optional[List[int]] = None) -> List[int]:
        """Return random items."""
        exclude = set(exclude_items) if exclude_items else set()
        candidates = [i for i in range(self.n_items) if i not in exclude]
        return self.rng.choice(candidates, min(n, len(candidates)), replace=False).tolist()


# Factory function
def create_recommender(
    model_name: str,
    n_users: int,
    n_items: int,
    embedding_dim: int = 64,
    device: str = "cpu",
    **kwargs
) -> BaseRecommender:
    """
    Create a recommender model.

    Args:
        model_name: Model name (mf, ncf, bpr, lightgcn, itemknn, userknn, pop, random)
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Embedding dimension
        device: Device for PyTorch models
        **kwargs: Additional arguments

    Returns:
        Recommender instance
    """
    model_name = model_name.lower()

    # Import here to avoid circular imports
    from reccache.models.recommender import MatrixFactorizationRecommender, NCFRecommender

    models = {
        "mf": lambda: MatrixFactorizationRecommender(n_users, n_items, embedding_dim, device),
        "ncf": lambda: NCFRecommender(n_users, n_items, embedding_dim, device=device),
        "bpr": lambda: BPRMF(n_users, n_items, embedding_dim, device),
        "lightgcn": lambda: LightGCNRecommender(n_users, n_items, embedding_dim, device=device, **kwargs),
        "itemknn": lambda: ItemKNNRecommender(n_users, n_items, k=kwargs.get("k", 50)),
        "userknn": lambda: UserKNNRecommender(n_users, n_items, k=kwargs.get("k", 50)),
        "pop": lambda: MostPopularRecommender(n_users, n_items),
        "popular": lambda: MostPopularRecommender(n_users, n_items),
        "random": lambda: RandomRecommender(n_users, n_items),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name]()


# Utility for comparing recommenders
class RecommenderComparator:
    """Compare multiple recommenders on the same data."""

    def __init__(
        self,
        model_names: List[str],
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        device: str = "cpu",
    ):
        self.models = {
            name: create_recommender(name, n_users, n_items, embedding_dim, device)
            for name in model_names
        }

    def fit_all(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        **kwargs
    ) -> Dict[str, Dict]:
        """Train all models."""
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            train_result = model.fit(user_ids, item_ids, ratings, **kwargs)
            train_time = time.time() - start_time
            results[name] = {
                "train_result": train_result,
                "train_time": train_time,
            }
        return results

    def evaluate_all(
        self,
        test_users: List[int],
        ground_truth: Dict[int, set],
        k: int = 20,
    ) -> Dict[str, Dict]:
        """Evaluate all models."""
        from reccache.evaluation.metrics import RecommendationMetrics

        results = {}
        for name, model in self.models.items():
            print(f"Evaluating {name}...")

            recommendations = {}
            start_time = time.time()
            for user_id in test_users:
                if user_id in ground_truth:
                    recommendations[user_id] = model.recommend(user_id, n=k)
            inference_time = time.time() - start_time

            metrics = RecommendationMetrics.evaluate_recommendations(
                recommendations, ground_truth, k=k
            )

            results[name] = {
                **metrics,
                "inference_time": inference_time,
                "avg_latency_ms": (inference_time / len(test_users)) * 1000,
            }

        return results
