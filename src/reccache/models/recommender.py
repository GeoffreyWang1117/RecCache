"""Recommendation models for RecCache."""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class RecommendationResult:
    """Result from recommendation model."""

    item_ids: List[int]
    scores: np.ndarray
    latency_ms: float


class MatrixFactorizationRecommender:
    """
    Matrix Factorization recommender using PyTorch.

    Simple but effective baseline for collaborative filtering.
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

        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, embedding_dim).to(device)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim).to(device)

        # Biases
        self.user_bias = nn.Embedding(n_users, 1).to(device)
        self.item_bias = nn.Embedding(n_items, 1).to(device)
        self.global_bias = nn.Parameter(torch.zeros(1)).to(device)

        # Initialize
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self._trained = False

    def _predict_batch(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Predict ratings for user-item pairs."""
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()

        # Dot product + biases
        pred = (user_emb * item_emb).sum(dim=1)
        pred = pred + user_b + item_b + self.global_bias

        return pred

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the recommender.

        Args:
            user_ids: User IDs
            item_ids: Item IDs
            ratings: Ratings
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization

        Returns:
            Training statistics
        """
        # Prepare data
        users_t = torch.tensor(user_ids, dtype=torch.long)
        items_t = torch.tensor(item_ids, dtype=torch.long)
        ratings_t = torch.tensor(ratings, dtype=torch.float32)

        dataset = TensorDataset(users_t, items_t, ratings_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        params = list(self.user_embeddings.parameters()) + \
                 list(self.item_embeddings.parameters()) + \
                 list(self.user_bias.parameters()) + \
                 list(self.item_bias.parameters()) + \
                 [self.global_bias]
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        criterion = nn.MSELoss()

        losses = []
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            epoch_loss = 0.0
            for batch_users, batch_items, batch_ratings in loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)

                optimizer.zero_grad()
                pred = self._predict_batch(batch_users, batch_items)
                loss = criterion(pred, batch_ratings)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if verbose:
                iterator.set_postfix({"loss": f"{avg_loss:.4f}"})

        self._trained = True

        return {
            "final_loss": losses[-1],
            "losses": losses,
            "n_samples": len(user_ids),
        }

    def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_items: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Get top-N recommendations for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_items: Items to exclude (e.g., already interacted)

        Returns:
            List of recommended item IDs
        """
        start_time = time.time()

        with torch.no_grad():
            user_t = torch.tensor([user_id], dtype=torch.long).to(self.device)
            user_emb = self.user_embeddings(user_t)  # (1, dim)
            user_b = self.user_bias(user_t)  # (1, 1)

            # Score all items
            all_items = torch.arange(self.n_items, dtype=torch.long).to(self.device)
            item_emb = self.item_embeddings(all_items)  # (n_items, dim)
            item_b = self.item_bias(all_items).squeeze()  # (n_items,)

            scores = (user_emb @ item_emb.T).squeeze()  # (n_items,)
            scores = scores + user_b.squeeze() + item_b + self.global_bias

            scores = scores.cpu().numpy()

        # Exclude items
        if exclude_items:
            for item in exclude_items:
                if item < len(scores):
                    scores[item] = -np.inf

        # Get top-N
        top_indices = np.argsort(-scores)[:n]

        return top_indices.tolist()

    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = 20,
    ) -> List[List[int]]:
        """Get recommendations for multiple users."""
        results = []
        for user_id in user_ids:
            recs = self.recommend(user_id, n)
            results.append(recs)
        return results

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get embedding for a user."""
        with torch.no_grad():
            user_t = torch.tensor([user_id], dtype=torch.long).to(self.device)
            emb = self.user_embeddings(user_t).cpu().numpy()
        return emb.squeeze()

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get embedding for an item."""
        with torch.no_grad():
            item_t = torch.tensor([item_id], dtype=torch.long).to(self.device)
            emb = self.item_embeddings(item_t).cpu().numpy()
        return emb.squeeze()

    def get_all_user_embeddings(self) -> np.ndarray:
        """Get all user embeddings."""
        with torch.no_grad():
            return self.user_embeddings.weight.cpu().numpy()

    def get_all_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings."""
        with torch.no_grad():
            return self.item_embeddings.weight.cpu().numpy()

    def save(self, path: str):
        """Save model to file."""
        state = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim,
            "user_embeddings": self.user_embeddings.state_dict(),
            "item_embeddings": self.item_embeddings.state_dict(),
            "user_bias": self.user_bias.state_dict(),
            "item_bias": self.item_bias.state_dict(),
            "global_bias": self.global_bias.data,
            "trained": self._trained,
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load model from file."""
        state = torch.load(path, map_location=self.device)
        self.user_embeddings.load_state_dict(state["user_embeddings"])
        self.item_embeddings.load_state_dict(state["item_embeddings"])
        self.user_bias.load_state_dict(state["user_bias"])
        self.item_bias.load_state_dict(state["item_bias"])
        self.global_bias.data = state["global_bias"]
        self._trained = state["trained"]


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model.

    Combines GMF (Generalized Matrix Factorization) and MLP paths.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        mlp_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        for dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        final_dim = embedding_dim + mlp_dims[-1]
        self.output_layer = nn.Linear(final_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        # GMF path
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_output = gmf_user * gmf_item

        # MLP path
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine and predict
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.output_layer(combined).squeeze(-1)

        return prediction


class NCFRecommender:
    """NCF recommender wrapper."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        mlp_dims: List[int] = [128, 64, 32],
        device: str = "cpu",
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.device = device

        self.model = NeuralCollaborativeFiltering(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            mlp_dims=mlp_dims,
        ).to(device)

        self._trained = False

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.001,
        verbose: bool = True,
    ) -> Dict:
        """Train the NCF model."""
        users_t = torch.tensor(user_ids, dtype=torch.long)
        items_t = torch.tensor(item_ids, dtype=torch.long)
        ratings_t = torch.tensor(ratings, dtype=torch.float32)

        dataset = TensorDataset(users_t, items_t, ratings_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        iterator = tqdm(range(epochs), desc="Training NCF") if verbose else range(epochs)

        self.model.train()
        for epoch in iterator:
            epoch_loss = 0.0
            for batch_users, batch_items, batch_ratings in loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_users, batch_items)
                loss = criterion(pred, batch_ratings)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

            if verbose:
                iterator.set_postfix({"loss": f"{avg_loss:.4f}"})

        self._trained = True
        return {"final_loss": losses[-1], "losses": losses}

    def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_items: Optional[List[int]] = None,
    ) -> List[int]:
        """Get top-N recommendations."""
        self.model.eval()

        with torch.no_grad():
            user_t = torch.tensor([user_id] * self.n_items, dtype=torch.long).to(self.device)
            items_t = torch.arange(self.n_items, dtype=torch.long).to(self.device)

            scores = self.model(user_t, items_t).cpu().numpy()

        if exclude_items:
            for item in exclude_items:
                if item < len(scores):
                    scores[item] = -np.inf

        top_indices = np.argsort(-scores)[:n]
        return top_indices.tolist()

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get combined user embedding (GMF + MLP)."""
        with torch.no_grad():
            user_t = torch.tensor([user_id], dtype=torch.long).to(self.device)
            gmf_emb = self.model.gmf_user_emb(user_t)
            mlp_emb = self.model.mlp_user_emb(user_t)
            combined = torch.cat([gmf_emb, mlp_emb], dim=-1)
        return combined.cpu().numpy().squeeze()

    def get_all_item_embeddings(self) -> np.ndarray:
        """Get GMF item embeddings (for clustering/similarity)."""
        with torch.no_grad():
            return self.model.gmf_item_emb.weight.cpu().numpy()

    def save(self, path: str):
        """Save model."""
        torch.save({
            "model_state": self.model.state_dict(),
            "n_users": self.n_users,
            "n_items": self.n_items,
            "trained": self._trained,
        }, path)

    def load(self, path: str):
        """Load model."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self._trained = state["trained"]
