"""Lightweight reranker for personalizing cached recommendations."""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RerankedResult:
    """Result from reranking operation."""

    items: List[int]
    scores: np.ndarray
    original_positions: List[int]
    personalization_boost: float  # How much reranking changed the order


class LightweightReranker:
    """
    Fast reranker for personalizing cached recommendations.

    Key design goals:
    - Ultra-low latency (<5ms)
    - Use pre-computed features only
    - Simple scoring function

    Reranking factors:
    - User's recent interaction history
    - Item popularity decay
    - Diversity promotion
    """

    def __init__(
        self,
        history_weight: float = 0.4,
        recency_weight: float = 0.3,
        diversity_weight: float = 0.3,
        n_history_items: int = 20,
    ):
        self.history_weight = history_weight
        self.recency_weight = recency_weight
        self.diversity_weight = diversity_weight
        self.n_history_items = n_history_items

        # Pre-computed data (set externally)
        self._user_history: Dict[int, List[int]] = {}  # user_id -> [item_ids]
        self._item_embeddings: Optional[np.ndarray] = None
        self._item_popularity: Optional[np.ndarray] = None
        self._item_categories: Optional[np.ndarray] = None

    def set_item_embeddings(self, embeddings: np.ndarray):
        """Set item embedding matrix for similarity computation."""
        self._item_embeddings = embeddings

    def set_item_popularity(self, popularity: np.ndarray):
        """Set item popularity scores."""
        self._item_popularity = popularity

    def set_item_categories(self, categories: np.ndarray):
        """Set item category assignments for diversity."""
        self._item_categories = categories

    def update_user_history(self, user_id: int, item_id: int):
        """Update user's interaction history."""
        if user_id not in self._user_history:
            self._user_history[user_id] = []

        self._user_history[user_id].append(item_id)

        # Keep only recent history
        if len(self._user_history[user_id]) > self.n_history_items:
            self._user_history[user_id] = self._user_history[user_id][-self.n_history_items:]

    def set_user_history(self, user_id: int, history: List[int]):
        """Set full history for a user."""
        self._user_history[user_id] = history[-self.n_history_items:]

    def rerank(
        self,
        user_id: int,
        items: List[int],
        context_features: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> RerankedResult:
        """
        Rerank cached recommendations for a specific user.

        Args:
            user_id: User to personalize for
            items: List of item IDs from cache
            context_features: Optional context features
            top_k: Number of items to return (default: all)

        Returns:
            RerankedResult with reordered items
        """
        if not items:
            return RerankedResult(
                items=[],
                scores=np.array([]),
                original_positions=[],
                personalization_boost=0.0,
            )

        n_items = len(items)
        items_array = np.array(items)

        # Initialize scores (higher position in original list = higher base score)
        position_scores = np.linspace(1.0, 0.5, n_items)

        # Component scores
        history_scores = self._compute_history_scores(user_id, items_array)
        recency_scores = self._compute_recency_scores(user_id, items_array)
        diversity_scores = self._compute_diversity_scores(items_array)

        # Combine scores
        final_scores = (
            position_scores * (1 - self.history_weight - self.recency_weight - self.diversity_weight)
            + history_scores * self.history_weight
            + recency_scores * self.recency_weight
            + diversity_scores * self.diversity_weight
        )

        # Sort by score
        sorted_indices = np.argsort(-final_scores)

        if top_k:
            sorted_indices = sorted_indices[:top_k]

        reranked_items = items_array[sorted_indices].tolist()
        reranked_scores = final_scores[sorted_indices]
        original_positions = sorted_indices.tolist()

        # Compute personalization boost (how much order changed)
        original_order = np.arange(len(sorted_indices))
        kendall_tau = self._kendall_tau_distance(original_order, sorted_indices[:len(original_order)])
        personalization_boost = kendall_tau

        return RerankedResult(
            items=reranked_items,
            scores=reranked_scores,
            original_positions=original_positions,
            personalization_boost=personalization_boost,
        )

    def _compute_history_scores(
        self, user_id: int, items: np.ndarray
    ) -> np.ndarray:
        """
        Score items based on similarity to user's history.

        Items similar to what user liked get higher scores.
        """
        if self._item_embeddings is None:
            return np.zeros(len(items))

        history = self._user_history.get(user_id, [])
        if not history:
            return np.zeros(len(items))

        # Get embeddings
        valid_history = [h for h in history if h < len(self._item_embeddings)]
        if not valid_history:
            return np.zeros(len(items))

        history_embs = self._item_embeddings[valid_history]
        item_embs = self._item_embeddings[items]

        # Average history embedding
        avg_history = history_embs.mean(axis=0)
        avg_history /= np.linalg.norm(avg_history) + 1e-8

        # Cosine similarity
        norms = np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-8
        item_embs_norm = item_embs / norms
        scores = item_embs_norm @ avg_history

        # Normalize to [0, 1]
        scores = (scores + 1) / 2

        return scores

    def _compute_recency_scores(
        self, user_id: int, items: np.ndarray
    ) -> np.ndarray:
        """
        Penalize items user has recently interacted with.

        Avoids recommending items user just saw.
        """
        history = self._user_history.get(user_id, [])
        if not history:
            return np.ones(len(items))

        recent_set = set(history[-10:])  # Last 10 items

        scores = np.ones(len(items))
        for i, item in enumerate(items):
            if item in recent_set:
                scores[i] = 0.3  # Penalize recently seen

        return scores

    def _compute_diversity_scores(self, items: np.ndarray) -> np.ndarray:
        """
        Promote diversity in recommendations.

        Uses a greedy approach: items from underrepresented categories
        get higher scores.
        """
        if self._item_categories is None:
            return np.ones(len(items))

        categories = self._item_categories[items]
        unique_cats, cat_counts = np.unique(categories, return_counts=True)

        # Category frequency in this list
        cat_freq = dict(zip(unique_cats, cat_counts))

        # Score inversely proportional to category frequency
        scores = np.array([
            1.0 / cat_freq.get(self._item_categories[item], 1)
            for item in items
        ])

        # Normalize
        scores = scores / scores.max() if scores.max() > 0 else scores

        return scores

    def _kendall_tau_distance(
        self, order1: np.ndarray, order2: np.ndarray
    ) -> float:
        """
        Compute normalized Kendall tau distance between two orderings.

        Returns value in [0, 1] where 0 = identical, 1 = reversed.
        """
        n = min(len(order1), len(order2))
        if n <= 1:
            return 0.0

        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (order1[i] < order1[j]) != (order2[i] < order2[j]):
                    discordant += 1

        max_discordant = n * (n - 1) / 2
        return discordant / max_discordant if max_discordant > 0 else 0.0


class MMRReranker:
    """
    Maximal Marginal Relevance reranker for diversity.

    Balances relevance with diversity through iterative selection.
    """

    def __init__(
        self,
        lambda_param: float = 0.5,  # Balance between relevance and diversity
    ):
        self.lambda_param = lambda_param
        self._item_embeddings: Optional[np.ndarray] = None

    def set_item_embeddings(self, embeddings: np.ndarray):
        """Set item embeddings for similarity computation."""
        self._item_embeddings = embeddings

    def rerank(
        self,
        items: List[int],
        relevance_scores: np.ndarray,
        top_k: int = 20,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Rerank using MMR.

        Args:
            items: Candidate items
            relevance_scores: Relevance score for each item
            top_k: Number of items to select

        Returns:
            Tuple of (selected_items, scores)
        """
        if self._item_embeddings is None:
            # No diversity, just return top by relevance
            sorted_idx = np.argsort(-relevance_scores)[:top_k]
            return [items[i] for i in sorted_idx], relevance_scores[sorted_idx]

        n_items = len(items)
        items_array = np.array(items)

        # Get embeddings
        item_embs = self._item_embeddings[items_array]

        # Normalize for cosine similarity
        norms = np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-8
        item_embs_norm = item_embs / norms

        # Precompute similarity matrix
        sim_matrix = item_embs_norm @ item_embs_norm.T

        selected = []
        selected_mask = np.zeros(n_items, dtype=bool)
        scores = []

        for _ in range(min(top_k, n_items)):
            # Compute MMR scores
            mmr_scores = np.full(n_items, -np.inf)

            for i in range(n_items):
                if selected_mask[i]:
                    continue

                relevance = relevance_scores[i]

                if not selected:
                    diversity = 0.0
                else:
                    # Max similarity to any selected item
                    max_sim = sim_matrix[i, selected].max()
                    diversity = max_sim

                mmr_scores[i] = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * diversity
                )

            # Select item with highest MMR
            best_idx = np.argmax(mmr_scores)
            selected.append(best_idx)
            selected_mask[best_idx] = True
            scores.append(mmr_scores[best_idx])

        selected_items = items_array[selected].tolist()
        return selected_items, np.array(scores)
