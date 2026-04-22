"""Embedding pool retrieval for per-user personalization from cached pools.

Replaces static cluster-level cached lists with per-cluster item embedding
pools.  At serving time, a user's embedding is used as a query to retrieve
the most relevant items via lightweight dot-product similarity -- inspired
by TriForce's RetrievalCache (COLM 2024) which selects 3.2% of KV entries
per query from a 128K-token context.

Key insight: the original speculative pipeline stores a single item list per
cluster, so *all* users in the same cluster get identical recommendations
regardless of clustering quality.  With retrieval pools, clustering quality
directly impacts pool relevance, and per-user retrieval yields personalised
results.

Usage in the speculative pipeline:
  Phase 1 (DRAFT): instead of looking up cluster_cache[c] -> List[int],
      use pool.retrieve(user_emb, top_k) -> personalised List[int].
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RetrievalResult:
    """Result of a pool retrieval."""
    item_ids: List[int]
    scores: np.ndarray
    confidence: float  # mean of top-K scores, for acceptance gating


class EmbeddingPool:
    """Per-cluster item embedding pool with lightweight retrieval.

    Instead of caching a fixed recommendation list, each cluster maintains
    a pool of item embeddings.  A user query retrieves the top-K items by
    dot-product similarity, giving per-user personalisation at sub-ms cost.

    Parameters
    ----------
    pool_size : int
        Maximum number of items in the pool.
    embedding_dim : int
        Dimensionality of item embeddings.
    """

    def __init__(self, pool_size: int = 200, embedding_dim: int = 64,
                 item_biases: Optional[np.ndarray] = None,
                 global_bias: float = 0.0):
        self.pool_size = pool_size
        self.embedding_dim = embedding_dim
        self.item_biases = item_biases  # (n_items,) — bias per item id
        self.global_bias = global_bias

        self.item_ids: List[int] = []
        self.embeddings: Optional[np.ndarray] = None  # (n, dim)
        self.importance: np.ndarray = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------
    def add_items(
        self,
        item_ids: List[int],
        embeddings: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ):
        """Add items to pool; evict lowest-importance if full.

        Parameters
        ----------
        item_ids : list of int
            Item identifiers.
        embeddings : ndarray, shape (n, dim)
            Item embedding vectors.
        scores : ndarray, optional
            Importance scores (e.g. recommendation frequency).
            If None, new items receive importance 1.0.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_new = len(item_ids)
        if scores is None:
            scores = np.ones(n_new, dtype=np.float32)

        # Update existing items (increase importance)
        existing_set = {iid: idx for idx, iid in enumerate(self.item_ids)}
        new_ids, new_embs, new_scores = [], [], []
        for i, iid in enumerate(item_ids):
            if iid in existing_set:
                idx = existing_set[iid]
                self.importance[idx] += scores[i]
                # EMA update embedding
                self.embeddings[idx] = 0.9 * self.embeddings[idx] + 0.1 * embeddings[i]
            else:
                new_ids.append(iid)
                new_embs.append(embeddings[i])
                new_scores.append(scores[i])

        if not new_ids:
            return

        new_embs_arr = np.array(new_embs, dtype=np.float32)
        new_scores_arr = np.array(new_scores, dtype=np.float32)

        if self.embeddings is None or len(self.item_ids) == 0:
            self.item_ids = list(new_ids)
            self.embeddings = new_embs_arr
            self.importance = new_scores_arr
        else:
            self.item_ids.extend(new_ids)
            self.embeddings = np.vstack([self.embeddings, new_embs_arr])
            self.importance = np.concatenate([self.importance, new_scores_arr])

        # Evict if over capacity
        if len(self.item_ids) > self.pool_size:
            self._evict()

    def _evict(self):
        """Keep top pool_size items by importance."""
        keep = min(self.pool_size, len(self.item_ids))
        top_idx = np.argsort(self.importance)[-keep:]
        top_idx.sort()  # preserve relative order

        self.item_ids = [self.item_ids[i] for i in top_idx]
        self.embeddings = self.embeddings[top_idx]
        self.importance = self.importance[top_idx]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        user_embedding: np.ndarray,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve top-K items by dot-product similarity with user embedding.

        Analogous to TriForce's RetrievalCache: user_emb is the query,
        pool embeddings are the KV entries, and we select the top-K by
        attention score.
        """
        if self.embeddings is None or len(self.item_ids) == 0:
            return RetrievalResult(item_ids=[], scores=np.array([]), confidence=0.0)

        # Dense dot-product retrieval: O(pool_size * dim)
        scores = self.embeddings @ user_embedding  # (pool_size,)
        # Add item biases for consistency with full model scoring
        if self.item_biases is not None:
            for i, iid in enumerate(self.item_ids):
                if iid < len(self.item_biases):
                    scores[i] += self.item_biases[iid] + self.global_bias
        top_k_clamped = min(top_k, len(self.item_ids))
        top_idx = np.argpartition(scores, -top_k_clamped)[-top_k_clamped:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        item_ids = [self.item_ids[i] for i in top_idx]
        top_scores = scores[top_idx]

        # Confidence: sigmoid of mean score (maps to [0, 1])
        mean_score = float(np.mean(top_scores))
        confidence = 1.0 / (1.0 + np.exp(-mean_score))

        return RetrievalResult(
            item_ids=item_ids,
            scores=top_scores,
            confidence=confidence,
        )

    def chunk_retrieve(
        self,
        user_embedding: np.ndarray,
        chunk_size: int = 16,
        top_k: int = 10,
        top_chunks: int = 4,
    ) -> RetrievalResult:
        """Two-stage chunk-level retrieval (TriForce variant).

        Stage 1: score chunk centroids, select top chunks.
        Stage 2: fine-grained scoring within selected chunks.

        Faster than dense retrieval for large pools (>500 items).
        """
        if self.embeddings is None or len(self.item_ids) == 0:
            return RetrievalResult(item_ids=[], scores=np.array([]), confidence=0.0)

        n = len(self.item_ids)
        if n <= chunk_size * top_chunks:
            return self.retrieve(user_embedding, top_k)

        # Stage 1: chunk centroids
        n_chunks = (n + chunk_size - 1) // chunk_size
        chunk_centroids = np.zeros((n_chunks, self.embedding_dim), dtype=np.float32)
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n)
            chunk_centroids[ci] = self.embeddings[start:end].mean(axis=0)

        chunk_scores = chunk_centroids @ user_embedding
        top_chunk_idx = np.argsort(chunk_scores)[-top_chunks:]

        # Stage 2: fine-grained within selected chunks
        candidate_idx = []
        for ci in top_chunk_idx:
            start = ci * chunk_size
            end = min(start + chunk_size, n)
            candidate_idx.extend(range(start, end))

        candidate_idx = np.array(candidate_idx)
        candidate_scores = self.embeddings[candidate_idx] @ user_embedding
        top_k_clamped = min(top_k, len(candidate_idx))
        local_top = np.argpartition(candidate_scores, -top_k_clamped)[-top_k_clamped:]
        local_top = local_top[np.argsort(candidate_scores[local_top])[::-1]]

        global_idx = candidate_idx[local_top]
        item_ids = [self.item_ids[i] for i in global_idx]
        top_scores = candidate_scores[local_top]

        mean_score = float(np.mean(top_scores))
        confidence = 1.0 / (1.0 + np.exp(-mean_score))

        return RetrievalResult(
            item_ids=item_ids,
            scores=top_scores,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def size(self) -> int:
        return len(self.item_ids)

    def get_stats(self) -> Dict:
        return {
            "pool_size": len(self.item_ids),
            "max_pool_size": self.pool_size,
            "mean_importance": float(np.mean(self.importance)) if len(self.importance) > 0 else 0.0,
        }


class PoolManager:
    """Manages embedding pools across all clusters.

    Integrates with the speculative pipeline by replacing the flat
    ``_cluster_cache: Dict[int, List[int]]`` with per-cluster EmbeddingPools.
    """

    def __init__(
        self,
        pool_size: int = 200,
        embedding_dim: int = 64,
        item_embeddings: Optional[np.ndarray] = None,
        item_biases: Optional[np.ndarray] = None,
        global_bias: float = 0.0,
    ):
        self.pool_size = pool_size
        self.embedding_dim = embedding_dim
        self.item_embeddings = item_embeddings
        self.item_biases = item_biases
        self.global_bias = global_bias
        self._pools: Dict[int, EmbeddingPool] = {}

    def get_or_create_pool(self, cluster_id: int) -> EmbeddingPool:
        if cluster_id not in self._pools:
            self._pools[cluster_id] = EmbeddingPool(
                pool_size=self.pool_size,
                embedding_dim=self.embedding_dim,
                item_biases=self.item_biases,
                global_bias=self.global_bias,
            )
        return self._pools[cluster_id]

    def populate_pool(self, cluster_id: int, item_ids: List[int]):
        """Populate a cluster pool from item IDs using stored item embeddings."""
        if self.item_embeddings is None:
            return
        pool = self.get_or_create_pool(cluster_id)
        valid = [i for i in item_ids if i < len(self.item_embeddings)]
        if valid:
            embs = self.item_embeddings[valid]
            pool.add_items(valid, embs)

    def retrieve_for_user(
        self,
        cluster_id: int,
        user_embedding: np.ndarray,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve personalised items from a cluster's pool."""
        pool = self._pools.get(cluster_id)
        if pool is None or pool.size() == 0:
            return RetrievalResult(item_ids=[], scores=np.array([]), confidence=0.0)
        return pool.retrieve(user_embedding, top_k)

    def get_pool_item_ids(self, cluster_id: int) -> List[int]:
        """Return flat item list for backward compatibility with acceptance criteria."""
        pool = self._pools.get(cluster_id)
        if pool is None:
            return []
        return list(pool.item_ids)

    def has_pool(self, cluster_id: int) -> bool:
        return cluster_id in self._pools and self._pools[cluster_id].size() > 0

    def get_stats(self) -> Dict:
        sizes = [p.size() for p in self._pools.values()]
        return {
            "n_pools": len(self._pools),
            "total_items": sum(sizes),
            "mean_pool_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_pool_size": max(sizes) if sizes else 0,
        }
