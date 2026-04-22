"""Speculative Recommendation: recommendation caching as speculative serving.

Applies the speculative decoding paradigm from LLM inference (Sequoia,
TriForce, MagicDec) to recommendation serving:

  Phase 1 — DRAFT:   Retrieve cached recs from K nearest clusters.
  Phase 2 — VERIFY:  Compute acceptance probability for each cluster.
  Phase 3 — ACCEPT or RESIDUAL:
      If max(alpha) >= threshold -> serve best cluster's cache (+ optional rerank).
      If all rejected -> compute fresh recs via full model, store in nearest cluster.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import AcceptanceCriterion, AcceptanceResult
from reccache.cache.retrieval_pool import PoolManager, RetrievalResult


@dataclass
class SpeculativeConfig:
    """Configuration for speculative recommendation."""

    top_k_clusters: int = 3
    acceptance_threshold: float = 0.5
    n_recs: int = 10
    rerank_on_accept: bool = False
    # Pool retrieval settings
    use_pool_retrieval: bool = False
    pool_size: int = 200


@dataclass
class SpeculativeResult:
    """Result from a single speculative recommendation request."""

    user_id: int
    items: List[int]
    accepted: bool
    acceptance_prob: float
    accepted_cluster_id: Optional[int]
    accepted_cluster_rank: int  # 0-based rank among K candidates (0 = nearest)
    latency_ms: float
    phase: str  # "accept" or "residual"
    retrieval_personalised: bool = False  # True if pool retrieval was used


class SpeculativeRecommender:
    """
    Three-phase speculative recommendation pipeline.

    Composes:
      - Any recommender (the "target model")
      - UserClusterManager (for multi-cluster lookup)
      - AcceptanceCriterion (for verify phase)
      - Optional LightweightReranker (post-accept personalisation)
      - Optional PoolManager (per-user retrieval from cluster pools)

    When ``config.use_pool_retrieval`` is True, the draft phase retrieves
    personalised items from per-cluster embedding pools instead of returning
    a static cached list.  This is inspired by TriForce's RetrievalCache
    (COLM 2024) and addresses the core limitation that all users in a cluster
    receive identical recommendations.
    """

    def __init__(
        self,
        recommender,
        cluster_manager: UserClusterManager,
        acceptance_criterion: AcceptanceCriterion,
        config: SpeculativeConfig = None,
        reranker=None,
        item_embeddings: Optional[np.ndarray] = None,
        user_history: Optional[Dict[int, List[int]]] = None,
    ):
        self.recommender = recommender
        self.cluster_manager = cluster_manager
        self.acceptance_criterion = acceptance_criterion
        self.config = config or SpeculativeConfig()
        self.reranker = reranker
        self.item_embeddings = item_embeddings
        self.user_history = user_history or {}

        # Per-cluster cache: cluster_id -> list of recommended item IDs
        self._cluster_cache: Dict[int, List[int]] = {}

        # Pool-based retrieval (when config.use_pool_retrieval is True)
        self._pool_manager: Optional[PoolManager] = None
        if self.config.use_pool_retrieval and item_embeddings is not None:
            # Extract item biases from model for consistent pool scoring
            item_biases = None
            global_bias = 0.0
            if hasattr(recommender, 'item_bias'):
                import torch
                with torch.no_grad():
                    item_biases = recommender.item_bias.weight.squeeze().cpu().numpy()
                    global_bias = float(recommender.global_bias.data.cpu())
            self._pool_manager = PoolManager(
                pool_size=self.config.pool_size,
                embedding_dim=item_embeddings.shape[1],
                item_embeddings=item_embeddings,
                item_biases=item_biases,
                global_bias=global_bias,
            )

        # Statistics
        self._stats = {
            "total_requests": 0,
            "accepted": 0,
            "rejected": 0,
            "cluster_rank_counts": defaultdict(int),  # rank -> count
            "acceptance_probs": [],
            "pool_retrievals": 0,
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def warm_cache(self, user_ids: List[int]):
        """Seed cluster caches by generating recs for a set of users.

        For each user, find its nearest cluster and populate the cache
        (or pool) if not already present.
        """
        for uid in user_ids:
            candidates = self.cluster_manager.get_nearest_clusters(uid, top_k=1)
            if not candidates:
                continue
            cid = candidates[0].cluster_id

            exclude = self.user_history.get(uid)

            if self._pool_manager is not None:
                # Pool mode: accumulate items into the cluster's embedding pool
                if not self._pool_manager.has_pool(cid):
                    recs = self.recommender.recommend(uid, n=self.config.n_recs, exclude_items=exclude)
                    self._pool_manager.populate_pool(cid, list(recs))
                else:
                    # Even if pool exists, add more items for diversity
                    recs = self.recommender.recommend(uid, n=self.config.n_recs, exclude_items=exclude)
                    self._pool_manager.populate_pool(cid, list(recs))
            else:
                # Flat-cache mode (original behaviour)
                if cid not in self._cluster_cache:
                    recs = self.recommender.recommend(uid, n=self.config.n_recs, exclude_items=exclude)
                    self._cluster_cache[cid] = list(recs)

    def clear_cache(self):
        self._cluster_cache.clear()
        if self._pool_manager is not None:
            self._pool_manager = PoolManager(
                pool_size=self.config.pool_size,
                embedding_dim=self.item_embeddings.shape[1] if self.item_embeddings is not None else 64,
                item_embeddings=self.item_embeddings,
                item_biases=self._pool_manager.item_biases,
                global_bias=self._pool_manager.global_bias,
            )

    # ------------------------------------------------------------------
    # Core three-phase pipeline
    # ------------------------------------------------------------------
    def recommend(self, user_id: int) -> SpeculativeResult:
        """Run the full Draft -> Verify -> Accept/Residual pipeline.

        When ``use_pool_retrieval`` is enabled, Phase 1 retrieves
        *personalised* items from each cluster's embedding pool instead
        of returning the same static list for every user in a cluster.
        """
        start = time.time()
        self._stats["total_requests"] += 1

        # --- Phase 1: DRAFT ---
        candidates = self.cluster_manager.get_nearest_clusters(
            user_id, top_k=self.config.top_k_clusters
        )
        user_emb = self.cluster_manager.get_user_embedding(user_id)

        use_pool = self._pool_manager is not None

        # --- Phase 2: VERIFY ---
        best_result: Optional[AcceptanceResult] = None
        best_rank = -1
        best_items: Optional[List[int]] = None
        used_pool = False

        for rank, cand in enumerate(candidates):
            if use_pool:
                # Pool retrieval: personalised draft per user
                retrieval = self._pool_manager.retrieve_for_user(
                    cand.cluster_id, user_emb, top_k=self.config.n_recs
                )
                if not retrieval.item_ids:
                    continue
                cached = retrieval.item_ids
            else:
                # Flat cache: same list for all users in cluster
                cached = self._cluster_cache.get(cand.cluster_id)
                if cached is None:
                    continue

            result = self.acceptance_criterion.compute_acceptance(
                user_embedding=user_emb,
                cluster_center=cand.center,
                cluster_id=cand.cluster_id,
                cached_item_ids=cached,
                item_embeddings=self.item_embeddings,
            )

            self._stats["acceptance_probs"].append(result.acceptance_prob)

            if result.acceptance_prob >= self.config.acceptance_threshold:
                if best_result is None or result.acceptance_prob > best_result.acceptance_prob:
                    best_result = result
                    best_rank = rank
                    best_items = list(cached)
                    used_pool = use_pool

        # --- Phase 3: ACCEPT or RESIDUAL ---
        if best_result is not None and best_items is not None:
            # ACCEPT
            items = best_items
            if self.config.rerank_on_accept and self.reranker is not None:
                reranked = self.reranker.rerank(user_id, items, top_k=self.config.n_recs)
                items = reranked.items

            self._stats["accepted"] += 1
            self._stats["cluster_rank_counts"][best_rank] += 1
            if used_pool:
                self._stats["pool_retrievals"] += 1

            elapsed = (time.time() - start) * 1000
            return SpeculativeResult(
                user_id=user_id,
                items=items[:self.config.n_recs],
                accepted=True,
                acceptance_prob=best_result.acceptance_prob,
                accepted_cluster_id=best_result.cluster_id,
                accepted_cluster_rank=best_rank,
                latency_ms=elapsed,
                phase="accept",
                retrieval_personalised=used_pool,
            )

        # RESIDUAL — compute fresh recs
        exclude = self.user_history.get(user_id)
        items = list(self.recommender.recommend(user_id, n=self.config.n_recs, exclude_items=exclude))

        # Store in nearest cluster cache/pool for future users
        if candidates:
            nearest_cid = candidates[0].cluster_id
            if use_pool:
                self._pool_manager.populate_pool(nearest_cid, list(items))
            else:
                self._cluster_cache[nearest_cid] = list(items)

        self._stats["rejected"] += 1
        elapsed = (time.time() - start) * 1000
        return SpeculativeResult(
            user_id=user_id,
            items=items[:self.config.n_recs],
            accepted=False,
            acceptance_prob=max(
                (r.acceptance_prob for r in [best_result] if r is not None),
                default=0.0,
            ),
            accepted_cluster_id=None,
            accepted_cluster_rank=-1,
            latency_ms=elapsed,
            phase="residual",
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        total = self._stats["total_requests"]
        if total == 0:
            return {"total_requests": 0, "acceptance_rate": 0.0}

        accept_rate = self._stats["accepted"] / total

        rank_dist = dict(self._stats["cluster_rank_counts"])

        stats = {
            "total_requests": total,
            "accepted": self._stats["accepted"],
            "rejected": self._stats["rejected"],
            "acceptance_rate": accept_rate,
            "cluster_rank_distribution": rank_dist,
            "mean_acceptance_prob": (
                float(np.mean(self._stats["acceptance_probs"]))
                if self._stats["acceptance_probs"]
                else 0.0
            ),
        }

        if self._pool_manager is not None:
            stats["pool_retrievals"] = self._stats["pool_retrievals"]
            stats["pool_retrieval_rate"] = self._stats["pool_retrievals"] / total if total > 0 else 0.0
            stats["pool_stats"] = self._pool_manager.get_stats()

        return stats

    def reset_stats(self):
        self._stats = {
            "total_requests": 0,
            "accepted": 0,
            "rejected": 0,
            "cluster_rank_counts": defaultdict(int),
            "acceptance_probs": [],
            "pool_retrievals": 0,
        }
