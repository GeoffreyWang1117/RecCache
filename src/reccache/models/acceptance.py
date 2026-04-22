"""Acceptance criteria for speculative recommendation serving.

Implements the verification phase of speculative decoding applied to
recommendation caching: given a user and a candidate cluster, compute the
probability that the cluster's cached recommendations are acceptable.

Five implementations:
  - CosineAcceptanceCriterion         (simple, no learned params)
  - ScoreRatioAcceptanceCriterion     (direct speculative decoding analogy, OURS)
  - HeuristicAcceptanceCriterion      (wraps existing QualityPredictor)
  - LASERRelaxedAcceptanceCriterion   (LASER-adapted: mean ratio vs product ratio)
  - LowRankDraftAcceptanceCriterion   (BiLD-style: trained low-rank draft model)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AcceptanceResult:
    """Result of an acceptance criterion evaluation."""

    acceptance_prob: float
    accept: bool
    cluster_id: int
    details: Dict = field(default_factory=dict)


class AcceptanceCriterion(ABC):
    """Base class for speculative acceptance criteria."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abstractmethod
    def compute_acceptance(
        self,
        user_embedding: np.ndarray,
        cluster_center: np.ndarray,
        cluster_id: int,
        cached_item_ids: Optional[List[int]] = None,
        item_embeddings: Optional[np.ndarray] = None,
    ) -> AcceptanceResult:
        """Compute acceptance probability for a (user, cluster) pair."""
        ...


class CosineAcceptanceCriterion(AcceptanceCriterion):
    """
    Acceptance based on cosine similarity between user embedding and cluster center.

        alpha = (cosine_sim(user_emb, cluster_center) + 1) / 2

    Simple, interpretable, no learned parameters. Accept if alpha >= threshold.
    """

    def compute_acceptance(
        self,
        user_embedding: np.ndarray,
        cluster_center: np.ndarray,
        cluster_id: int,
        cached_item_ids: Optional[List[int]] = None,
        item_embeddings: Optional[np.ndarray] = None,
    ) -> AcceptanceResult:
        norm_u = np.linalg.norm(user_embedding)
        norm_c = np.linalg.norm(cluster_center)
        if norm_u < 1e-8 or norm_c < 1e-8:
            alpha = 0.0
        else:
            cosine_sim = float(np.dot(user_embedding, cluster_center) / (norm_u * norm_c))
            alpha = (cosine_sim + 1.0) / 2.0

        return AcceptanceResult(
            acceptance_prob=alpha,
            accept=alpha >= self.threshold,
            cluster_id=cluster_id,
            details={"cosine_sim": 2.0 * alpha - 1.0},
        )


class ScoreRatioAcceptanceCriterion(AcceptanceCriterion):
    """
    Direct speculative-decoding analogy using score ratios.

    For each cached item i in top-K:
        draft_prob  = softmax(cluster_center @ item_embs)[i]
        target_prob = softmax(user_emb      @ item_embs)[i]
        ratio_i = min(1, target_prob[i] / draft_prob[i])

    alpha = product of ratio_i across cached items.

    This mirrors speculative decoding's min(1, p_target / p_draft) token-level
    acceptance rule, giving the paper its theoretical contribution.
    """

    def __init__(self, threshold: float = 0.5, temperature: float = 1.0):
        super().__init__(threshold)
        self.temperature = temperature

    def compute_acceptance(
        self,
        user_embedding: np.ndarray,
        cluster_center: np.ndarray,
        cluster_id: int,
        cached_item_ids: Optional[List[int]] = None,
        item_embeddings: Optional[np.ndarray] = None,
    ) -> AcceptanceResult:
        if cached_item_ids is None or item_embeddings is None or len(cached_item_ids) == 0:
            # Fall back to cosine when item info is unavailable
            norm_u = np.linalg.norm(user_embedding)
            norm_c = np.linalg.norm(cluster_center)
            if norm_u < 1e-8 or norm_c < 1e-8:
                alpha = 0.0
            else:
                alpha = float((np.dot(user_embedding, cluster_center) / (norm_u * norm_c) + 1.0) / 2.0)
            return AcceptanceResult(
                acceptance_prob=alpha,
                accept=alpha >= self.threshold,
                cluster_id=cluster_id,
                details={"fallback": "cosine"},
            )

        valid_ids = [i for i in cached_item_ids if i < len(item_embeddings)]
        if not valid_ids:
            return AcceptanceResult(
                acceptance_prob=0.0, accept=False, cluster_id=cluster_id,
                details={"error": "no_valid_items"},
            )

        cached_embs = item_embeddings[valid_ids]  # (K, dim)

        # Draft scores (cluster center as proxy model)
        draft_logits = cached_embs @ cluster_center / self.temperature
        draft_logits -= draft_logits.max()  # numerical stability
        draft_probs = np.exp(draft_logits)
        draft_probs /= draft_probs.sum() + 1e-12

        # Target scores (user embedding as true model)
        target_logits = cached_embs @ user_embedding / self.temperature
        target_logits -= target_logits.max()
        target_probs = np.exp(target_logits)
        target_probs /= target_probs.sum() + 1e-12

        # Per-item acceptance ratio: min(1, target / draft)
        ratios = np.minimum(1.0, target_probs / (draft_probs + 1e-12))
        alpha = float(np.prod(ratios))

        return AcceptanceResult(
            acceptance_prob=alpha,
            accept=alpha >= self.threshold,
            cluster_id=cluster_id,
            details={
                "mean_ratio": float(np.mean(ratios)),
                "min_ratio": float(np.min(ratios)),
                "n_items": len(valid_ids),
            },
        )


class LASERRelaxedAcceptanceCriterion(AcceptanceCriterion):
    """
    LASER-adapted relaxed verification criterion (competitor baseline).

    Based on LASER (Xi et al., SIGIR 2025) which applies "relaxed verification"
    allowing partial token matches in LLM-based recommendation. Adapted here to
    the item-level setting: instead of the strict product rule, we use the
    arithmetic mean of per-item ratios.

        alpha = mean_j( min(1, target_prob[j] / draft_prob[j]) )

    This is strictly more permissive than ScoreRatioAcceptanceCriterion:
      prod(ratios) <= mean(ratios)^K  (AM-GM), so LASER accepts more requests.
    The trade-off is lower NDCG retention vs higher cache hit rate.

    When ``relaxation=0``, this reduces to the mean-ratio rule.
    When ``relaxation=1``, this uses the product-ratio rule (= ScoreRatio).
    The ``relaxation`` parameter interpolates: exp(relaxation * log(prod) + (1-relaxation) * log(mean)).
    """

    def __init__(self, threshold: float = 0.5, temperature: float = 1.0, relaxation: float = 0.0):
        super().__init__(threshold)
        self.temperature = temperature
        self.relaxation = float(np.clip(relaxation, 0.0, 1.0))

    def compute_acceptance(
        self,
        user_embedding: np.ndarray,
        cluster_center: np.ndarray,
        cluster_id: int,
        cached_item_ids: Optional[List[int]] = None,
        item_embeddings: Optional[np.ndarray] = None,
    ) -> AcceptanceResult:
        if cached_item_ids is None or item_embeddings is None or len(cached_item_ids) == 0:
            norm_u = np.linalg.norm(user_embedding)
            norm_c = np.linalg.norm(cluster_center)
            alpha = 0.0 if (norm_u < 1e-8 or norm_c < 1e-8) else float(
                (np.dot(user_embedding, cluster_center) / (norm_u * norm_c) + 1.0) / 2.0
            )
            return AcceptanceResult(
                acceptance_prob=alpha, accept=alpha >= self.threshold,
                cluster_id=cluster_id, details={"fallback": "cosine"},
            )

        valid_ids = [i for i in cached_item_ids if i < len(item_embeddings)]
        if not valid_ids:
            return AcceptanceResult(
                acceptance_prob=0.0, accept=False, cluster_id=cluster_id,
                details={"error": "no_valid_items"},
            )

        cached_embs = item_embeddings[valid_ids]

        draft_logits = cached_embs @ cluster_center / self.temperature
        draft_logits -= draft_logits.max()
        draft_probs = np.exp(draft_logits)
        draft_probs /= draft_probs.sum() + 1e-12

        target_logits = cached_embs @ user_embedding / self.temperature
        target_logits -= target_logits.max()
        target_probs = np.exp(target_logits)
        target_probs /= target_probs.sum() + 1e-12

        ratios = np.minimum(1.0, target_probs / (draft_probs + 1e-12))
        mean_ratio = float(np.mean(ratios))
        prod_ratio = float(np.prod(ratios))

        # Interpolate: relaxation=0 → mean, relaxation=1 → product
        if self.relaxation < 1e-6:
            alpha = mean_ratio
        elif self.relaxation > 1.0 - 1e-6:
            alpha = prod_ratio
        else:
            log_mean = np.log(mean_ratio + 1e-12)
            log_prod = np.log(prod_ratio + 1e-12)
            alpha = float(np.exp((1.0 - self.relaxation) * log_mean + self.relaxation * log_prod))

        return AcceptanceResult(
            acceptance_prob=alpha,
            accept=alpha >= self.threshold,
            cluster_id=cluster_id,
            details={
                "mean_ratio": mean_ratio,
                "prod_ratio": prod_ratio,
                "relaxation": self.relaxation,
                "n_items": len(valid_ids),
            },
        )


class LowRankDraftAcceptanceCriterion(AcceptanceCriterion):
    """
    BiLD-style trained low-rank draft acceptance criterion (competitor baseline).

    Instead of using the cluster centroid as the draft distribution, this uses
    a pre-trained low-rank recommender's user embedding as the draft. This models
    the BiLD (Kim et al., 2023) paradigm where a small trained draft model
    generates candidates verified by the full target model.

        draft_prob  = softmax(low_rank_user_emb @ item_embs)[i]
        target_prob = softmax(full_rank_user_emb @ item_embs)[i]
        alpha       = prod_j min(1, target_prob[j] / draft_prob[j])

    The ``draft_user_embeddings`` dict maps user_id -> low-rank embedding vector.
    When a user's draft embedding is unavailable, falls back to ScoreRatio with
    the cluster centroid as draft.
    """

    def __init__(
        self,
        draft_user_embeddings: Dict[int, np.ndarray],
        threshold: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__(threshold)
        self.draft_user_embeddings = draft_user_embeddings
        self.temperature = temperature

    def compute_acceptance(
        self,
        user_embedding: np.ndarray,
        cluster_center: np.ndarray,
        cluster_id: int,
        cached_item_ids: Optional[List[int]] = None,
        item_embeddings: Optional[np.ndarray] = None,
        user_id: Optional[int] = None,
    ) -> AcceptanceResult:
        # Determine draft embedding: low-rank user emb if available, else cluster center
        draft_emb = cluster_center
        used_low_rank = False
        if user_id is not None and user_id in self.draft_user_embeddings:
            draft_emb = self.draft_user_embeddings[user_id]
            used_low_rank = True

        if cached_item_ids is None or item_embeddings is None or len(cached_item_ids) == 0:
            norm_u = np.linalg.norm(user_embedding)
            norm_d = np.linalg.norm(draft_emb)
            alpha = 0.0 if (norm_u < 1e-8 or norm_d < 1e-8) else float(
                (np.dot(user_embedding, draft_emb) / (norm_u * norm_d) + 1.0) / 2.0
            )
            return AcceptanceResult(
                acceptance_prob=alpha, accept=alpha >= self.threshold,
                cluster_id=cluster_id, details={"fallback": "cosine"},
            )

        # Align embedding dimensions if low-rank draft has smaller dim
        valid_ids = [i for i in cached_item_ids if i < len(item_embeddings)]
        if not valid_ids:
            return AcceptanceResult(
                acceptance_prob=0.0, accept=False, cluster_id=cluster_id,
                details={"error": "no_valid_items"},
            )

        full_embs = item_embeddings[valid_ids]  # (K, full_dim)
        d_draft = len(draft_emb)
        d_full = full_embs.shape[1]

        # Project item embeddings to draft space if needed
        if d_draft < d_full:
            draft_item_embs = full_embs[:, :d_draft]
        else:
            draft_item_embs = full_embs

        draft_logits = draft_item_embs @ draft_emb / self.temperature
        draft_logits -= draft_logits.max()
        draft_probs = np.exp(draft_logits)
        draft_probs /= draft_probs.sum() + 1e-12

        target_logits = full_embs @ user_embedding / self.temperature
        target_logits -= target_logits.max()
        target_probs = np.exp(target_logits)
        target_probs /= target_probs.sum() + 1e-12

        # Project target probs back to draft-space items (same indices)
        ratios = np.minimum(1.0, target_probs / (draft_probs + 1e-12))
        alpha = float(np.prod(ratios))

        return AcceptanceResult(
            acceptance_prob=alpha,
            accept=alpha >= self.threshold,
            cluster_id=cluster_id,
            details={
                "mean_ratio": float(np.mean(ratios)),
                "used_low_rank_draft": used_low_rank,
                "n_items": len(valid_ids),
            },
        )


class HeuristicAcceptanceCriterion(AcceptanceCriterion):
    """
    Wraps the existing QualityPredictor as an acceptance criterion baseline.

    Maps QualityPredictor.predict().quality_score -> acceptance probability.
    """

    def __init__(self, quality_predictor, threshold: float = 0.5):
        super().__init__(threshold)
        self.quality_predictor = quality_predictor

    def compute_acceptance(
        self,
        user_embedding: np.ndarray,
        cluster_center: np.ndarray,
        cluster_id: int,
        cached_item_ids: Optional[List[int]] = None,
        item_embeddings: Optional[np.ndarray] = None,
    ) -> AcceptanceResult:
        # Compute distance to center (Euclidean, matching QualityPredictor's expected input)
        distance = float(np.linalg.norm(user_embedding - cluster_center))

        prediction = self.quality_predictor.predict(
            distance_to_center=distance,
            cluster_size=100,  # default; not available here
        )

        return AcceptanceResult(
            acceptance_prob=prediction.quality_score,
            accept=prediction.quality_score >= self.threshold,
            cluster_id=cluster_id,
            details={
                "quality_score": prediction.quality_score,
                "confidence": prediction.confidence,
                "qa_use_cache": prediction.use_cache,
            },
        )
