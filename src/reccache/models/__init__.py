"""Models module for recommendation and quality prediction."""

from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker
from reccache.models.recommender import MatrixFactorizationRecommender, NCFRecommender
from reccache.models.baselines import (
    BaseRecommender,
    MostPopularRecommender,
    ItemKNNRecommender,
    UserKNNRecommender,
    BPRMF,
    LightGCNRecommender,
    RandomRecommender,
    create_recommender,
    RecommenderComparator,
)
from reccache.models.acceptance import (
    AcceptanceCriterion,
    AcceptanceResult,
    CosineAcceptanceCriterion,
    ScoreRatioAcceptanceCriterion,
    HeuristicAcceptanceCriterion,
    LASERRelaxedAcceptanceCriterion,
    LowRankDraftAcceptanceCriterion,
)
from reccache.models.speculative import (
    SpeculativeRecommender,
    SpeculativeConfig,
    SpeculativeResult,
)

__all__ = [
    "QualityPredictor",
    "LightweightReranker",
    "MatrixFactorizationRecommender",
    "NCFRecommender",
    # Baselines
    "BaseRecommender",
    "MostPopularRecommender",
    "ItemKNNRecommender",
    "UserKNNRecommender",
    "BPRMF",
    "LightGCNRecommender",
    "RandomRecommender",
    "create_recommender",
    "RecommenderComparator",
    # Speculative recommendation
    "AcceptanceCriterion",
    "AcceptanceResult",
    "CosineAcceptanceCriterion",
    "ScoreRatioAcceptanceCriterion",
    "HeuristicAcceptanceCriterion",
    "LASERRelaxedAcceptanceCriterion",
    "LowRankDraftAcceptanceCriterion",
    "SpeculativeRecommender",
    "SpeculativeConfig",
    "SpeculativeResult",
]
