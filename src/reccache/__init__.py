"""
RecCache: ML-aware recommendation caching system.

Core components:
- User clustering based on behavior embeddings
- Two-level caching (local + distributed)
- Quality prediction for cache decisions
- Lightweight reranking for cached results
"""

__version__ = "0.1.0"

from reccache.cache.manager import CacheManager
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.quality_predictor import QualityPredictor
from reccache.models.reranker import LightweightReranker

__all__ = [
    "CacheManager",
    "UserClusterManager",
    "QualityPredictor",
    "LightweightReranker",
]
