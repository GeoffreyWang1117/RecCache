"""Configuration management for RecCache."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheConfig:
    """Cache configuration."""

    # Local cache settings
    local_cache_size: int = 10000
    local_cache_ttl: int = 300  # seconds

    # Redis cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_cache_ttl: int = 3600  # seconds
    redis_max_connections: int = 10

    # Cache behavior
    use_local_cache: bool = True
    use_redis_cache: bool = True
    quality_threshold: float = 0.1  # Max acceptable quality loss


@dataclass
class ClusterConfig:
    """User clustering configuration."""

    n_clusters: int = 100
    embedding_dim: int = 64
    min_cluster_size: int = 10
    update_interval: int = 1000  # Update clusters every N requests
    distance_threshold: float = 0.5  # Max distance to use cluster cache


@dataclass
class ModelConfig:
    """Model configuration."""

    # Recommender
    recommender_embedding_dim: int = 64
    recommender_hidden_dims: list = field(default_factory=lambda: [128, 64])
    recommender_lr: float = 0.001
    recommender_batch_size: int = 256
    recommender_epochs: int = 20

    # Quality predictor
    quality_predictor_hidden_dim: int = 32
    quality_predictor_lr: float = 0.001

    # Reranker
    reranker_top_k: int = 100
    reranker_output_k: int = 20


@dataclass
class Config:
    """Main configuration."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # General settings
    random_seed: int = 42
    device: str = "cpu"
    log_level: str = "INFO"

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()

    @classmethod
    def for_testing(cls) -> "Config":
        """Create configuration for testing with smaller values."""
        return cls(
            cache=CacheConfig(
                local_cache_size=100,
                local_cache_ttl=60,
                redis_cache_ttl=300,
            ),
            cluster=ClusterConfig(
                n_clusters=10,
                embedding_dim=32,
                update_interval=100,
            ),
            model=ModelConfig(
                recommender_embedding_dim=32,
                recommender_hidden_dims=[64, 32],
                recommender_epochs=5,
            ),
        )
