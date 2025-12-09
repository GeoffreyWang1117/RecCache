# RecCache: ML-Aware Recommendation Caching System

RecCache is an intelligent caching system for personalized recommendations that balances latency, cost, and recommendation quality. It uses machine learning to decide when cached results are "good enough" for similar users.

## Problem

Online recommendation systems face a fundamental tradeoff:
- **Fresh recommendations**: High quality but expensive (high latency, compute cost)
- **Cached recommendations**: Fast and cheap but may be stale or less personalized

RecCache bridges this gap by learning when similar users can share recommendations without significant quality loss.

## Key Features

### 🎯 ML-Aware Caching
- **User Clustering**: Groups users by behavior embeddings using online K-Means
- **Quality Prediction**: Predicts quality loss before using cached results
- **Adaptive Decisions**: Dynamically chooses cache vs. fresh based on predicted quality

### ⚡ Two-Level Cache Architecture
- **L1 (Local)**: In-process cache for ultra-low latency (<1ms)
- **L2 (Redis)**: Distributed cache for cross-instance sharing

### 🔄 Smart Cache Management
- **Quality-Aware Eviction**: Combines LRU with predicted recommendation quality
- **Lightweight Reranking**: Fast personalization of cached results
- **Intelligent Key Design**: (user_cluster, context_hash) → recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Recommendation Request                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      User Cluster Manager                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ User Embedding│→│Online K-Means │→│ Cluster Assignment   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Quality Predictor                           │
│  Input: (distance_to_center, cluster_size, context_match)       │
│  Output: predicted_quality_loss, use_cache_decision             │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌──────────────┐        ┌──────────────┐
            │  Use Cache   │        │ Fresh Compute│
            └──────────────┘        └──────────────┘
                    │                       │
                    ▼                       │
┌─────────────────────────────────────────┐ │
│            Two-Level Cache              │ │
│  ┌─────────┐    ┌─────────────────┐   │ │
│  │L1 Local │ → │ L2 Redis       │   │ │
│  │(<1ms)   │    │ (distributed)  │   │ │
│  └─────────┘    └─────────────────┘   │ │
└─────────────────────────────────────────┘ │
                    │                       │
                    ▼                       ▼
            ┌──────────────┐        ┌──────────────┐
            │   Reranker   │        │Store in Cache│
            │(personalize) │        └──────────────┘
            └──────────────┘                │
                    │                       │
                    └───────────┬───────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Recommendations                              │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/example/reccache.git
cd reccache

# Install dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from reccache import CacheManager, UserClusterManager
from reccache.models import MatrixFactorizationRecommender
from reccache.utils import DataLoader, Config

# Load data
loader = DataLoader()
train, val, test = loader.load_movielens_100k()

# Train recommender
recommender = MatrixFactorizationRecommender(
    n_users=train.n_users,
    n_items=train.n_items,
    embedding_dim=64,
)
recommender.fit(train.user_ids, train.item_ids, train.ratings)

# Setup clustering
cluster_manager = UserClusterManager(n_clusters=50, embedding_dim=64)
cluster_manager.set_item_embeddings(recommender.get_all_item_embeddings())
cluster_manager.initialize_from_interactions(
    train.user_ids, train.item_ids, train.ratings
)

# Setup cache
cache_manager = CacheManager(cluster_manager=cluster_manager)

# Use cache-aware recommendations
from reccache.cache import CacheAwareRecommender

cached_recommender = CacheAwareRecommender(
    recommender=recommender,
    cache_manager=cache_manager,
)

# Get recommendations (automatically uses cache when beneficial)
recs, metadata = cached_recommender.recommend(user_id=123, n=20)
print(f"Cache hit: {metadata['cache_hit']}, Latency: {metadata['latency_ms']:.2f}ms")
```

## Running the Demo

```bash
# Run MovieLens demo
python scripts/demo_movielens.py

# Run benchmarks
python scripts/benchmark.py --n-requests 10000
```

## Evaluation Results

On MovieLens 100K with default configuration:

| Metric | Value |
|--------|-------|
| Cache Hit Rate | ~45% |
| Latency Reduction | ~85% (on cache hits) |
| NDCG Degradation | <3% |
| Compute Cost Savings | ~40% |

## Key Components

### User Clustering (`reccache/clustering/`)
- **OnlineKMeans**: Incremental K-Means with decay for evolving user preferences
- **UserClusterManager**: Manages user embeddings and cluster assignments

### Cache System (`reccache/cache/`)
- **LocalCache**: LRU + quality-aware eviction
- **RedisCache**: Distributed cache with quality tracking
- **CacheKeyBuilder**: Builds keys from (cluster, context) tuples
- **CacheManager**: Coordinates two-level caching strategy

### Models (`reccache/models/`)
- **MatrixFactorizationRecommender**: PyTorch-based collaborative filtering
- **QualityPredictor**: Predicts cache quality degradation
- **LightweightReranker**: Fast reranking for personalization

### Evaluation (`reccache/evaluation/`)
- **RecommendationMetrics**: NDCG, Precision, Recall, Hit Rate, MRR
- **CacheEvaluator**: Cache hit rate and quality tradeoff analysis
- **OnlineSimulator**: Traffic simulation with various patterns

## Configuration

```python
from reccache.utils import Config, CacheConfig, ClusterConfig

config = Config(
    cache=CacheConfig(
        local_cache_size=10000,
        local_cache_ttl=300,  # seconds
        redis_host="localhost",
        redis_port=6379,
        quality_threshold=0.1,  # Max acceptable quality loss
    ),
    cluster=ClusterConfig(
        n_clusters=100,
        embedding_dim=64,
        update_interval=1000,
    ),
)
```

## Project Structure

```
reccache/
├── src/reccache/
│   ├── cache/           # Caching components
│   │   ├── local_cache.py
│   │   ├── redis_cache.py
│   │   ├── key_builder.py
│   │   └── manager.py
│   ├── clustering/      # User clustering
│   │   ├── online_kmeans.py
│   │   └── user_cluster.py
│   ├── models/          # ML models
│   │   ├── recommender.py
│   │   ├── quality_predictor.py
│   │   └── reranker.py
│   ├── evaluation/      # Metrics and simulation
│   │   ├── metrics.py
│   │   └── simulator.py
│   └── utils/           # Utilities
│       ├── config.py
│       └── data_loader.py
├── scripts/             # Demo and benchmark scripts
├── tests/               # Unit tests
└── data/                # Dataset storage
```

## Skills Demonstrated

### Software Engineering
- Two-level cache architecture design
- Thread-safe implementations with locking
- Redis integration for distributed caching
- Modular, extensible code structure

### Machine Learning
- User representation learning via embeddings
- Online K-Means clustering with incremental updates
- Quality prediction models
- Recommendation system fundamentals (Matrix Factorization, NCF)

## Future Improvements

- [ ] Support for more datasets (Amazon, KuaiRec)
- [ ] A/B testing framework integration
- [ ] Real-time cluster updates with streaming
- [ ] GPU acceleration for embeddings
- [ ] Kubernetes deployment configs

## License

MIT License
