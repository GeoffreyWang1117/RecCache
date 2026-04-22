# RecCache: Speculative Recommendation Serving

Bridging LLM inference acceleration and recommendation caching. Transfers the **draft-and-verify** paradigm from speculative decoding to recommendation serving.

**Target venue**: RecSys 2026 (Abstract: Apr 14, Paper: Apr 21)

## Core Idea

```
Speculative Decoding (LLMs)          Speculative Rec Serving (Ours)
─────────────────────────────        ──────────────────────────────
Draft model q(x)              <-->   Cluster cache C[c]
Target model p(x)             <-->   Full recommender R(u)
min(1, p(x)/q(x)) acceptance <-->   Score-ratio acceptance
Multi-token speculation       <-->   Multi-cluster speculation
```

Three-phase pipeline:
1. **Draft**: Retrieve cached recs from K nearest user clusters
2. **Verify**: Score-ratio acceptance criterion (Eq. alpha = prod min(1, p_target/p_draft))
3. **Accept/Residual**: Serve cache if alpha >= tau, else compute fresh

Two caching modes:
- **Static cache**: One recommendation list per cluster (fast, good NDCG on dense data)
- **Embedding pool retrieval**: Per-cluster item embedding pool with per-user dot-product retrieval + bias correction (better coverage, inspired by TriForce's RetrievalCache)

## Key Results (4 datasets, corrected methodology)

All results exclude user training history and use empirically measured latencies.
Numbers from paper Table 4 (S4 end-to-end, K=3, τ=0.35, 3 seeds).

| Dataset | Method | NDCG@10 | Accept | Speedup | MCG | Coverage |
|---------|--------|---------|--------|---------|-----|----------|
| ML-1M | Fresh | .166 | -- | 1× | -- | .202 |
| ML-1M | Speculative K=3 | .095 | 91% | 1.6× | 55% | .056 |
| ML-1M | Spec.+Pool K=3 | .094 | 93% | 0.7× | 60% | .062 |
| A-Electronics | Fresh | .011 | -- | 1× | -- | .063 |
| A-Electronics | Speculative K=3 | .010 | 59% | 3.1× | 56% | .028 |
| A-Electronics | Spec.+Pool K=3 | .007 | 60% | 2.0× | 61% | .074 |
| A-Arts | Fresh | .004 | -- | 1× | -- | .051 |
| A-Arts | Speculative K=3 | .002 | 96% | 6.7× | 64% | .010 |
| A-Arts | Spec.+Pool K=3 | .004 | 96% | 3.6× | 62% | .036 |
| MIND | Fresh | .014 | -- | 1× | -- | .113 |
| MIND | Speculative K=3 | **.014** | 32% | 3.8× | 49% | .095 |
| MIND | Spec.+Pool K=3 | **.014** | 26% | 2.1× | 49% | .106 |

- **Multi-cluster gain**: 43–64% of accepted results from non-nearest clusters (robust across datasets, models, thresholds)
- **Quality retention**: 80% on Electronics, 100% on MIND (matches fresh NDCG); 65% on ML-1M
- **Training-free**: matches CARL-like RL caching in NDCG with zero training cost
- **Empirical speedup**: 1.5–9.5× on real datasets; near-constant speculative latency scales to 27× at 100K items (synthetic)
- **Regret bound**: O(T·ε) without training, empirically validated (Spearman ρ=-0.655, p<10⁻¹⁸⁰)
- **Model-agnostic**: validated across MF, NeuMF, LightGCN, Gemma-4 backbones on 6 datasets (up to 86K users)
- **Fairness**: pool retrieval increases coverage up to 2.6×, reduces recommendation Gini from 0.996 to 0.966

## Experiment Progress

### Paper Status
- **Paper**: `paper/recsys2026/main.tex`, **8 content + 1 ref pages**, 31 references, ACM sigconf
- **Venue**: RecSys 2026 (Abstract: Apr 14 AoE, Full paper: Apr 21 AoE)

### Completed Experiments

| Group | Description | Datasets | Status |
|-------|-------------|----------|--------|
| **S1** | Multi-cluster K sweep (K=1,3,5,7) | ML-1M, A-Elec, A-Arts, MIND | done |
| **S2** | Acceptance criterion comparison | All 4 | done |
| **S3** | Threshold Pareto front | ML-1M, A-Elec | done |
| **S4** | End-to-end (Fresh/Naive/QA/Spec/Spec+Pool) | All 4 | done |
| **S5** | Pool retrieval vs static cache | All 4 | done |
| **S6** | Semantic vs random clustering | All 4 | done |
| **S7** | Pool size sensitivity | ML-1M, A-Elec | done |
| **S8** | Model-agnostic (MF vs NeuMF) | ML-1M | done |
| **S9** | BiLD-style low-rank draft baseline | ML-1M | done |
| **S10** | LASER-adapted relaxed verification | ML-1M | done |
| **S11** | MIND-large scale-out (72K users) | MIND-large | done |
| **S12** | Amazon Movies scale-out (86K users) | Amazon Movies | done |
| **S13** | LightGCN GNN backbone | ML-1M, A-Elec | done |
| **S14** | Gemma-4 LLM embeddings | ML-1M | done |
| **S15** | Latency scaling (1K–100K items) | ML-1M | done |
| **A1** | Multi-metric (Recall/HR/MRR/MAP/ILD) | ML-1M, A-Elec | done |
| **A2** | Long-tail / popularity-bias / Gini | ML-1M, A-Elec | done |
| **A3** | FAISS / ANN baseline | ML-1M, A-Elec | done |
| **A4** | Cluster stability + ε-bound validation | ML-1M | done |
| **B2** | Extended log replay (3 orderings) | ML-1M | done |
| **C1** | Online serving replay (6 policies, CTR proxy) | ML-1M | done |
| **C2** | Cache refresh sensitivity (4 frequencies) | ML-1M | done |
| **TT** | Two-Tower retrieve-and-rerank baseline | ML-1M, A-Elec | done |
| **CARL** | RL caching baseline | ML-1M | done |
| **Sig** | Statistical significance | All 4 | done |
| A1-A4 | Ablations (temperature, cold-start, latency, clusters) | ML-1M, A-Elec | done |

### Methodology Fixes Applied (Mar 30-31)
1. **User history exclusion** — `exclude_items` passed to all `recommend()` calls; prevents NDCG inflation from training items
2. **Empirical speedup** — `speedup_estimate()` uses actual `latency_ms` from `SpeculativeResult`, not hardcoded 0.1ms/5ms constants
3. **Independent seeds** — clusters re-initialized per seed (K-means is stochastic); model trained once per dataset (deterministic)
4. **Pool bias correction** — `retrieval_pool.py` includes item bias + global bias in dot-product scoring, consistent with full MF model
5. **Paper fully updated** — all tables, figures references, abstract, intro, discussion, conclusion synchronized with corrected numbers

## Project Structure

**Dataset stats (after preprocessing, from paper Table 2):**

| Dataset | Users | Items | Interactions | Domain |
|---------|-------|-------|--------------|--------|
| ML-1M | 6,040 | 3,706 | 800K | Movie |
| A-Electronics | 15,306 | 10,539 | 61K | E-comm |
| A-Arts | 29,598 | 19,663 | 135K | E-comm |
| MIND-small | 8,449 | 4,027 | 50K | News |
| MIND-large | 72,366 | 20,870 | 3M | News |
| Amazon Movies | 85,992 | 32,999 | 2M | E-comm |

## Project Structure

```
RecCache/
├── src/reccache/
│   ├── cache/
│   │   ├── manager.py            # Two-level cache (L1 local + L2 Redis)
│   │   ├── retrieval_pool.py     # Embedding pool retrieval + bias correction
│   │   ├── baselines.py          # LRU, LFU, ARC, LeCaR, Oracle, etc.
│   │   ├── local_cache.py        # In-process LRU + quality-aware eviction
│   │   ├── redis_cache.py        # Distributed Redis cache
│   │   ├── key_builder.py        # (cluster, context) -> cache key
│   │   └── warming.py            # Cache warmup strategies
│   ├── models/
│   │   ├── speculative.py        # 3-phase pipeline (core) + user_history support
│   │   ├── acceptance.py         # Score-ratio, cosine, heuristic criteria
│   │   ├── recommender.py        # MF + NeuMF (PyTorch, BPR implicit support)
│   │   ├── quality_predictor.py  # Predict cache quality loss
│   │   └── reranker.py           # Lightweight post-cache reranking
│   ├── clustering/
│   │   ├── online_kmeans.py      # Incremental K-Means with decay
│   │   └── user_cluster.py       # User embedding + cluster assignment
│   ├── evaluation/
│   │   ├── metrics.py            # NDCG, HR, MRR, ILD, coverage, MCG, empirical speedup
│   │   ├── experiment.py         # Experiment framework
│   │   └── simulator.py          # Traffic simulation
│   ├── monitoring/               # Metrics collection + exporters
│   └── utils/
│       ├── data_loader.py        # ML-1M, Amazon (parquet), MIND, Yelp, etc.
│       └── config.py             # CacheConfig, ClusterConfig
├── scripts/
│   ├── run_recsys_experiments.py       # Main suite (S1-S7, 4 datasets, 3 seeds)
│   ├── run_ablation_experiments.py     # A1-A4 ablations
│   ├── run_competitor_baselines.py     # S9 BiLD + S10 LASER
│   ├── run_largescale_experiments.py   # S11/S12 MIND-large + Amazon Movies
│   ├── run_lightgcn_experiments.py     # S13 LightGCN backbone
│   ├── run_gemma4_experiments.py       # S14 Gemma-4 LLM embeddings
│   ├── run_gpu_latency.py             # S15 latency scaling
│   ├── run_supplementary_metrics.py   # A1+A2 multi-metric + long-tail
│   ├── run_faiss_baseline.py          # A3 FAISS / ANN baseline
│   ├── run_cluster_analysis.py        # A4 cluster stability + ε-bound
│   ├── run_twotower_baseline.py       # Two-tower retrieve-and-rerank
│   ├── run_online_serving_replay.py   # C1 6-policy online replay + CTR
│   ├── run_cache_refresh.py           # C2 cache refresh sensitivity
│   ├── run_carl_comparison.py         # CARL RL baseline
│   ├── run_online_simulation.py       # Streaming temporal simulation
│   └── run_extended_replay.py         # B2 3-ordering replay
├── paper/
│   ├── recsys2026/main.tex       # RecSys 2026 submission (ACM sigconf, 8pp)
│   ├── references.bib            # 31 references
│   └── figures/                  # Auto-generated experiment figures
├── baselines/
│   ├── nezha/                    # NEZHA (WWW 2026) code
│   └── atspeed/                  # AtSpeed (ICLR 2025) code
├── results/                      # Experiment results (JSON)
└── data/                         # Auto-downloaded datasets
```

## Environment Setup

Dedicated conda environment `reccache` (Python 3.11):

```
conda           miniconda3
Python          3.11
torch           2.11.0+cu130
numpy           2.4.4
scikit-learn    1.8.0
pandas          3.0.1
pyarrow         23.0.1
faiss-cpu       1.13.2
matplotlib      3.10.8
seaborn         0.13.2
```

```bash
# Create env and install
conda create -n reccache python=3.11 -y
conda activate reccache
pip install -e ".[dev]"
pip install pyarrow  # for Amazon parquet datasets
```

Datasets auto-download (MovieLens) or load from `~/DataSets/` (Amazon parquet, MIND).

## Running Experiments

```bash
conda activate reccache

# Full main experiments (S1-S7, 4 datasets, 3 seeds each)
python scripts/run_recsys_experiments.py

# Single experiment group
python scripts/run_recsys_experiments.py --group S4    # End-to-end comparison
python scripts/run_recsys_experiments.py --group S5    # Pool vs Static

# Supplementary experiments
python scripts/run_carl_comparison.py                  # CARL RL comparison
python scripts/run_model_agnostic_experiments.py       # MF vs NeuMF
python scripts/run_significance_tests.py               # Statistical tests
python scripts/run_scalability_experiments.py           # Scalability
python scripts/run_online_simulation.py                # Streaming simulation
python scripts/run_concept_drift.py                    # ε vs regret

# Ablation studies
python scripts/run_ablation_experiments.py             # A1-A4

# Results saved to results/*.json
# Figures saved to paper/figures/
```

## Quick Start

```python
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.utils.data_loader import DataLoader
from collections import defaultdict

# Load data
loader = DataLoader("data")
train, val, test = loader.load_dataset("ml-1m")

# Build user history for exclusion
user_history = defaultdict(list)
for uid, iid in zip(train.user_ids, train.item_ids):
    user_history[int(uid)].append(int(iid))
user_history = dict(user_history)

# Train model
model = MatrixFactorizationRecommender(n_users=train.n_users, n_items=train.n_items)
model.fit(train.user_ids, train.item_ids, train.ratings, epochs=15)

# Setup clustering
item_embs = model.get_all_item_embeddings()
cm = UserClusterManager(n_clusters=50, embedding_dim=64)
cm.set_item_embeddings(item_embs)
cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

# Speculative serving (with pool retrieval)
config = SpeculativeConfig(
    top_k_clusters=3,
    acceptance_threshold=0.35,
    use_pool_retrieval=True,
    pool_size=200,
)
spec = SpeculativeRecommender(
    recommender=model,
    cluster_manager=cm,
    acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=0.35),
    config=config,
    item_embeddings=item_embs,
    user_history=user_history,
)
spec.warm_cache(list(range(train.n_users)))

# Serve
result = spec.recommend(user_id=42)
print(f"Items: {result.items}, Accepted: {result.accepted}, "
      f"Cluster: {result.accepted_cluster_id}, Latency: {result.latency_ms:.2f}ms")
```

## Compiling the Paper

```bash
cd paper/recsys2026
pdflatex main && bibtex main && pdflatex main && pdflatex main
# Output: main.pdf (8 pages, ACM sigconf format)
```

## Competitor Baselines

| Baseline | Type | Notes |
|----------|------|-------|
| Two-Tower (d=16/32) | Retrieve-and-rerank | Standard industry pipeline; beats Spec on dense data |
| FAISS (Flat/IVF) | ANN retrieval | Faster on dense data; Spec wins on sparse (bias-aware) |
| BiLD-style draft | Trained low-rank draft | 55% accept vs RecCache 91% |
| LASER-relaxed | Mean-ratio criterion | 100% accept vs product rule's 91% (lower NDCG) |
| CARL-like (DQN) | RL caching policy | RecCache matches NDCG with zero training |
| Popularity | Global top-N | Highest CTR on ML-1M due to popularity bias |
| Naive cache | Nearest cluster, no verify | No MCG, lower quality |
| NEZHA / AtSpeed | Generative rec decoding | Different target (LLM-based gen rec) |

## Citation

```bibtex
@inproceedings{reccache2026,
  title={Speculative Recommendation Serving: Bridging LLM Inference
         Acceleration and Recommendation Caching},
  author={Anonymous},
  booktitle={Proceedings of the 20th ACM Conference on Recommender Systems},
  year={2026}
}
```

## License

MIT License
