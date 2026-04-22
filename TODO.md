# RecCache — Research TODO

**Paper**: RecSys 2026 | Abstract: Apr 14 (AoE) | Full paper: Apr 21 (AoE)

---

## Priority 1 — Competitor Baselines (REQUIRED before submission)

> "竞争对手需要作为强基线，最终还是需要正面击败竞争者的"

### S9: BiLD-Style Trained Low-Rank Draft ✅ DONE
- [x] Train MF(d=16) as draft model on ML-1M; implement `LowRankDraftAcceptanceCriterion`
- [x] Results (3 seeds avg): Fresh=0.1597, RecCache=0.1065 (91.2% acc), BiLD d=16=0.1231 (54.9% acc)
- [x] **Key finding**: RecCache 91% accept vs BiLD 55% — 1.7× more caching at only −13pp NDCG retention
- Results: `results/s9_competitor_lowrank.json`, Figure: `paper/figures/s9_competitor_lowrank.pdf`

### S10: LASER-Adapted Relaxed Verification ✅ DONE
- [x] Implemented `LASERRelaxedAcceptanceCriterion` (mean-ratio rule, relaxation param)
- [x] Results (3 seeds avg): ScoreRatio NDCG=0.1065 (91% acc) > LASER-Relaxed 0.1053 (100% acc)
- [x] **Key finding**: Product-ratio (ours) beats mean-ratio (LASER) in NDCG — selective rejection of poor clusters improves average quality vs LASER's blind full-acceptance
- Results: `results/s10_laser_criterion.json`, Figure: `paper/figures/s10_laser_criterion.pdf`

### Paper Integration ✅ DONE
- [x] Added "Competitor Baselines" paragraph to paper with inline S9+S10 numbers
- [x] Fixed 20.6pt overfull hbox (paragraph title shortened)
- [x] Paper compiles: 8 pages, only 1pt + 4.3pt overfull remain (acceptable)

---

## Priority 2 — Final Submission Prep

- [ ] Final proofread — check abstract MCG "49-64%" matches Table 4
- [ ] Fix overfull hbox at line 95 (20.57pt) if possible
- [ ] **Submit abstract by Apr 14 (AoE)**
- [ ] Final paper submission by Apr 21 (AoE)

---

## Priority 2b — Large-Scale & SOTA Experiments (RecSys Scale Gap)

> Target: add ≥1 dataset with 100K+ users; add GNN and LLM-embedding backbone

### S11: MIND-large (Scale Benchmark) ✅ DONE
- [x] 72,366 users × 20,870 articles after 5-core (3M sample, train/dev split)
- [x] Fresh NDCG=0.0240, Spec NDCG=0.0200 (83% retention), MCG=60.3%, speedup=9.45×
- Results: `results/s11_s12_largescale.json`

### S12: Amazon Movies-and-TV (Industrial Scale) ✅ DONE
- [x] 85,992 users × 32,999 items, very sparse → poor base NDCG (0.0006)
- [x] Pool retrieval improves: NDCG 0.0006→0.0015 (+136%), accept 53%, MCG=58.8%
- Results: `results/s11_s12_largescale.json`

### S13: LightGCN Backbone (GNN Model-Agnostic) ✅ DONE
- [x] ML-1M: MF MCG=51.9% vs LightGCN MCG=22.8%; LightGCN fresh NDCG=0.172 > MF 0.156
- [x] A-Electronics: MF MCG=61.8% vs LightGCN MCG=6.5% (LightGCN much sharper clusters)
- [x] **Key finding**: GNN geometry produces tighter clusters → lower MCG but framework still valid
- Results: `results/s13_lightgcn.json`

### S14: Gemma-4-E2B-it as Embedding Backbone (LLM Model-Agnostic) ← KEY ✅ DONE (ML-1M)
- [x] Encoded ML-1M item titles with Gemma-4-E2B-it (3706×1536); cached at `results/gemma_embeddings/`
- [x] PCA 1536→64 (32% variance retained); hybrid mode (Gemma clusters + MF recs)
- [x] Results (3 seeds, K=3, τ=0.35): MF MCG=51.9% → Gemma-4 MCG=63.8%
- [x] **Key finding**: MCG scales with embedding quality — structural property strengthens with richer semantics
- [x] Paper paragraph "LLM Embeddings as Backbone" added; `gemma4_2025` bibtex entry added
- [ ] TODO: Run on MIND-large (add to `run_gemma4_experiments.py --dataset mind-large`)
- Results: `results/s14_gemma4_embeddings.json`

### S15: GPU Latency (Production Speedup Correction) ✅ DONE
- [x] `GPUScorer` wraps MF weights as plain tensors; tiles to max_catalog for benchmark
- [x] Spec latency = 0.14ms (constant); Fresh CPU latency scales linearly with catalog
- [x] **Key finding**: speedup grows from 0.4× (1K items) → 27.6× (100K items)
- [x] Demonstrates speculative serving wins are dominated by catalog size
- Results: `results/s15_gpu_latency.json`

---

## Priority 2c — Reviewer-Critique Supplementary (Apr 8-13, before abstract DDL)

> Reviewer feedback identified gaps. Most "missing" metrics already exist in `metrics.py` —
> just need to be exposed. Strategy: Tier 1 (must), Tier 2 (should), skip SIGMOD-flavored items.

### Tier 1 — Must-do (~1 day, biggest reviewer ROI)

#### A1: Multi-Metric Table (Recall/HR/MRR/MAP/ILD) ✅ DONE
- [x] ML-1M + Amazon-Elec: Spec K=3 retention (NDCG 67%/80%, Recall 62%/79%, HR 76%/84%, MRR 67%/79%, MAP 68%/79%, ILD 98%/91%)
- [x] Paper: "Beyond NDCG" paragraph with inline percentages
- Results: `results/supp_a1_a2_multimetric.json`

#### A2: Long-tail / Popularity-bias / Gini ✅ DONE
- [x] Spec+Pool triples coverage on A-Elec (3.8%→9.7%), reduces Gini 0.996→0.966
- [x] Per-cluster acceptance Gini = 0.262 (fairly balanced)
- [x] Paper: "Long-tail and Fairness" paragraph added
- Results: `results/supp_a1_a2_multimetric.json`, Figure: `paper/figures/supp_popularity_bias.pdf`

#### A3: FAISS / ANN Baseline ✅ DONE
- [x] FAISS Flat: ML-1M NDCG=0.100 at 0.07ms (6.6× vs Fresh); A-Elec NDCG=0.005 (35% retention)
- [x] RecCache NDCG=0.012 >> FAISS 0.005 on A-Elec: full bias-aware scoring preserves quality on sparse data
- [x] Paper: FAISS comparison added to "Competitor Baselines" paragraph
- Results: `results/supp_a3_faiss.json`

#### A4: Cluster Stability + Acceptance-vs-Distance ✅ DONE
- [x] Spearman ρ(ε, α) = -0.655 (p < 1e-180): strongest empirical validation of ε-bound
- [x] Cluster assignment 3.2% stable under 50% history truncation (confirms online adaptation needed)
- [x] Per-cluster acceptance Gini = 0.262 (fair)
- [x] Paper: "Empirical validation of the ε-bound" paragraph added to §3.3
- Results: `results/supp_a4_cluster_analysis.json`
- Figures: `paper/figures/supp_acceptance_vs_distance.pdf`, `supp_cluster_stability.pdf`

### Tier 2 — Should-do (if time after Tier 1)

#### B1: Two-Tower Re-framing ✅ DONE
- [x] Added "structurally identical to a two-tower / dual-encoder retrieval model" to Base Model paragraph
- [x] Preempts "why not compare to two-tower?" reviewer question

#### B2: Log Replay Simulation (extended) ✅ DONE
- [x] 3 random orderings × 1188 requests each
- [x] Cross-seed stability: NDCG σ=0.001, Accept σ=0.5%, MCG σ=0.7% (order-invariant)
- [x] Paper: "Online Stability" paragraph updated with 3-ordering ± results
- Results: `results/supp_b2_extended_replay.json`, Figure: `paper/figures/supp_extended_replay.pdf`

#### B3: Per-cluster Acceptance Histogram ✅ DONE (via A4)
- [x] Already computed in A4: Gini=0.262, histogram in `supp_cluster_stability.pdf`

### Tier 3 — RecSys-context experiments (Apr 13)

#### C1: Online Serving Replay (6-policy comparison) ✅ DONE
- [x] Temporal replay of 1188 ML-1M requests with CTR proxy + novelty
- [x] 6 policies: Fresh, Popularity, FAISS, Naive, Spec K=3, Spec+Pool
- [x] Finding: Popularity highest CTR (0.169) due to ML-1M popularity bias; Spec K=3 comparable to FAISS
- [x] Paper: "Online Serving Replay" paragraph replaces old "Online Stability"
- Results: `results/supp_online_serving_replay.json`, Figure: `paper/figures/supp_online_serving.pdf`

#### C2: Cache Refresh Sensitivity ✅ DONE
- [x] 4 refresh frequencies × 6 temporal windows on ML-1M
- [x] Finding: NDCG varies <0.002, accept stable 92% — re-clustering unnecessary on stable data
- [x] Cluster purity=0.530, rec overlap 0.515-0.552
- [x] Paper: "Cache Refresh Sensitivity" paragraph added
- Results: `results/supp_cache_refresh.json`, Figure: `paper/figures/supp_cache_refresh.pdf`

### Tier 4 — Skip (SIGMOD-flavor, low ROI for RecSys)
- ~~P95/P99 tail latency~~
- ~~Distributed / fault tolerance~~
- ~~Real online A/B test~~

---

## Priority 3 — Post-Submission Extensions (Camera-Ready / Revision)

### Tree Speculation (Hierarchical Clusters)
- [ ] Implement 2-level cluster hierarchy (e.g., K=5 top-level, K=3 per subtree)
- [ ] Expected: higher MCG, lower miss rate on sparse datasets
- [ ] Analogous to EAGLE/SpecInfer's tree verification

### Online Drift Robustness
- [ ] Implement `IncrementalClusterUpdate` triggered by rolling acceptance-rate drop
- [ ] Test on MIND (temporal windows) — measure regret reduction vs static clusters

### Self-Speculative Mode
- [ ] Use MF(d=64) user embedding but cached from previous request as draft
- [ ] No cluster needed; works per-user

---

## Completed

- [x] S1–S7 main experiments (4 datasets, 3 seeds)
- [x] S8 model-agnostic (MF vs NeuMF on ML-1M)
- [x] CARL comparison (DQN RL caching baseline)
- [x] Statistical significance (paired t-test, Wilcoxon)
- [x] Online simulation (1188 streaming requests)
- [x] Concept drift validation
- [x] Scalability experiments
- [x] A1–A4 ablations
- [x] Paper at 8/8 pages with 39 references, compiles cleanly
- [x] LASER (SIGIR 2025) explicitly differentiated in related work
- [x] Analogy scope note added to §3.3
- [x] Model-agnostic paragraph with actual NeuMF results
- [x] Archive stale planning docs → `archive/`
