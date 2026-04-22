# Supplementary Materials — Speculative Recommendation Serving

This branch hosts the supplementary materials accompanying the RecSys 2026
submission *"Speculative Recommendation Serving: A Principled Framework for
Cache-Based Recommendation."*

The repository is provided for reproducibility; reviewers are not required
to consult it.

## Contents

The branch will be populated incrementally with the following materials.
Items marked `[pending]` will be added prior to the camera-ready deadline.

### Code
- `src/reccache/` — core framework (speculative pipeline, acceptance
  criteria, clustering, embedding pool retrieval, cache manager).
- `scripts/` — experiment drivers for every result reported in the paper,
  organised by experiment ID (`run_recsys_experiments.py`,
  `run_ablation_experiments.py`, `run_competitor_baselines.py`,
  `run_largescale_experiments.py`, `run_lightgcn_experiments.py`,
  `run_gemma4_experiments.py`, `run_gpu_latency.py`,
  `run_faiss_baseline.py`, `run_cluster_analysis.py`,
  `run_twotower_baseline.py`, `run_online_serving_replay.py`,
  `run_cache_refresh.py`, `run_carl_comparison.py`,
  `run_significance_tests.py`, `run_extended_replay.py`).

### Results `[pending]`
- Per-experiment JSON result files under `results/` covering all tables
  and figures in the paper (S1–S15, A1–A4, B2, C1–C2).

### Cached artefacts `[pending]`
- Gemma-4-derived item embeddings (PCA-projected, 64-dim) under
  `results/gemma_embeddings/` to reproduce the LLM-backbone experiment
  without re-running the encoder.

### Dataset preparation `[pending]`
- Preprocessing notes for MovieLens-1M, Amazon Electronics / Arts /
  Movies-and-TV, and MIND-small / MIND-large, including 5-core
  filtering and temporal splits.

### Figures `[pending]`
- Source scripts regenerating every figure from the corresponding JSON
  result file.

## Environment

Dedicated conda environment (Python 3.11):

```bash
conda create -n reccache python=3.11 -y
conda activate reccache
pip install -e ".[dev]"
pip install pyarrow  # Amazon parquet datasets
```

Core dependencies: `torch`, `numpy`, `scikit-learn`, `pandas`,
`pyarrow`, `faiss-cpu`, `matplotlib`, `seaborn`.

All reported numbers were obtained on CPU (single workstation) with
3-seed averaging (seeds 42–44).

## Reproducing a headline result

```bash
conda activate reccache

# Table 4 (end-to-end, K=3, tau=0.35)
python scripts/run_recsys_experiments.py --group S4

# Table for Gemma-4 embedding backbone
python scripts/run_gemma4_experiments.py --dataset ml-1m

# Latency scaling (Figure on catalog-size sweep)
python scripts/run_gpu_latency.py
```

Result JSON files are written to `results/` and figure sources to
`paper/figures/`.

## Anonymisation

This branch is mirrored to an anonymous read-only snapshot for review.
Please refer to the anonymous URL provided in the submission form; do
not follow any links from this branch's git history.
