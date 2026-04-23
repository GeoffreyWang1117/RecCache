# Supplementary Materials — Speculative Recommendation Serving

This branch hosts the supplementary materials accompanying the RecSys 2026
submission *"Speculative Recommendation Serving: A Principled Framework for
Cache-Based Recommendation."*

The repository is provided for reproducibility; reviewers are not required
to consult it.

## Layout

```
├── src/reccache/              # Core framework (Python package)
├── scripts/                   # 32 experiment drivers (one per result group)
├── paper/
│   ├── recsys2026/main.tex    # Submitted LaTeX source
│   ├── recsys2026/main.pdf    # Submitted PDF (8 content + 1 ref pages)
│   ├── references.bib         # 31 entries, all verified (see below)
│   └── figures/               # 59 PDF/PNG figures sourced from results
├── results/                   # 29 experiment result JSONs
│   └── gemma_embeddings/
│       └── ml-1m_gemma4_item_embs_pca64.npy   # Cached Gemma-4 embeddings
└── SUPPLEMENTARY.md           # This file
```

## Experiment → result-file → script mapping

| ID | Table / §         | Result JSON                                 | Script                                            |
|----|-------------------|---------------------------------------------|---------------------------------------------------|
| S1 | Table 3           | `recsys2026_results.json`                   | `run_recsys_experiments.py --group S1`            |
| S2 | Table 4           | `recsys2026_results.json`                   | `run_recsys_experiments.py --group S2`            |
| S3 | Table 5           | `recsys2026_results.json`                   | `run_recsys_experiments.py --group S3`            |
| S4 | Table 7           | `recsys2026_results.json`                   | `run_recsys_experiments.py --group S4`            |
| S5 | Table 6           | `recsys2026_results.json`                   | `run_recsys_experiments.py --group S5`            |
| S6 | Table 8           | `recsys2026_results.json`                   | `run_recsys_experiments.py --group S6`            |
| S7 | Table 9           | `recsys2026_results.json`                   | `run_recsys_experiments.py --group S7`            |
| S9 | Competitor §      | `s9_competitor_lowrank.json`                | `run_competitor_baselines.py`                     |
| S10| Competitor §      | `s10_laser_criterion.json`                  | `run_competitor_baselines.py`                     |
| S11| Scale-Out §       | `s11_s12_largescale.json`                   | `run_largescale_experiments.py`                   |
| S12| Scale-Out §       | `s11_s12_largescale.json`                   | `run_largescale_experiments.py`                   |
| S13| GNN backbone §    | `s13_lightgcn.json`                         | `run_lightgcn_experiments.py`                     |
| S14| LLM backbone §    | `s14_gemma4_embeddings.json`                | `run_gemma4_experiments.py`                       |
| S15| Latency scaling § | `s15_gpu_latency.json`                      | `run_gpu_latency.py`                              |
| A1 | Table 10          | `ablation_results.json`                     | `run_ablation_experiments.py`                     |
| A2 | Table 11          | `ablation_results.json`                     | `run_ablation_experiments.py`                     |
| A3 | §                 | `ablation_results.json`                     | `run_ablation_experiments.py`                     |
| A4 | Table 12          | `ablation_results.json`                     | `run_ablation_experiments.py`                     |
| —  | Beyond-NDCG §     | `supp_a1_a2_multimetric.json`               | `run_supplementary_metrics.py`                    |
| —  | Long-tail §       | `supp_a1_a2_multimetric.json`               | `run_supplementary_metrics.py`                    |
| —  | FAISS baseline §  | `supp_a3_faiss.json`                        | `run_faiss_baseline.py`                           |
| —  | ε-bound §         | `supp_a4_cluster_analysis.json`             | `run_cluster_analysis.py`                         |
| B2 | Online Stability §| `supp_b2_extended_replay.json`              | `run_extended_replay.py`                          |
| C1 | Online Serving §  | `supp_online_serving_replay.json`           | `run_online_serving_replay.py`                    |
| C2 | Cache Refresh §   | `supp_cache_refresh.json`                   | `run_cache_refresh.py`                            |
| TT | Competitor §      | `supp_twotower_baseline.json`               | `run_twotower_baseline.py`                        |
|CARL| Table 13          | `carl_comparison_results.json`              | `run_carl_comparison.py`                          |
|Sig | Significance §    | `significance_tests.json`                   | `run_significance_tests.py`                       |

All scripts live under `scripts/`. Outputs are written to `results/*.json`.

## Cached Gemma-4 embeddings

To avoid requiring reviewers to re-run the 9.6 GB Gemma-4-E2B-it encoder,
we ship the PCA-projected 64-dim item embeddings for ML-1M at
`results/gemma_embeddings/ml-1m_gemma4_item_embs_pca64.npy`
(3706 × 64, 928 KB). Loading these reproduces the LLM-backbone paragraph
(Experiment S14) exactly without GPU:

```python
import numpy as np
item_embs = np.load("results/gemma_embeddings/ml-1m_gemma4_item_embs_pca64.npy")
# Feed into speculative pipeline as usual.
```

The full 1536-dim raw embeddings (22 MB) are regenerable via
`python scripts/run_gemma4_experiments.py --dataset ml-1m`.

## Reference validity

The `.bib` (31 entries) has been independently verified with
`bibguard` against DBLP / arXiv / OpenAlex / Crossref / Semantic Scholar:
all entries found in at least one source, **0 FAIL**, TeX
cross-audit clean (no orphan references).

## Environment

Python 3.11 via conda:

```bash
conda create -n reccache python=3.11 -y
conda activate reccache
pip install -e ".[dev]"
pip install pyarrow  # for Amazon parquet datasets
```

Core dependencies: `torch>=2.11`, `numpy>=2.4`, `scikit-learn>=1.8`,
`pandas>=3.0`, `pyarrow`, `faiss-cpu`, `matplotlib`, `seaborn`.

All reported numbers were obtained on a single CPU workstation
(Intel i9-14900HX, 64 GB RAM) with 3-seed averaging (seeds 42–44).

## Reproducing a headline result

```bash
conda activate reccache

# Table 7 (end-to-end)
python scripts/run_recsys_experiments.py --group S4

# Gemma-4 backbone (loads cached embeddings)
python scripts/run_gemma4_experiments.py --dataset ml-1m

# Latency scaling sweep
python scripts/run_gpu_latency.py
```

Fresh result JSONs are written to `results/` and figure sources to
`paper/figures/`. To regenerate the submitted PDF:

```bash
cd paper/recsys2026
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Dataset acquisition

- **MovieLens-1M**: auto-downloaded on first run.
- **Amazon Reviews 2023** (Electronics / Arts / Movies-and-TV): manual
  download from [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/)
  (parquet).
- **MIND-small / MIND-large**: manual download from
  [msnews.github.io](https://msnews.github.io/).

Paths are configured in `src/reccache/utils/config.py`; see
`src/reccache/utils/data_loader.py` for loaders.

## Scope

This supplementary branch mirrors the `main` branch of the source
repository with reproducibility artefacts (result JSONs and cached
embeddings) force-added. It is the version served by the anonymous
4open.science mirror referenced from the paper.
