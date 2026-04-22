#!/usr/bin/env python3
"""
S14 — Gemma-4-E2B-it as Embedding Backbone Experiment.

Uses Gemma-4-E2B-it (2B MoE, ~9.6GB, locally downloaded) to encode item
titles/metadata into 1536-dim embeddings. These LLM embeddings replace the
MF-learned item embeddings in the speculative serving pipeline.

Pipeline:
  1. Encode item titles with Gemma-4-E2B-it (mean-pool last hidden states)
  2. PCA-project to 64-dim for clustering (keeps computational budget)
  3. Keep full 1536-dim for score-ratio acceptance (richer similarity signal)
  4. User embeddings: weighted average of interacted item Gemma embeddings
  5. Run full speculative serving; compare vs MF-embedding baseline

Key claim: speculative serving is embedding-model-agnostic — works with
           LLM-quality semantic embeddings, not just CF-trained embeddings.

Usage:
    conda activate reccache
    python scripts/run_gemma4_experiments.py
    python scripts/run_gemma4_experiments.py --dataset ml-1m
    python scripts/run_gemma4_experiments.py --dataset mind-large
    python scripts/run_gemma4_experiments.py --no-pca  # use full 1536-dim
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reccache.utils.data_loader import DataLoader
from reccache.models.recommender import MatrixFactorizationRecommender
from reccache.clustering.user_cluster import UserClusterManager
from reccache.models.acceptance import ScoreRatioAcceptanceCriterion
from reccache.models.speculative import SpeculativeRecommender, SpeculativeConfig, SpeculativeResult
from reccache.evaluation.metrics import RecommendationMetrics, compute_coverage

EXTERNAL_DATA_DIR = Path.home() / "DataSets"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
GEMMA_EMBED_DIM = 1536
CLUSTER_DIM = 64    # PCA projection for clustering
BATCH_SIZE = 32     # encoding batch size

DATASET_CONFIGS = {
    "ml-1m": {
        "max_samples": None, "min_user": 5, "min_item": 5, "min_rating_gt": 4.0,
        "n_clusters": 50, "embedding_dim": 64, "epochs": 15,
        "n_test_users": 500,
        "metadata_fn": None,  # built from movies.dat
    },
    "mind-large": {
        "max_samples": 300_000, "min_user": 5, "min_item": 5, "min_rating_gt": 0.5,
        "n_clusters": 100, "embedding_dim": 64, "epochs": 15,
        "n_test_users": 1000,
        "metadata_fn": None,
    },
}

TOP_K = 3
THRESHOLD = 0.35
N_RECS = 10
N_RUNS = 3


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------
def load_ml1m_titles(data_dir: Path) -> dict:
    """Load MovieLens-1M movie titles. Returns {item_id: title_string}."""
    movies_path = data_dir / "ml-1m" / "movies.dat"
    if not movies_path.exists():
        # Try auto-download location
        movies_path = data_dir / "movielens-1m" / "movies.dat"
    if not movies_path.exists():
        # Try listing where ml-1m data is
        for p in data_dir.rglob("movies.dat"):
            movies_path = p
            break

    titles = {}
    if movies_path.exists():
        with open(movies_path, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 3:
                    iid = int(parts[0])
                    title = parts[1]
                    genre = parts[2].replace("|", " ")
                    titles[iid] = f"{title} [{genre}]"
    else:
        print(f"  [WARN] movies.dat not found at {movies_path}; using item IDs as titles")
    return titles


def load_mind_titles(mind_data_dir: Path) -> dict:
    """Load MIND news titles from behaviors / news.tsv. Returns {item_id: title}."""
    import zipfile, csv, io

    titles = {}
    # Try extracted directory first
    extracted = mind_data_dir / "extracted"
    news_tsv = None
    for p in ["MINDlarge_train/news.tsv", "MINDsmall_train/news.tsv"]:
        candidate = extracted / p
        if candidate.exists():
            news_tsv = candidate
            break

    if news_tsv is None:
        # Try extracting from zip inline
        for zip_name in ["MINDlarge_train.zip", "MINDsmall_train.zip"]:
            zp = mind_data_dir / zip_name
            if zp.exists():
                try:
                    with zipfile.ZipFile(zp) as z:
                        names = z.namelist()
                        tsv_name = next((n for n in names if "news.tsv" in n), None)
                        if tsv_name:
                            data = z.read(tsv_name).decode("utf-8")
                            reader = csv.reader(io.StringIO(data), delimiter="\t")
                            for row in reader:
                                if len(row) >= 4:
                                    # Format: news_id, cat, subcat, title, abstract, ...
                                    titles[row[0]] = f"{row[3]} [{row[1]} {row[2]}]"
                            print(f"  Loaded {len(titles)} MIND news titles from {zip_name}")
                            return titles
                except Exception as e:
                    print(f"  [WARN] Failed to read {zip_name}: {e}")

    if news_tsv and news_tsv.exists():
        with open(news_tsv, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 4:
                    titles[row[0]] = f"{row[3]} [{row[1]} {row[2]}]"
        print(f"  Loaded {len(titles)} MIND news titles")

    return titles


# ---------------------------------------------------------------------------
# Gemma-4 embedding
# ---------------------------------------------------------------------------
class Gemma4Embedder:
    """Encodes text strings into Gemma-4-E2B-it mean-pooled embeddings."""

    def __init__(self, model_id: str = GEMMA_MODEL_ID, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModel
        print(f"  Loading {model_id} for embedding...")
        # Normalize to explicit device string (cuda → cuda:0)
        if device == "cuda":
            device = "cuda:0"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Pin to single GPU to avoid cross-device tensor errors with 2×3090
        device_map = {"": device} if device.startswith("cuda") else "cpu"
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
        print(f"  Gemma-4-E2B-it loaded on {device}")

    @torch.no_grad()
    def encode(self, texts: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Encode list of strings → (N, 1536) float32 embeddings via mean pooling."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=False)
            # Mean-pool last hidden state over non-padding tokens
            last = out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            emb = (last.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-8)  # (B, H)
            all_embs.append(emb.cpu().float().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"    Encoded {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.vstack(all_embs)  # (N, 1536)


# ---------------------------------------------------------------------------
# Item embedding construction
# ---------------------------------------------------------------------------
def build_gemma_item_embeddings(
    item_ids: list,        # contiguous 0..n_items-1
    id_to_title: dict,     # item_id -> title string
    embedder: Gemma4Embedder,
    cache_path: Path,
) -> np.ndarray:
    """Build or load cached Gemma-4 item embeddings."""
    if cache_path.exists():
        print(f"  Loading cached Gemma embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"  Encoding {len(item_ids)} items with Gemma-4...")
    texts = [id_to_title.get(iid, f"item {iid}") for iid in item_ids]
    embs = embedder.encode(texts)

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(cache_path, embs)
    print(f"  Saved Gemma item embeddings → {cache_path}  shape={embs.shape}")
    return embs


def pca_project(embs: np.ndarray, n_components: int = CLUSTER_DIM) -> tuple:
    """PCA-project to n_components dimensions. Returns (projected, pca_obj)."""
    from sklearn.decomposition import PCA
    print(f"  PCA: {embs.shape[1]}d → {n_components}d...")
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(embs).astype(np.float32)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance retained: {var:.1%}")
    return projected, pca


# ---------------------------------------------------------------------------
# Gemma-based recommender (zero-shot, no training)
# ---------------------------------------------------------------------------
class GemmaRecommender:
    """
    Zero-shot recommender using Gemma-4 item embeddings.

    User embedding = mean of interacted item embeddings (weighted by rating if available).
    Score = user_emb @ item_emb.  No training required.
    """

    def __init__(self, item_embeddings: np.ndarray, n_users: int, n_items: int):
        self.item_embeddings = item_embeddings  # (n_items, dim)
        self.n_users = n_users
        self.n_items = n_items
        self._user_embs: dict = {}

    def build_user_embeddings(self, user_ids, item_ids, ratings):
        """Compute user embeddings from interaction history."""
        from collections import defaultdict
        user_items = defaultdict(list)
        user_ratings = defaultdict(list)
        for uid, iid, r in zip(user_ids, item_ids, ratings):
            uid, iid = int(uid), int(iid)
            if iid < len(self.item_embeddings):
                user_items[uid].append(iid)
                user_ratings[uid].append(float(r))

        for uid, items in user_items.items():
            wts = np.array(user_ratings[uid], dtype=np.float32)
            wts = wts / (wts.sum() + 1e-8)
            emb = (self.item_embeddings[items] * wts[:, None]).sum(0)
            norm = np.linalg.norm(emb)
            self._user_embs[uid] = emb / (norm + 1e-8)

        print(f"  Built Gemma user embeddings for {len(self._user_embs)} users")

    def recommend(self, user_id: int, n: int = 10,
                  exclude_items=None) -> list:
        user_emb = self._user_embs.get(user_id)
        if user_emb is None:
            return list(range(n))
        scores = self.item_embeddings @ user_emb
        if exclude_items:
            for iid in exclude_items:
                if iid < len(scores):
                    scores[iid] = -np.inf
        return np.argsort(-scores)[:n].tolist()

    def get_all_item_embeddings(self) -> np.ndarray:
        return self.item_embeddings

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        return self._user_embs.get(user_id,
               np.zeros(self.item_embeddings.shape[1], dtype=np.float32))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def aggregate(results, test_lookup, n_items, n_recs=10):
    ndcgs, hrs, lats, accepts, ranks = [], [], [], [], []
    for r in results:
        gt = set(test_lookup.get(r.user_id, []))
        if not gt: continue
        ndcgs.append(RecommendationMetrics.ndcg_at_k(r.items, gt, k=n_recs))
        hrs.append(RecommendationMetrics.hit_rate(r.items, gt, k=n_recs))
        lats.append(r.latency_ms)
        accepts.append(int(r.accepted))
        if r.accepted: ranks.append(r.accepted_cluster_rank)
    accept_rate = float(np.mean(accepts)) if accepts else 0.0
    mcg = float(np.mean([v > 0 for v in ranks])) if ranks else 0.0
    cov = compute_coverage({r.user_id: r.items for r in results}, n_items)
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hr": float(np.mean(hrs)) if hrs else 0.0,
        "accept_rate": accept_rate,
        "mean_latency_ms": float(np.mean(lats)) if lats else 1.0,
        "mcg": mcg, "coverage": cov, "n": len(ndcgs),
    }


def run_speculative_with_embs(
    model, cm, item_embs, test_users, user_history, train,
    threshold=THRESHOLD, top_k=TOP_K, n_recs=N_RECS,
):
    config = SpeculativeConfig(
        top_k_clusters=top_k, acceptance_threshold=threshold,
        n_recs=n_recs, use_pool_retrieval=False,
    )
    spec = SpeculativeRecommender(
        recommender=model, cluster_manager=cm,
        acceptance_criterion=ScoreRatioAcceptanceCriterion(threshold=threshold),
        config=config, item_embeddings=item_embs, user_history=user_history,
    )
    spec.warm_cache(list(range(train.n_users)))
    return [spec.recommend(uid) for uid in test_users]


def run_one_seed(dataset_name, cfg, seed, embedder, emb_cache_dir, use_pca=True):
    print(f"\n  seed={seed}")
    set_seed(seed)

    loader = DataLoader("data", external_data_dir=str(EXTERNAL_DATA_DIR))
    train, val, test = loader.load_dataset(
        dataset_name,
        max_samples=cfg["max_samples"],
        min_user_interactions=cfg["min_user"],
        min_item_interactions=cfg["min_item"],
    )
    user_history = defaultdict(list)
    for uid, iid in zip(train.user_ids, train.item_ids):
        user_history[int(uid)].append(int(iid))
    min_r = cfg["min_rating_gt"]
    test_lookup = defaultdict(list)
    for uid, iid, r in zip(test.user_ids, test.item_ids, test.ratings):
        if r >= min_r:
            test_lookup[int(uid)].append(int(iid))

    n_users, n_items = train.n_users, train.n_items
    print(f"    {n_users} users, {n_items} items")

    # Test user selection (same for all methods → fair comparison)
    rng = np.random.default_rng(seed)
    test_uid_pool = [u for u, items in test_lookup.items() if len(items) > 0]
    test_users = rng.choice(test_uid_pool, size=min(cfg["n_test_users"], len(test_uid_pool)),
                            replace=False).tolist()

    # ---- MF (standard, learned embeddings) ----
    print("    Training MF (d=64)...")
    mf = MatrixFactorizationRecommender(n_users=n_users, n_items=n_items,
                                        embedding_dim=cfg["embedding_dim"])
    mf.fit(train.user_ids, train.item_ids, train.ratings, epochs=cfg["epochs"])
    mf_item_embs = mf.get_all_item_embeddings()

    mf_cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=cfg["embedding_dim"])
    mf_cm.set_item_embeddings(mf_item_embs)
    mf_cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    mf_fresh = []
    for uid in test_users:
        exc = user_history.get(uid)
        t0 = time.time()
        items = list(mf.recommend(uid, n=N_RECS, exclude_items=exc))
        mf_fresh.append(SpeculativeResult(
            user_id=uid, items=items, accepted=False, acceptance_prob=0.0,
            accepted_cluster_id=None, accepted_cluster_rank=-1,
            latency_ms=(time.time()-t0)*1000, phase="fresh"
        ))
    mf_fresh_m = aggregate(mf_fresh, test_lookup, n_items)
    mf_spec_r  = run_speculative_with_embs(mf, mf_cm, mf_item_embs, test_users, user_history, train)
    mf_spec_m  = aggregate(mf_spec_r, test_lookup, n_items)
    fn = mf_fresh_m["ndcg"]
    mf_spec_m["retention"] = mf_spec_m["ndcg"] / fn if fn > 0 else 0
    print(f"    MF:    fresh={fn:.4f}, spec={mf_spec_m['ndcg']:.4f} "
          f"(ret={mf_spec_m['retention']:.0%}), acc={mf_spec_m['accept_rate']:.0%}, "
          f"MCG={mf_spec_m['mcg']:.0%}")

    # ---- Gemma-4 embeddings ----
    # Load item titles
    if dataset_name == "ml-1m":
        id_to_title = load_ml1m_titles(Path("data"))
    elif "mind" in dataset_name:
        id_to_title = load_mind_titles(EXTERNAL_DATA_DIR / "MIND")
    else:
        id_to_title = {}

    # Build Gemma embeddings (cached per dataset, not per seed)
    cache_path = emb_cache_dir / f"{dataset_name}_gemma4_item_embs.npy"
    item_ids_list = list(range(n_items))
    gemma_embs = build_gemma_item_embeddings(item_ids_list, id_to_title, embedder, cache_path)

    # If item count changed between seeds (shouldn't), truncate/pad
    if len(gemma_embs) != n_items:
        print(f"  [WARN] gemma_embs shape {gemma_embs.shape} vs n_items {n_items}; re-encoding")
        cache_path.unlink(missing_ok=True)
        gemma_embs = build_gemma_item_embeddings(item_ids_list, id_to_title, embedder, cache_path)

    # PCA project for clustering (64-dim)
    if use_pca:
        pca_cache = emb_cache_dir / f"{dataset_name}_gemma4_item_embs_pca{CLUSTER_DIM}.npy"
        if pca_cache.exists():
            gemma_embs_cluster = np.load(pca_cache)
        else:
            gemma_embs_cluster, _ = pca_project(gemma_embs, CLUSTER_DIM)
            np.save(pca_cache, gemma_embs_cluster)
    else:
        gemma_embs_cluster = gemma_embs  # use full 1536-dim

    cluster_dim = gemma_embs_cluster.shape[1]

    # Build cluster manager with Gemma PCA embeddings
    gemma_cm = UserClusterManager(n_clusters=cfg["n_clusters"], embedding_dim=cluster_dim)
    gemma_cm.set_item_embeddings(gemma_embs_cluster)
    # User embeddings: weighted avg of Gemma PCA item embeddings
    gemma_cm.initialize_from_interactions(train.user_ids, train.item_ids, train.ratings)

    # Speculative serving with Gemma embeddings (acceptance uses PCA embs)
    gemma_spec_r = run_speculative_with_embs(
        mf, gemma_cm, gemma_embs_cluster, test_users, user_history, train
    )
    gemma_spec_m = aggregate(gemma_spec_r, test_lookup, n_items)
    gemma_spec_m["retention"] = gemma_spec_m["ndcg"] / fn if fn > 0 else 0
    print(f"    Gemma: spec={gemma_spec_m['ndcg']:.4f} "
          f"(ret={gemma_spec_m['retention']:.0%}), acc={gemma_spec_m['accept_rate']:.0%}, "
          f"MCG={gemma_spec_m['mcg']:.0%}")

    # ---- Pure Gemma recommender (zero-shot, no MF) ----
    # Items scored by dot-product in Gemma-4 embedding space.
    # This tests whether the speculative framework works end-to-end with LLM embeddings.
    print("    Building pure Gemma recommender (zero-shot)...")
    gemma_rec = GemmaRecommender(gemma_embs_cluster, n_users=n_users, n_items=n_items)
    gemma_rec.build_user_embeddings(train.user_ids, train.item_ids, train.ratings)

    # Fresh Gemma recommendations (zero-shot baseline)
    gemma_fresh = []
    for uid in test_users:
        exc = user_history.get(uid)
        t0 = time.time()
        items = list(gemma_rec.recommend(uid, n=N_RECS, exclude_items=exc))
        gemma_fresh.append(SpeculativeResult(
            user_id=uid, items=items, accepted=False, acceptance_prob=0.0,
            accepted_cluster_id=None, accepted_cluster_rank=-1,
            latency_ms=(time.time()-t0)*1000, phase="fresh"
        ))
    gemma_fresh_m = aggregate(gemma_fresh, test_lookup, n_items)
    fn_g = gemma_fresh_m["ndcg"]
    print(f"    Gemma fresh: NDCG={fn_g:.4f} (zero-shot, no training)")

    # Pure Gemma speculative: cluster+recommend+verify all in Gemma space
    gemma_rec_spec_r = run_speculative_with_embs(
        gemma_rec, gemma_cm, gemma_embs_cluster, test_users, user_history, train
    )
    gemma_rec_spec_m = aggregate(gemma_rec_spec_r, test_lookup, n_items)
    gemma_rec_spec_m["retention"] = gemma_rec_spec_m["ndcg"] / fn_g if fn_g > 0 else 0
    print(f"    Gemma spec:  NDCG={gemma_rec_spec_m['ndcg']:.4f} "
          f"(ret={gemma_rec_spec_m['retention']:.0%}), "
          f"acc={gemma_rec_spec_m['accept_rate']:.0%}, MCG={gemma_rec_spec_m['mcg']:.0%}")

    return {
        "mf_fresh":          mf_fresh_m,
        "mf_spec":           mf_spec_m,
        "gemma_hybrid_spec": gemma_spec_m,    # Gemma clusters + MF recommender
        "gemma_fresh":       gemma_fresh_m,   # Gemma zero-shot fresh
        "gemma_pure_spec":   gemma_rec_spec_m, # Gemma clusters + Gemma recommender
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["ml-1m"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pca", action="store_true", help="Use full 1536-dim (slower)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="paper/figures")
    parser.add_argument("--emb-cache-dir", default="results/gemma_embeddings")
    args = parser.parse_args()

    results_dir  = Path(args.results_dir);  results_dir.mkdir(exist_ok=True)
    figures_dir  = Path(args.figures_dir);  figures_dir.mkdir(exist_ok=True, parents=True)
    emb_cache    = Path(args.emb_cache_dir); emb_cache.mkdir(exist_ok=True, parents=True)
    use_pca      = not args.no_pca

    print(f"Device: {args.device}")
    embedder = Gemma4Embedder(model_id=GEMMA_MODEL_ID, device=args.device)

    all_results = {}
    for ds in args.dataset:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n{'='*60}\nS14: Gemma-4 Embeddings | {ds}\n{'='*60}")

        seed_results = []
        for seed in args.seeds:
            seed_results.append(run_one_seed(ds, cfg, seed, embedder, emb_cache, use_pca))

        def avg_key(method, key):
            return float(np.mean([r[method].get(key, 0) for r in seed_results]))

        methods = ["mf_fresh", "mf_spec", "gemma_hybrid_spec", "gemma_fresh", "gemma_pure_spec"]
        keys = list(seed_results[0]["mf_fresh"].keys())
        averaged = {m: {k: avg_key(m, k) for k in keys} for m in methods}
        all_results[ds] = averaged

        fn_mf = averaged["mf_fresh"]["ndcg"]
        fn_g  = averaged["gemma_fresh"]["ndcg"]
        print(f"\n  {ds} Summary:")
        print(f"  {'Method':<36} {'NDCG':>8} {'Ret':>7} {'Accept':>8} {'MCG':>6}")
        print("  " + "-" * 70)
        rows = [
            ("MF Fresh",                 "mf_fresh",          fn_mf),
            ("MF Spec K=3",              "mf_spec",           fn_mf),
            ("Gemma-4 Hybrid Spec",      "gemma_hybrid_spec", fn_mf),
            ("Gemma-4 Fresh (zero-shot)","gemma_fresh",       fn_g),
            ("Gemma-4 Pure Spec",        "gemma_pure_spec",   fn_g),
        ]
        for label, key, base in rows:
            m   = averaged[key]
            ret = m["ndcg"]/base if base>0 and "fresh" not in key else 1.0
            acc = m.get("accept_rate",0) if "fresh" not in key else float("nan")
            mcg = m.get("mcg",0) if "fresh" not in key else float("nan")
            if "fresh" in key:
                print(f"  {label:<36} {m['ndcg']:>8.4f} {'—':>7} {'—':>8} {'—':>6}")
            else:
                print(f"  {label:<36} {m['ndcg']:>8.4f} {ret:>7.1%} {acc:>8.1%} {mcg:>6.1%}")

    out = results_dir / "s14_gemma4_embeddings.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    # Figure
    if all_results:
        datasets = list(all_results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        x = np.arange(len(datasets)); w = 0.2
        plot_methods = [
            ("MF Spec",              "mf_spec"),
            ("Gemma-4 Hybrid Spec",  "gemma_hybrid_spec"),
            ("Gemma-4 Pure Spec",    "gemma_pure_spec"),
        ]
        for ax, metric in zip(axes, ["ndcg","accept_rate","mcg"]):
            for i, (label, key) in enumerate(plot_methods):
                vals = [all_results[ds][key].get(metric,0) for ds in datasets]
                ax.bar(x + (i-1)*w, vals, w, label=label)
            ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=15)
            ax.set_ylabel(metric); ax.legend(fontsize=8)
            ax.set_title(f"S14: {metric} — MF vs Gemma-4")
        plt.tight_layout()
        fig_path = figures_dir / "s14_gemma4.pdf"
        plt.savefig(fig_path, bbox_inches="tight"); plt.close()
        print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
