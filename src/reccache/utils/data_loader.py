"""Data loading utilities for recommendation datasets."""

import os
import gzip
import json
import zipfile
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import urllib.request

import numpy as np
import pandas as pd


@dataclass
class InteractionData:
    """Container for user-item interaction data."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    ratings: np.ndarray
    timestamps: Optional[np.ndarray] = None

    n_users: int = 0
    n_items: int = 0

    # Mappings
    user_id_map: dict = None  # original_id -> internal_id
    item_id_map: dict = None  # original_id -> internal_id
    reverse_user_map: dict = None  # internal_id -> original_id
    reverse_item_map: dict = None  # internal_id -> original_id

    def __post_init__(self):
        if self.n_users == 0:
            self.n_users = len(np.unique(self.user_ids))
        if self.n_items == 0:
            self.n_items = len(np.unique(self.item_ids))

    def get_user_items(self, user_id: int) -> List[int]:
        """Get all items interacted by a user."""
        mask = self.user_ids == user_id
        return self.item_ids[mask].tolist()

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        n_interactions = len(self.user_ids)
        sparsity = 1 - n_interactions / (self.n_users * self.n_items)
        avg_items_per_user = n_interactions / self.n_users
        avg_users_per_item = n_interactions / self.n_items

        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": n_interactions,
            "sparsity": sparsity,
            "avg_items_per_user": avg_items_per_user,
            "avg_users_per_item": avg_users_per_item,
            "rating_mean": float(np.mean(self.ratings)),
            "rating_std": float(np.std(self.ratings)),
        }


# Amazon 2023 category name mapping: short name -> parquet file prefix
AMAZON_CATEGORIES = {
    "beauty": "All_Beauty",
    "books": "Books",
    "electronics": "Electronics",
    "movies": "Movies_and_TV",
    "home": "Home_and_Kitchen",
    "sports": "Sports_and_Outdoors",
    "toys": "Toys_and_Games",
    "automotive": "Automotive",
    "arts": "Arts_Crafts_and_Sewing",
    "office": "Office_Products",
}


class DataLoader:
    """Load and preprocess recommendation datasets."""

    # Dataset URLs (for auto-download)
    DATASET_URLS = {
        "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    }

    SUPPORTED_DATASETS = [
        "ml-100k",
        "ml-1m",
        # Amazon 2023 (parquet format from ~/DataSets/amazon/)
        "amazon-beauty",
        "amazon-books",
        "amazon-electronics",
        "amazon-movies",
        "amazon-home",
        "amazon-sports",
        "amazon-toys",
        "amazon-automotive",
        "amazon-arts",
        "amazon-office",
        # MIND news recommendation
        "mind-small",
        "mind-large",
        # Require manual download
        "yelp",
        "gowalla",
        "lastfm",
    ]

    def __init__(
        self,
        data_dir: str = "data",
        external_data_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Project-local data directory for MovieLens etc.
            external_data_dir: External datasets directory (e.g. ~/DataSets).
                             Auto-detected from ~/DataSets if not specified.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if external_data_dir is not None:
            self.external_data_dir = Path(external_data_dir)
        else:
            # Auto-detect ~/DataSets
            candidate = Path.home() / "DataSets"
            self.external_data_dir = candidate if candidate.exists() else None

    def load_dataset(
        self,
        name: str,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        max_samples: Optional[int] = None,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """
        Load dataset by name.

        Args:
            name: Dataset name (ml-100k, ml-1m, amazon-beauty, mind-small, etc.)
            min_user_interactions: Minimum interactions per user (k-core filtering)
            min_item_interactions: Minimum interactions per item
            max_samples: Maximum number of interactions to load (for large datasets)

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        name = name.lower().strip()

        if name == "ml-100k":
            return self.load_movielens_100k()
        elif name == "ml-1m":
            return self.load_movielens_1m()
        elif name.startswith("amazon-"):
            category = name.split("-", 1)[1]
            return self.load_amazon(
                category, min_user_interactions, min_item_interactions, max_samples
            )
        elif name.startswith("mind"):
            size = "small" if "small" in name else "large"
            return self.load_mind(
                size, min_user_interactions, min_item_interactions, max_samples
            )
        elif name == "yelp":
            return self.load_yelp(min_user_interactions, min_item_interactions)
        elif name == "gowalla":
            return self.load_gowalla(min_user_interactions, min_item_interactions)
        elif name == "lastfm":
            return self.load_lastfm(min_user_interactions, min_item_interactions)
        else:
            raise ValueError(
                f"Unknown dataset: {name}. Supported: {self.SUPPORTED_DATASETS}"
            )

    # ------------------------------------------------------------------ #
    #  MovieLens
    # ------------------------------------------------------------------ #

    def load_movielens_100k(
        self, min_rating: float = 0.0
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """Load MovieLens 100K dataset."""
        ml_dir = self.data_dir / "ml-100k"

        if not ml_dir.exists():
            self._download_and_extract("ml-100k")

        ratings_file = ml_dir / "u.data"
        df = pd.read_csv(
            ratings_file,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        if min_rating > 0:
            df = df[df["rating"] >= min_rating]

        return self._process_dataframe(df)

    def load_movielens_1m(
        self, min_rating: float = 0.0
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """Load MovieLens 1M dataset."""
        ml_dir = self.data_dir / "ml-1m"

        if not ml_dir.exists():
            self._download_and_extract("ml-1m")

        ratings_file = ml_dir / "ratings.dat"
        df = pd.read_csv(
            ratings_file,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )

        if min_rating > 0:
            df = df[df["rating"] >= min_rating]

        return self._process_dataframe(df)

    # ------------------------------------------------------------------ #
    #  Amazon 2023 (parquet format)
    # ------------------------------------------------------------------ #

    def load_amazon(
        self,
        category: str = "beauty",
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        max_samples: Optional[int] = None,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """
        Load Amazon Reviews 2023 dataset (parquet format).

        Looks for files in: external_data_dir/amazon/{Category}_reviews.parquet
        Falls back to: data_dir/amazon/{category}.json.gz (legacy format)

        Args:
            category: Short name (beauty, books, electronics, movies, etc.)
            min_user_interactions: k-core filtering threshold for users
            min_item_interactions: k-core filtering threshold for items
            max_samples: Max interactions to load (sampled by recency)
        """
        category = category.lower()

        # Try parquet format first (Amazon 2023)
        parquet_prefix = AMAZON_CATEGORIES.get(category)
        if parquet_prefix and self.external_data_dir is not None:
            parquet_path = (
                self.external_data_dir / "amazon" / f"{parquet_prefix}_reviews.parquet"
            )
            if parquet_path.exists():
                return self._load_amazon_parquet(
                    parquet_path,
                    min_user_interactions,
                    min_item_interactions,
                    max_samples,
                )

        # Try legacy json.gz format in project data dir
        legacy_path = self.data_dir / "amazon" / f"{category}.json.gz"
        if legacy_path.exists():
            return self._load_amazon_json_gz(
                legacy_path, min_user_interactions, min_item_interactions
            )

        # Nothing found
        search_paths = []
        if parquet_prefix and self.external_data_dir is not None:
            search_paths.append(
                str(self.external_data_dir / "amazon" / f"{parquet_prefix}_reviews.parquet")
            )
        search_paths.append(str(legacy_path))

        raise FileNotFoundError(
            f"Amazon '{category}' dataset not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
            + "\n\nDownload Amazon Reviews 2023 from: "
            "https://amazon-reviews-2023.github.io/"
        )

    def _load_amazon_parquet(
        self,
        path: Path,
        min_user_interactions: int,
        min_item_interactions: int,
        max_samples: Optional[int],
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """Load Amazon reviews from parquet file (2023 format)."""
        print(f"Loading Amazon parquet: {path.name}")

        # Read only needed columns to save memory
        pf = pd.read_parquet(path, columns=None)
        cols = pf.columns.tolist()

        # Determine item column: 'item_id' (Books) or 'parent_asin' (others)
        if "item_id" in cols:
            item_col = "item_id"
        elif "parent_asin" in cols:
            item_col = "parent_asin"
        elif "asin" in cols:
            item_col = "asin"
        else:
            raise ValueError(f"Cannot determine item column from: {cols}")

        # Select and rename columns
        needed = ["user_id", item_col, "rating", "timestamp"]
        available = [c for c in needed if c in cols]
        df = pf[available].copy()
        if item_col != "item_id":
            df = df.rename(columns={item_col: "item_id"})

        # Ensure timestamp is in seconds (some files use milliseconds)
        if "timestamp" in df.columns and df["timestamp"].max() > 1e12:
            df["timestamp"] = df["timestamp"] // 1000

        del pf  # free memory

        print(f"  Raw interactions: {len(df):,}")

        # Sample if dataset is too large
        if max_samples is not None and len(df) > max_samples:
            # Keep most recent interactions
            df = df.nlargest(max_samples, "timestamp")
            print(f"  Sampled to: {len(df):,} (most recent)")

        # Remove duplicates
        df = df.drop_duplicates(subset=["user_id", "item_id"])

        # k-core filtering
        df = self._filter_k_core(df, min_user_interactions, min_item_interactions)
        print(f"  After k-core filtering: {len(df):,}")

        return self._process_dataframe(df)

    def _load_amazon_json_gz(
        self,
        path: Path,
        min_user_interactions: int,
        min_item_interactions: int,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """Load Amazon reviews from legacy json.gz format."""
        print(f"Loading Amazon json.gz: {path.name}")

        reviews = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                review = json.loads(line)
                if "overall" in review and "reviewerID" in review and "asin" in review:
                    reviews.append(
                        {
                            "user_id": review["reviewerID"],
                            "item_id": review["asin"],
                            "rating": float(review["overall"]),
                            "timestamp": review.get("unixReviewTime", 0),
                        }
                    )

        df = pd.DataFrame(reviews)
        df = self._filter_k_core(df, min_user_interactions, min_item_interactions)

        return self._process_dataframe(df)

    # ------------------------------------------------------------------ #
    #  MIND (Microsoft News Dataset)
    # ------------------------------------------------------------------ #

    def load_mind(
        self,
        size: str = "small",
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        max_samples: Optional[int] = None,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """
        Load MIND (Microsoft News Dataset).

        Converts click behaviors to implicit user-item interactions.
        Looks in: external_data_dir/MIND/

        Args:
            size: "small" or "large"
            min_user_interactions: k-core filtering threshold
            min_item_interactions: k-core filtering threshold
            max_samples: Max interactions to load
        """
        if self.external_data_dir is None:
            raise FileNotFoundError(
                "MIND dataset requires external_data_dir. "
                "Set external_data_dir or place data in ~/DataSets/MIND/"
            )

        mind_dir = self.external_data_dir / "MIND"
        if not mind_dir.exists():
            raise FileNotFoundError(
                f"MIND directory not found at {mind_dir}. "
                "Download from: https://msnews.github.io/"
            )

        size_label = "small" if size == "small" else "large"

        # Collect interactions from train (and dev for small).
        # Pass max_samples to each split to avoid loading the full ~90M-row file.
        all_interactions = []
        # Allocate 80% of budget to train, 20% to dev (train is ~6× larger)
        per_split_max = {
            "train": int(max_samples * 0.8) if max_samples else None,
            "dev":   int(max_samples * 0.2) if max_samples else None,
        }

        for split in ["train", "dev"]:
            zip_name = f"MIND{size_label}_{split}.zip"
            zip_path = mind_dir / zip_name
            if not zip_path.exists():
                if split == "train":
                    raise FileNotFoundError(
                        f"MIND {size_label} train zip not found: {zip_path}"
                    )
                continue

            prefix = f"MIND{size_label}_{split}"
            interactions = self._parse_mind_behaviors(
                zip_path, prefix, max_interactions=per_split_max[split]
            )
            all_interactions.append(interactions)
            print(f"  {zip_name}: {len(interactions):,} click interactions")

        df = pd.concat(all_interactions, ignore_index=True)
        print(f"  Total raw interactions: {len(df):,}")

        # Remove duplicates (same user clicking same news)
        df = df.drop_duplicates(subset=["user_id", "item_id"])

        # k-core filtering
        df = self._filter_k_core(df, min_user_interactions, min_item_interactions)
        print(f"  After k-core filtering: {len(df):,}")

        return self._process_dataframe(df, implicit=True)

    def _parse_mind_behaviors(
        self, zip_path: Path, prefix: str, max_interactions: Optional[int] = None,
    ) -> pd.DataFrame:
        """Parse MIND behaviors.tsv from zip file into click interactions.

        Stops early once max_interactions rows have been collected to avoid
        loading the full 90M-row file into memory.
        """
        user_ids, item_ids, ratings = [], [], []

        with zipfile.ZipFile(zip_path, "r") as zf:
            behaviors_path = f"{prefix}/behaviors.tsv"
            with zf.open(behaviors_path) as f:
                for line in f:
                    if max_interactions and len(user_ids) >= max_interactions:
                        break
                    parts = line.decode("utf-8").strip().split("\t")
                    if len(parts) < 5:
                        continue

                    # Format: impression_id, user_id, time, history, impressions
                    user_id = parts[1]
                    # history: space-separated news IDs the user previously clicked
                    history = parts[3].strip() if len(parts) > 3 and parts[3].strip() else ""

                    # Add history clicks as positive interactions
                    if history:
                        for news_id in history.split():
                            user_ids.append(user_id)
                            item_ids.append(news_id)
                            ratings.append(1.0)
                            if max_interactions and len(user_ids) >= max_interactions:
                                break

                    if max_interactions and len(user_ids) >= max_interactions:
                        break

                    # impressions: space-separated "newsId-label" pairs
                    impressions = parts[4].strip() if len(parts) > 4 else ""
                    if impressions:
                        for imp in impressions.split():
                            if "-" in imp:
                                news_id, label = imp.rsplit("-", 1)
                                if label == "1":  # clicked
                                    user_ids.append(user_id)
                                    item_ids.append(news_id)
                                    ratings.append(1.0)
                                    if max_interactions and len(user_ids) >= max_interactions:
                                        break

        return pd.DataFrame({"user_id": user_ids, "item_id": item_ids,
                              "rating": ratings, "timestamp": 0})

    # ------------------------------------------------------------------ #
    #  Yelp, Gowalla, Last.fm (require manual download)
    # ------------------------------------------------------------------ #

    def load_yelp(
        self,
        min_user_interactions: int = 10,
        min_item_interactions: int = 10,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """
        Load Yelp dataset.

        Requires: data/yelp/yelp_academic_dataset_review.json
        Download from: https://www.yelp.com/dataset
        """
        yelp_dir = self.data_dir / "yelp"
        file_path = yelp_dir / "yelp_academic_dataset_review.json"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Yelp dataset not found at {file_path}\n"
                "Download from: https://www.yelp.com/dataset\n"
                "Place yelp_academic_dataset_review.json in data/yelp/"
            )

        reviews = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                review = json.loads(line)
                reviews.append(
                    {
                        "user_id": review["user_id"],
                        "item_id": review["business_id"],
                        "rating": float(review["stars"]),
                        "timestamp": 0,
                    }
                )

        df = pd.DataFrame(reviews)
        df = self._filter_k_core(df, min_user_interactions, min_item_interactions)

        return self._process_dataframe(df)

    def load_gowalla(
        self,
        min_user_interactions: int = 10,
        min_item_interactions: int = 10,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """
        Load Gowalla check-in dataset.

        Requires: data/gowalla/Gowalla_totalCheckins.txt
        Download from: https://snap.stanford.edu/data/loc-gowalla.html
        """
        gowalla_dir = self.data_dir / "gowalla"
        file_path = gowalla_dir / "Gowalla_totalCheckins.txt"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Gowalla dataset not found at {file_path}\n"
                "Download from: https://snap.stanford.edu/data/loc-gowalla.html"
            )

        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["user_id", "timestamp", "latitude", "longitude", "item_id"],
        )

        df["rating"] = 1.0
        df = df[["user_id", "item_id", "rating", "timestamp"]]
        df = df.drop_duplicates(subset=["user_id", "item_id"])
        df = self._filter_k_core(df, min_user_interactions, min_item_interactions)

        return self._process_dataframe(df, implicit=True)

    def load_lastfm(
        self,
        min_user_interactions: int = 10,
        min_item_interactions: int = 10,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """
        Load Last.fm dataset.

        Requires: data/lastfm/usersha1-artmbid-artname-plays.tsv
        Download from: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html
        """
        lastfm_dir = self.data_dir / "lastfm"
        file_path = lastfm_dir / "usersha1-artmbid-artname-plays.tsv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"LastFM dataset not found at {file_path}\n"
                "Download from: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html"
            )

        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["user_id", "item_id", "artist_name", "plays"],
            usecols=["user_id", "item_id", "plays"],
        )

        # Log-transform play counts as ratings
        df["rating"] = np.log1p(df["plays"])
        df["rating"] = (
            (df["rating"] - df["rating"].min())
            / (df["rating"].max() - df["rating"].min())
            * 4
            + 1
        )
        df["timestamp"] = 0
        df = df[["user_id", "item_id", "rating", "timestamp"]]
        df = self._filter_k_core(df, min_user_interactions, min_item_interactions)

        return self._process_dataframe(df)

    # ------------------------------------------------------------------ #
    #  Feature loading
    # ------------------------------------------------------------------ #

    def load_user_features(self, dataset: str = "ml-100k") -> Optional[pd.DataFrame]:
        """Load user features if available."""
        if dataset in ("movielens-100k", "ml-100k"):
            user_file = self.data_dir / "ml-100k" / "u.user"
            if user_file.exists():
                return pd.read_csv(
                    user_file,
                    sep="|",
                    names=["user_id", "age", "gender", "occupation", "zip_code"],
                )
        return None

    def load_item_features(self, dataset: str = "ml-100k") -> Optional[pd.DataFrame]:
        """Load item features if available."""
        if dataset in ("movielens-100k", "ml-100k"):
            item_file = self.data_dir / "ml-100k" / "u.item"
            if item_file.exists():
                genre_cols = [
                    "unknown", "Action", "Adventure", "Animation", "Children",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                    "Sci-Fi", "Thriller", "War", "Western",
                ]
                cols = [
                    "item_id", "title", "release_date", "video_date", "url"
                ] + genre_cols
                return pd.read_csv(
                    item_file, sep="|", names=cols, encoding="latin-1"
                )
        return None

    def create_context_features(self, timestamps: np.ndarray) -> np.ndarray:
        """Create context features from timestamps."""
        import datetime

        features = []
        for ts in timestamps:
            dt = datetime.datetime.fromtimestamp(ts)

            hour = dt.hour
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)

            day = dt.weekday()
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)

            weekend = 1.0 if day >= 5 else 0.0

            features.append([hour_sin, hour_cos, day_sin, day_cos, weekend])

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _download_and_extract(self, dataset_name: str):
        """Download and extract dataset."""
        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"No download URL for {dataset_name}")

        url = self.DATASET_URLS[dataset_name]
        zip_path = self.data_dir / f"{dataset_name}.zip"

        print(f"Downloading {dataset_name}...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        os.remove(zip_path)
        print("Done!")

    def _filter_k_core(
        self,
        df: pd.DataFrame,
        min_user: int,
        min_item: int,
        max_iterations: int = 100,
    ) -> pd.DataFrame:
        """Apply k-core filtering to ensure minimum interactions."""
        for _ in range(max_iterations):
            initial_size = len(df)

            user_counts = df["user_id"].value_counts()
            valid_users = user_counts[user_counts >= min_user].index
            df = df[df["user_id"].isin(valid_users)]

            item_counts = df["item_id"].value_counts()
            valid_items = item_counts[item_counts >= min_item].index
            df = df[df["item_id"].isin(valid_items)]

            if len(df) == initial_size:
                break

        return df

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        implicit: bool = False,
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """Process DataFrame into train/val/test splits."""
        # Create ID mappings
        unique_users = df["user_id"].unique()
        unique_items = df["item_id"].unique()

        user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        reverse_user_map = {idx: uid for uid, idx in user_id_map.items()}
        reverse_item_map = {idx: iid for iid, idx in item_id_map.items()}

        # Map IDs
        df["user_idx"] = df["user_id"].map(user_id_map)
        df["item_idx"] = df["item_id"].map(item_id_map)

        # Sort by timestamp for temporal split
        if "timestamp" in df.columns and df["timestamp"].max() > 0:
            df = df.sort_values("timestamp")
        else:
            df = df.sample(frac=1, random_state=42)

        # Split
        n = len(df)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        n_users = len(unique_users)
        n_items = len(unique_items)

        def create_interaction_data(data: pd.DataFrame) -> InteractionData:
            timestamps = (
                data["timestamp"].values if "timestamp" in data.columns else None
            )
            return InteractionData(
                user_ids=data["user_idx"].values.astype(np.int32),
                item_ids=data["item_idx"].values.astype(np.int32),
                ratings=data["rating"].values.astype(np.float32),
                timestamps=timestamps,
                n_users=n_users,
                n_items=n_items,
                user_id_map=user_id_map,
                item_id_map=item_id_map,
                reverse_user_map=reverse_user_map,
                reverse_item_map=reverse_item_map,
            )

        return (
            create_interaction_data(train_df),
            create_interaction_data(val_df),
            create_interaction_data(test_df),
        )


def generate_synthetic_data(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 50000,
    n_user_clusters: int = 10,
    seed: int = 42,
) -> InteractionData:
    """Generate synthetic recommendation data with clustered users.

    This function is intended for unit tests only, not for experiments.
    """
    np.random.seed(seed)

    user_clusters = np.random.randint(0, n_user_clusters, size=n_users)
    cluster_item_prefs = np.random.rand(n_user_clusters, n_items)

    user_ids = []
    item_ids = []
    ratings = []
    timestamps = []

    base_time = 1000000000

    for i in range(n_interactions):
        user = np.random.randint(0, n_users)
        cluster = user_clusters[user]

        prefs = cluster_item_prefs[cluster]
        prefs = prefs / prefs.sum()
        item = np.random.choice(n_items, p=prefs)

        base_rating = 3.0 + 2.0 * cluster_item_prefs[cluster, item]
        rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)

        user_ids.append(user)
        item_ids.append(item)
        ratings.append(rating)
        timestamps.append(base_time + i * 100)

    return InteractionData(
        user_ids=np.array(user_ids),
        item_ids=np.array(item_ids),
        ratings=np.array(ratings),
        timestamps=np.array(timestamps),
        n_users=n_users,
        n_items=n_items,
        user_id_map={i: i for i in range(n_users)},
        item_id_map={i: i for i in range(n_items)},
        reverse_user_map={i: i for i in range(n_users)},
        reverse_item_map={i: i for i in range(n_items)},
    )
