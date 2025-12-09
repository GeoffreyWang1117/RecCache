"""Data loading utilities for recommendation datasets."""

import os
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import urllib.request
import zipfile

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


class DataLoader:
    """Load and preprocess recommendation datasets."""

    MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_movielens_100k(
        self, min_rating: float = 0.0
    ) -> Tuple[InteractionData, InteractionData, InteractionData]:
        """
        Load MovieLens 100K dataset.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        ml_dir = self.data_dir / "ml-100k"

        if not ml_dir.exists():
            self._download_movielens_100k()

        # Load ratings
        ratings_file = ml_dir / "u.data"
        df = pd.read_csv(
            ratings_file,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        if min_rating > 0:
            df = df[df["rating"] >= min_rating]

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
        df = df.sort_values("timestamp")

        # Split: 80% train, 10% val, 10% test
        n = len(df)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        n_users = len(unique_users)
        n_items = len(unique_items)

        def create_interaction_data(data: pd.DataFrame) -> InteractionData:
            return InteractionData(
                user_ids=data["user_idx"].values,
                item_ids=data["item_idx"].values,
                ratings=data["rating"].values,
                timestamps=data["timestamp"].values,
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

    def _download_movielens_100k(self):
        """Download MovieLens 100K dataset."""
        zip_path = self.data_dir / "ml-100k.zip"

        print("Downloading MovieLens 100K dataset...")
        urllib.request.urlretrieve(self.MOVIELENS_URL, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        os.remove(zip_path)
        print("Done!")

    def load_user_features(self, dataset: str = "movielens-100k") -> Optional[pd.DataFrame]:
        """Load user features if available."""
        if dataset == "movielens-100k":
            ml_dir = self.data_dir / "ml-100k"
            user_file = ml_dir / "u.user"

            if user_file.exists():
                df = pd.read_csv(
                    user_file,
                    sep="|",
                    names=["user_id", "age", "gender", "occupation", "zip_code"],
                )
                return df
        return None

    def load_item_features(self, dataset: str = "movielens-100k") -> Optional[pd.DataFrame]:
        """Load item features if available."""
        if dataset == "movielens-100k":
            ml_dir = self.data_dir / "ml-100k"
            item_file = ml_dir / "u.item"

            if item_file.exists():
                genre_cols = [
                    "unknown", "Action", "Adventure", "Animation", "Children",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                    "Sci-Fi", "Thriller", "War", "Western"
                ]
                cols = ["item_id", "title", "release_date", "video_date", "url"] + genre_cols

                df = pd.read_csv(
                    item_file,
                    sep="|",
                    names=cols,
                    encoding="latin-1",
                )
                return df
        return None

    def create_context_features(
        self, timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Create context features from timestamps.

        Features: [hour_sin, hour_cos, day_sin, day_cos, weekend]
        """
        import datetime

        features = []
        for ts in timestamps:
            dt = datetime.datetime.fromtimestamp(ts)

            # Hour of day (0-23) -> sin/cos encoding
            hour = dt.hour
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)

            # Day of week (0-6) -> sin/cos encoding
            day = dt.weekday()
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)

            # Weekend flag
            weekend = 1.0 if day >= 5 else 0.0

            features.append([hour_sin, hour_cos, day_sin, day_cos, weekend])

        return np.array(features, dtype=np.float32)


def generate_synthetic_data(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 50000,
    n_user_clusters: int = 10,
    seed: int = 42,
) -> InteractionData:
    """
    Generate synthetic recommendation data with clustered users.

    This is useful for testing clustering and cache behavior.
    """
    np.random.seed(seed)

    # Generate user cluster assignments
    user_clusters = np.random.randint(0, n_user_clusters, size=n_users)

    # Generate cluster preferences (which items each cluster likes)
    cluster_item_prefs = np.random.rand(n_user_clusters, n_items)

    # Generate interactions
    user_ids = []
    item_ids = []
    ratings = []
    timestamps = []

    base_time = 1000000000

    for i in range(n_interactions):
        user = np.random.randint(0, n_users)
        cluster = user_clusters[user]

        # Sample item with preference bias
        prefs = cluster_item_prefs[cluster]
        prefs = prefs / prefs.sum()
        item = np.random.choice(n_items, p=prefs)

        # Generate rating based on preference
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
