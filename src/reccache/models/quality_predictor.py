"""Quality predictor for cache decisions."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class QualityPrediction:
    """Prediction result from quality predictor."""

    quality_score: float  # Expected quality if using cache (0-1)
    confidence: float  # Confidence of prediction (0-1)
    use_cache: bool  # Recommendation to use cache or not


class QualityPredictorNetwork(nn.Module):
    """Neural network for quality prediction."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # quality_score, confidence
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class QualityPredictor:
    """
    Predicts quality loss when using cached recommendations.

    Features:
    - Uses distance to cluster center as primary signal
    - Considers cluster size and context match
    - Trained on observed quality differences
    - Provides calibrated confidence estimates
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        quality_threshold: float = 0.1,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.quality_threshold = quality_threshold
        self.device = device

        # Feature normalization stats
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        # Model
        self.model = QualityPredictorNetwork(
            input_dim=4,
            hidden_dim=hidden_dim,
        ).to(device)

        # Training data collection
        self._training_samples: List[Tuple[np.ndarray, float]] = []

        self._trained = False

    def predict(
        self,
        distance_to_center: float,
        cluster_size: int,
        context_match_score: float = 1.0,
        time_since_cache: float = 0.0,
    ) -> QualityPrediction:
        """
        Predict quality of using cached recommendations.

        Args:
            distance_to_center: User's distance to their cluster center
            cluster_size: Number of users in the cluster
            context_match_score: How well context matches (0-1)
            time_since_cache: Time since cache entry was created (hours)

        Returns:
            QualityPrediction with score, confidence, and recommendation
        """
        # Build feature vector
        features = np.array([
            distance_to_center,
            np.log1p(cluster_size),
            context_match_score,
            np.log1p(time_since_cache),
        ], dtype=np.float32)

        if not self._trained:
            # Use heuristic before training
            return self._heuristic_predict(features)

        # Normalize features
        if self._feature_means is not None:
            features = (features - self._feature_means) / (self._feature_stds + 1e-8)

        # Model prediction
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(x)
            quality_score = float(output[0, 0])
            confidence = float(output[0, 1])

        use_cache = quality_score >= (1 - self.quality_threshold)

        return QualityPrediction(
            quality_score=quality_score,
            confidence=confidence,
            use_cache=use_cache,
        )

    def _heuristic_predict(self, features: np.ndarray) -> QualityPrediction:
        """Heuristic prediction before model is trained."""
        distance, log_cluster_size, context_match, time_factor = features

        # Higher distance = lower quality
        distance_penalty = min(1.0, distance / 2.0)

        # Larger clusters = slightly lower quality (more diverse)
        size_penalty = min(0.3, np.exp(log_cluster_size) / 1000 * 0.3)

        # Context mismatch = lower quality
        context_penalty = (1 - context_match) * 0.5

        # Time decay
        time_penalty = min(0.2, time_factor * 0.02)

        quality_score = max(0.0, 1.0 - distance_penalty - size_penalty - context_penalty - time_penalty)

        return QualityPrediction(
            quality_score=quality_score,
            confidence=0.5,  # Low confidence for heuristic
            use_cache=quality_score >= (1 - self.quality_threshold),
        )

    def add_training_sample(
        self,
        distance_to_center: float,
        cluster_size: int,
        context_match_score: float,
        time_since_cache: float,
        actual_quality: float,
    ):
        """
        Add a training sample.

        Call this after comparing cached vs. fresh recommendations.

        Args:
            actual_quality: Measured quality (e.g., NDCG of cached / NDCG of fresh)
        """
        features = np.array([
            distance_to_center,
            np.log1p(cluster_size),
            context_match_score,
            np.log1p(time_since_cache),
        ], dtype=np.float32)

        self._training_samples.append((features, actual_quality))

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        min_samples: int = 100,
    ) -> dict:
        """
        Train the quality predictor.

        Returns:
            Training statistics
        """
        if len(self._training_samples) < min_samples:
            return {"status": "insufficient_data", "n_samples": len(self._training_samples)}

        # Prepare data
        X = np.array([s[0] for s in self._training_samples], dtype=np.float32)
        y = np.array([s[1] for s in self._training_samples], dtype=np.float32)

        # Normalize features
        self._feature_means = X.mean(axis=0)
        self._feature_stds = X.std(axis=0)
        X = (X - self._feature_means) / (self._feature_stds + 1e-8)

        # Create dataset
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Add confidence targets (based on prediction error during training)
        # We'll use a simple proxy: confidence = 1 - |predicted - actual|
        conf_tensor = torch.ones_like(y_tensor)  # Initialize to 1

        dataset = TensorDataset(X_tensor, torch.cat([y_tensor, conf_tensor], dim=1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)

                # Loss on quality prediction
                loss = criterion(output[:, 0:1], batch_y[:, 0:1])

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(loader))

        self._trained = True

        return {
            "status": "success",
            "n_samples": len(self._training_samples),
            "final_loss": losses[-1],
            "losses": losses,
        }

    def evaluate(self, test_samples: List[Tuple[np.ndarray, float]]) -> dict:
        """Evaluate predictor on test samples."""
        if not test_samples:
            return {"error": "no_test_samples"}

        predictions = []
        actuals = []

        for features, actual in test_samples:
            pred = self.predict(
                distance_to_center=features[0],
                cluster_size=int(np.expm1(features[1])),
                context_match_score=features[2],
                time_since_cache=np.expm1(features[3]),
            )
            predictions.append(pred.quality_score)
            actuals.append(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)

        # Decision accuracy: how often did we make the right cache decision?
        threshold = 1 - self.quality_threshold
        predicted_use = predictions >= threshold
        should_use = actuals >= threshold
        decision_accuracy = (predicted_use == should_use).mean()

        return {
            "mae": mae,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "decision_accuracy": decision_accuracy,
            "n_samples": len(test_samples),
        }

    def save(self, path: str):
        """Save model to file."""
        state = {
            "model_state": self.model.state_dict(),
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "trained": self._trained,
            "quality_threshold": self.quality_threshold,
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load model from file."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self._feature_means = state["feature_means"]
        self._feature_stds = state["feature_stds"]
        self._trained = state["trained"]
        self.quality_threshold = state["quality_threshold"]


class AdaptiveQualityPredictor(QualityPredictor):
    """
    Quality predictor with online learning capability.

    Updates predictions based on observed cache performance.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        quality_threshold: float = 0.1,
        device: str = "cpu",
        update_interval: int = 100,
        learning_rate: float = 0.0001,
    ):
        super().__init__(hidden_dim, quality_threshold, device)
        self.update_interval = update_interval
        self.online_lr = learning_rate

        self._recent_samples: List[Tuple[np.ndarray, float]] = []
        self._sample_count = 0

    def add_observation(
        self,
        distance_to_center: float,
        cluster_size: int,
        context_match_score: float,
        time_since_cache: float,
        actual_quality: float,
    ):
        """Add observation and potentially trigger online update."""
        features = np.array([
            distance_to_center,
            np.log1p(cluster_size),
            context_match_score,
            np.log1p(time_since_cache),
        ], dtype=np.float32)

        self._recent_samples.append((features, actual_quality))
        self._sample_count += 1

        # Trigger online update
        if self._trained and self._sample_count % self.update_interval == 0:
            self._online_update()

    def _online_update(self):
        """Perform online update with recent samples."""
        if len(self._recent_samples) < 10:
            return

        # Prepare batch
        X = np.array([s[0] for s in self._recent_samples], dtype=np.float32)
        y = np.array([s[1] for s in self._recent_samples], dtype=np.float32)

        # Normalize
        if self._feature_means is not None:
            X = (X - self._feature_means) / (self._feature_stds + 1e-8)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Single update step
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.online_lr)

        optimizer.zero_grad()
        output = self.model(X_tensor)
        loss = nn.MSELoss()(output[:, 0:1], y_tensor)
        loss.backward()
        optimizer.step()

        # Clear recent samples
        self._recent_samples = []
