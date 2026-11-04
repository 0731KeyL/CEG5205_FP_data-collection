"""Eight-class gesture classifier for the dataset's 84 statistical features."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GestureMLP(nn.Module):
    """Accept unstandardized features and return logits in dataset label order."""

    def __init__(self, normalization_path: str | Path):
        super().__init__()
        with np.load(normalization_path, allow_pickle=False) as stats:
            mean = torch.tensor(stats["feature_mean"], dtype=torch.float32)
            std = torch.tensor(stats["feature_std"], dtype=torch.float32)
        if mean.shape != (84,) or std.shape != (84,):
            raise ValueError("Expected 84 feature means and standard deviations.")
        if not (mean.isfinite().all() and std.isfinite().all() and (std > 0).all()):
            raise ValueError("Expected finite statistics and positive standard deviations.")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.network = nn.Sequential(nn.Linear(84, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map (batch_size, 84) features to (batch_size, 8) logits."""
        standardized = (features - self.feature_mean) / self.feature_std
        return self.network(standardized)
