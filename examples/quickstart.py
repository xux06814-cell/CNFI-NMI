"""Minimal, deterministic CNFI-NMI example using synthetic tabular data."""

import numpy as np
import pandas as pd

from cnfi_nmi import CNFINMISelector

rng = np.random.default_rng(42)
n_samples = 500
rain = rng.integers(0, 2, n_samples)
rush_hour = np.where(rng.random(n_samples) < 0.85, rain, 1 - rain)
target = ((rain == 1) & (rush_hour == 1)).astype(int)
target ^= (rng.random(n_samples) < 0.05).astype(int)

X = pd.DataFrame(
    {
        "rain": rain,
        "rush_hour": rush_hour,
        "temperature": rng.normal(20, 5, n_samples),
        "noise": rng.normal(size=n_samples),
    }
)
y = pd.Series(target, name="major_incident")

selector = CNFINMISelector(
    n_features_to_select=2,
    cnfi_features=3,
    alpha=0.5,
    min_support=0.01,
    min_confidence=0.1,
)
X_selected = selector.fit_transform(X, y)

print("Selected features:", selector.selected_features_)
print("Fused scores:", selector.feature_scores_)
print("Transformed shape:", X_selected.shape)
