"""Online noise estimator for robust DeePC.

Estimates measurement noise / model mismatch from prediction residuals.
The estimated noise level scales the Wasserstein robustness radius
(lambda_g, lambda_y) per Coulson/Lygeros/Dörfler (arXiv:1903.06804).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class NoiseEstimator:
    """Rolling prediction-residual variance estimator."""

    def __init__(self, p: int, window: int = 10) -> None:
        self.p = p
        self.window = window
        self._residuals: deque[np.ndarray] = deque(maxlen=window)
        self._noise_std = np.zeros(p)

    def update(self, y_predicted: np.ndarray, y_actual: np.ndarray) -> None:
        """Record one-step prediction residual and update noise estimate."""
        residual = np.asarray(y_actual, dtype=float) - np.asarray(y_predicted, dtype=float)
        self._residuals.append(residual)
        if len(self._residuals) >= 3:
            arr = np.array(self._residuals)
            self._noise_std = np.std(arr, axis=0, ddof=1)

    def get_noise_std(self) -> np.ndarray:
        """Per-channel noise standard deviation estimate."""
        return self._noise_std.copy()

    def get_scaling_factor(self, baseline_noise_std: float, max_scaling: float = 10.0) -> float:
        """Ratio of estimated noise to baseline, clamped to [1, max_scaling].

        This scales the Wasserstein ball radius: larger noise → larger
        regularization → more robust (but more conservative) control.
        """
        if baseline_noise_std <= 0.0 or len(self._residuals) < 3:
            return 1.0
        mean_std = float(np.mean(self._noise_std))
        ratio = mean_std / baseline_noise_std
        return float(np.clip(ratio, 1.0, max_scaling))
