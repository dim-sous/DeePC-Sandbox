"""Excitation signals the plants' data-collection recipes draw from: a random-hold binary signal
(``generate_prbs``) and a multisine (``generate_multisine``)."""

from __future__ import annotations

import numpy as np


def generate_prbs(
    length: int,
    amplitude: float,
    min_period: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a pseudo-random binary sequence (PRBS)."""
    signal = np.empty(length)
    idx = 0
    value = amplitude if rng.random() < 0.5 else -amplitude

    while idx < length:
        hold = rng.integers(min_period, 2 * min_period + 1)
        signal[idx : min(idx + hold, length)] = value
        idx += hold
        value = -value

    return signal


def generate_multisine(
    length: int,
    amplitude: float,
    n_freqs: int,
    Ts: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a multi-sine excitation signal."""
    t = np.arange(length) * Ts
    nyquist = 0.5 / Ts
    freqs = rng.uniform(0.01 * nyquist, 0.4 * nyquist, size=n_freqs)
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)

    signal = np.zeros(length)
    for f, phi in zip(freqs, phases):
        signal += np.sin(2 * np.pi * f * t + phi)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= amplitude / peak

    return signal
