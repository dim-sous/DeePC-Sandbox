"""Data generation using persistently exciting input signals.

Collects error-based outputs relative to a plant-defined nominal
reference.  Multiple episodes at different initial conditions ensure
output diversity.  The plant provides excitation signals, a stabilizing
baseline controller, and episode initial conditions via its
``get_data_collection_config`` method.
"""

from __future__ import annotations

import numpy as np

from control.config import DeePCConfig
from plants.base import PlantBase


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


def collect_data(
    plant: PlantBase,
    config: DeePCConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect persistently exciting input-output data.

    Uses the plant's data-collection config for episode initial
    conditions, excitation signals, stabilizing controller, and
    nominal reference trajectory.
    """
    rng = np.random.default_rng(seed)
    dc = plant.get_data_collection_config()

    n_episodes = len(dc.initial_conditions)
    T_per = config.T_data // n_episodes
    remainder = config.T_data - T_per * n_episodes

    u_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for i, condition in enumerate(dc.initial_conditions):
        T_ep = T_per + (1 if i < remainder else 0)

        # Plant-specific initial state for this episode
        initial_state = plant.make_episode_initial_state(condition, rng)
        plant.reset(initial_state)

        # Plant-specific excitation
        excitation = dc.excitation_fn(T_ep, rng)

        u_ep = np.zeros((T_ep, plant.m))
        y_ep = np.zeros((T_ep, plant.p))

        for k in range(T_ep):
            ref_k = dc.nominal_reference(k, plant.Ts)
            errors = plant.get_output(plant.state, ref_k)

            u_stab = dc.stabilizing_controller(errors)
            u_k = u_stab + excitation[k]

            plant.step(u_k)

            errors = plant.get_output(plant.state, ref_k)
            errors = errors + rng.normal(0, config.noise_std_output, size=plant.p)

            u_ep[k] = u_k
            y_ep[k] = errors

        u_parts.append(u_ep)
        y_parts.append(y_ep)

    return np.vstack(u_parts), np.vstack(y_parts)
