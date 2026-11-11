"""Data generation using persistently exciting input signals.

Collects error-based outputs relative to a plant-defined nominal
reference.  Multiple episodes at different initial conditions ensure
output diversity.  The plant provides excitation signals, a stabilizing
baseline controller, and episode initial conditions via its
``get_data_collection_config`` method.
"""

from __future__ import annotations

import numpy as np

from deepc.config import DeePCConfig
from plants.base import PlantBase


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

    lengths = episode_lengths(
        config.T_data, len(dc.initial_conditions),
    )

    u_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for i, condition in enumerate(dc.initial_conditions):
        T_ep = lengths[i]

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


def episode_lengths(T_data: int, n_episodes: int) -> list[int]:
    """The samples each of ``n_episodes`` episodes collects: ``T_data`` split as evenly as it
    goes, the first ``T_data mod n_episodes`` episodes one sample longer."""
    T_per, remainder = divmod(T_data, n_episodes)
    return [T_per + (1 if i < remainder else 0) for i in range(n_episodes)]
