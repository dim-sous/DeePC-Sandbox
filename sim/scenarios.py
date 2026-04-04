"""Reference scenario dispatcher.

Each plant defines its own scenarios via ``get_scenarios()``.
This module provides a thin convenience function.
"""

from __future__ import annotations

import numpy as np

from control.config import DeePCConfig
from plants.base import PlantBase


def get_reference(
    plant: PlantBase,
    scenario: str,
    config: DeePCConfig,
) -> np.ndarray:
    """Generate a reference trajectory for the given scenario.

    Delegates to the plant's scenario generator.
    """
    scenarios = plant.get_scenarios()
    if scenario not in scenarios:
        available = ", ".join(sorted(scenarios))
        raise ValueError(
            f"Unknown scenario '{scenario}' for this plant. "
            f"Available: {available}"
        )
    return scenarios[scenario](config.Tini, config.N, config.sim_steps, config.Ts)
