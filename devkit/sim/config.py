"""The DeePCConfig a run uses, built from a plant's defaults and the run's overrides. On the bench,
because it reads a plant: the controller's config itself (``deepc.config``) names none.
"""

from __future__ import annotations

from typing import Any

from deepc.config import DeePCConfig
from plants.base import PlantBase


def build_deepc_config(plant: PlantBase, **overrides: Any) -> DeePCConfig:
    """Build a DeePCConfig from a plant's defaults + user overrides.

    Plant defaults are applied first, then ``m``, ``p``, ``Ts`` from the
    plant object, then any explicit *overrides* (e.g. from CLI args). ``Ts``
    is refused as an override: it is the plant's step, set where the plant is
    built, and an override reached only the config (D1).
    """
    if overrides.get("Ts") is not None:
        raise ValueError(f"Ts={overrides['Ts']!r} is given to the config, but T_s is the plant's "
                         f"step (plant.Ts = {plant.Ts}): build the plant with it (D1)")
    defaults = plant.get_default_config_overrides()
    defaults["m"] = plant.m
    defaults["p"] = plant.p
    defaults["Ts"] = plant.Ts
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    return DeePCConfig(**defaults)
