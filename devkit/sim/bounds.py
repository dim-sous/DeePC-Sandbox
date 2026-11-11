"""A plant's input bounds as a record keeps them, read where the controller reads them."""

from __future__ import annotations

import numpy as np

from deepc.controller import constraints_to_arrays
from plants.base import PlantBase


def input_bounds_of(plant: PlantBase) -> np.ndarray:
    """Each input's ``[lower, upper]`` bound, in ``input_names`` order, in the input's unit: the
    plant's ``get_constraints``, as the controller reads them (``constraints_to_arrays``). Both
    plants' ``step`` clips to the same values (``tests/test_records_panels.py`` holds it)."""
    u_lb, u_ub, *_ = constraints_to_arrays(plant.get_constraints(), plant.input_names,
                                           plant.output_names)
    return np.column_stack([u_lb, u_ub]).astype(float)
