"""The constraints a plant hands the controller: named bounds on its inputs, their changes and its
outputs. The controller reads them as values (``controller.constraints_to_arrays``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Constraints:
    """Named constraint specification from a plant.

    Keys are input/output channel names (matching input_names / output_names).
    Missing keys default to +/-inf (unconstrained).
    """

    u_lb: dict[str, float] = field(default_factory=dict)
    u_ub: dict[str, float] = field(default_factory=dict)
    du_min: dict[str, float] = field(default_factory=dict)
    du_max: dict[str, float] = field(default_factory=dict)
    y_lb: dict[str, float] = field(default_factory=dict)
    y_ub: dict[str, float] = field(default_factory=dict)
