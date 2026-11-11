"""The dataset piece's settings: which plant at which step, how much data, how noisy, which seed.

A setting left at ``None`` takes the plant's own default (its ``get_default_config_overrides``),
resolved when the piece runs and saved in the record's ``constants``. The labels and help lines
follow the developer's pattern for the words a Lab user reads (CLAUDE.md;
``docs/project/lab-words.md``).
"""

from __future__ import annotations

import param

from plants import PLANT_REGISTRY
from record.config import Settings

PLANT, DATA, RUN = "Plant", "Data", "Run"
GROUPS: tuple[str, ...] = (PLANT, DATA, RUN)


class Config(Settings):
    """The dataset piece's settings; see the module."""

    plant = param.Selector(
        default="coupled_masses", objects=list(PLANT_REGISTRY), label="Plant",
        doc="The simulated plant the data is collected from",
        metadata={"group": PLANT})
    Ts = param.Number(
        None, allow_None=True, bounds=(0.0, None), inclusive_bounds=(False, True), label="T_s",
        doc="The plant's step and the data's sampling time, in seconds; the closed loop on this "
            "run steps at it too; none: the plant's own",
        metadata={"group": PLANT, "unit": "s"})
    T_data = param.Integer(
        None, allow_None=True, label="T",
        doc="Samples collected, split across the plant's episodes; none: the plant's own",
        metadata={"group": DATA, "unit": "samples"})
    noise_std_output = param.Number(
        None, allow_None=True, bounds=(0.0, None), label="σ_n",
        doc="Standard deviation of the Gaussian noise added to each recorded output; none: the "
            "plant's own; in each output's own unit",
        metadata={"group": DATA, "unit": "output unit"})
    seed = param.Integer(
        42, label="Seed",
        doc="The seed of the episodes' initial states, the excitation and the noise",
        metadata={"group": RUN, "unit": "-", "replicate": True})
