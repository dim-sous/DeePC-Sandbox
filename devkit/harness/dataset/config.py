"""The dataset piece's settings: which plant, how much data, how noisy, which seed.

A setting left at ``None`` takes the plant's own default (its ``get_default_config_overrides``),
resolved when the piece runs and saved in the record's ``constants``. The labels and help lines
are placeholders: the words a Lab user reads are the developer's to choose (RCWS #96), asked in
the Lab branch.
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
        default="bicycle", objects=list(PLANT_REGISTRY), label="plant",
        doc="The simulated plant the data is collected from (open words)",
        metadata={"group": PLANT})
    T_data = param.Integer(
        None, allow_None=True, label="T_data",
        doc="Samples collected, split across the plant's episodes; none: the plant's own "
            "(open words)",
        metadata={"group": DATA, "unit": "samples"})
    noise_std_output = param.Number(
        None, allow_None=True, bounds=(0.0, None), label="noise_std_output",
        doc="Standard deviation of the Gaussian noise added to each recorded output; none: the "
            "plant's own; in each output's own unit (open words)",
        metadata={"group": DATA, "unit": "output unit"})
    seed = param.Integer(
        42, label="seed",
        doc="The seed of the episodes' initial states, the excitation and the noise (open words)",
        metadata={"group": RUN, "unit": "-", "replicate": True})
