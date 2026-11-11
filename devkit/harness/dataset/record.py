"""What one run of the dataset piece leaves behind, and the panels and numbers it declares.

The data is the record ``collect_data`` returns: the episodes stacked end to end, so the sample
index runs across episode boundaries (``episode_starts``; D9). Every number is a reduction of an
array on this record. Every axis carries its unit, a channel's the plant's own (``input_units``,
``output_units``). Panel names and labels are placeholders (RCWS #96).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from record import Number, Provenance, Series
from record import Record as _Record

from .config import Config

_INPUTS, _OUTPUTS = "Inputs", "Outputs"


@dataclasses.dataclass(frozen=True)
class Record(_Record):
    name: str
    config: Config
    provenance: Provenance | None
    constants: dict          # what the run used beyond its Config: the resolved settings, the plant's step
    inputs: list             # input channel names, in column order
    outputs: list            # output channel names, in column order
    input_units: list        # the unit of each input, the plant's own (input_units)
    output_units: list       # the unit of each output, the plant's own (output_units)
    u: np.ndarray            # (T_data, m) the inputs as recorded
    y: np.ndarray            # (T_data, p) the outputs as recorded, noise included
    episode_starts: list     # the sample index each episode starts at
    numbers: dict

    component = "dataset"
    label = "dataset"
    produced_by = "uv run deepc run dataset [config.json]"

    def series(self) -> tuple[Series, ...]:
        sample = np.arange(len(self.u))
        out = []
        for group, names, units, data in ((_INPUTS, self.inputs, self.input_units, self.u),
                                          (_OUTPUTS, self.outputs, self.output_units, self.y)):
            for j, (channel, unit) in enumerate(zip(names, units)):
                out.append(Series(
                    name=f"{group[:-1].lower()}-{channel}".replace("_", "-"), x=sample,
                    x_label="sample [-]", y_label=f"{channel} [{unit}]",
                    traces=((channel, data[:, j]),),
                    doc=f"The recorded {channel} at each sample, episodes stacked end to end "
                        f"(open words)", group=group))
        return tuple(out)

    def number_specs(self) -> tuple[Number, ...]:
        return tuple(Number(f"{kind}_rms_{channel}", f"{channel} RMS", unit, group=group,
                            doc=f"Root mean square of the recorded {channel} over all samples "
                                f"(open words)")
                     for kind, group, names, units in (("u", _INPUTS, self.inputs, self.input_units),
                                                       ("y", _OUTPUTS, self.outputs, self.output_units))
                     for channel, unit in zip(names, units))
