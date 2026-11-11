"""What one run of the system piece leaves behind: the system as SymPy read its rows (θ kept as
symbols) and its sizes. A system run is a definition: nothing is simulated, so it declares no
chart; the open-loop piece simulates it (docs/project/design.md).
"""

from __future__ import annotations

import dataclasses

from record import Number, Provenance, Series
from record import Record as _Record

from .config import Config

_SIZES = "Sizes"


@dataclasses.dataclass(frozen=True)
class Record(_Record):
    name: str
    config: Config
    provenance: Provenance | None
    constants: dict          # the rows as SymPy read them: f_read (x_i'), h_read (y_j)
    numbers: dict            # the sizes n, m, p, q

    component = "system"
    label = "System"
    produced_by = "uv run deepc run system [config.json]"

    def series(self) -> tuple[Series, ...]:
        return ()

    def number_specs(self) -> tuple[Number, ...]:
        return tuple(Number(name, name, headline=(name == "n"), group=_SIZES, doc=doc)
                     for name, doc in (("n", "The number of states, x1…xn: one per state row"),
                                       ("m", "The number of inputs, u1…um"),
                                       ("p", "The number of outputs, y1…yp: one per output row"),
                                       ("q", "The number of parameters, theta1…thetaq")))
