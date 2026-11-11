"""The system piece: a system typed as rows, x' = f(x, u, d, θ) and y = h(x, θ), with θ's values,
read and checked. Nothing is simulated. A saved run, checked, is what the open-loop piece runs on
(:func:`system_of`)."""

from .config import Config
from .record import Record
from .run import run, system_of

__all__ = ["Config", "run", "Record", "system_of"]
