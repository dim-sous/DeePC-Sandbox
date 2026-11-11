"""The system piece: a system typed as rows, x' = f(x, u, θ) and y = h(x, θ), with its θ values,
read and checked. Nothing is simulated."""

from .config import Config
from .record import Record
from .run import run

__all__ = ["Config", "run", "Record"]
