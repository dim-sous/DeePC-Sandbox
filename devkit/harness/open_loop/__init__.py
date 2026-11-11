"""The open-loop piece: a saved system run simulated open loop, from x0, with an i.i.d. uniform
excitation clipped to the actuator bounds, the outputs measured with noise; the data the closed
loop is built from."""

from .config import Config
from .record import Record
from .run import run

__all__ = ["Config", "run", "Record"]
