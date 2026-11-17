"""The open-loop piece: a saved system run simulated open loop, from x0, with an i.i.d. uniform
excitation clipped to the actuator bounds, the outputs measured with noise; the data the closed
loop is built from. A saved run, checked, is what the closed-loop piece runs on
(:func:`open_loop_of`)."""

from .config import Config
from .record import Record
from .run import open_loop_of, run

__all__ = ["Config", "run", "Record", "open_loop_of"]
