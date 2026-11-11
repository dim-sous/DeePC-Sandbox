"""``run(config) -> Record``: read the typed rows with their θ values and check them
(``system.parse``). Nothing is simulated. No printing, no saving, no drawing. And
``system_of(path)``: a saved system run, read again and checked, for the pieces that run on one.
"""

from __future__ import annotations

import os
import time

from record import stamp
from system import Parsed, parse

from .config import Config
from .record import Record


def _read(cfg: Config) -> tuple[Record, Parsed]:
    """The run's record and the system it read, from one parse."""
    started = time.perf_counter()
    parsed = parse(cfg.f, cfg.h, cfg.theta)
    return Record(
        name="default", config=cfg, provenance=stamp(time.perf_counter() - started),
        constants={"f_read": list(parsed.f_read), "h_read": list(parsed.h_read)},
        numbers={"n": parsed.n, "m": parsed.m, "p": parsed.p, "q": parsed.q,
                 "r": parsed.r}), parsed


def run(cfg: Config) -> Record:
    return _read(cfg)[0]


def system_of(path: str) -> Parsed:
    """The system the saved run at ``path`` defines, checked: its settings are read again and
    must give the rows as read and the sizes the file holds -- a file written by code this
    checkout no longer holds is refused, with the command that writes it again (as
    ``harness.dataset.dataset_of``: the run made again from the saved settings, every constant and
    number compared). Reading the rows again costs milliseconds, so it is not cached."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no saved system run at {path}; the system piece writes one: "
                                f"{Record.produced_by} [--name NAME]")
    saved = Record.load(path)
    again, parsed = _read(saved.config)
    differ = [k for k in set(saved.constants) | set(again.constants)
              if saved.constants.get(k) != again.constants.get(k)]
    differ += [k for k in set(saved.numbers) | set(again.numbers)
               if saved.numbers.get(k) != again.numbers.get(k)]
    if differ:
        raise ValueError(f"the system run at {path} does not read again to what it holds "
                         f"({', '.join(differ)} differ): code this checkout no longer holds wrote "
                         f"it; run it again ({Record.produced_by})")
    return parsed
