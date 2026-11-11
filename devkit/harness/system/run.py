"""``run(config) -> Record``: read the typed rows with their θ values and check them
(``system.parse``). Nothing is simulated. No printing, no saving, no drawing."""

from __future__ import annotations

import time

from record import stamp
from system import parse

from .config import Config
from .record import Record


def run(cfg: Config) -> Record:
    started = time.perf_counter()
    parsed = parse(cfg.f, cfg.h, cfg.theta)
    return Record(
        name="default", config=cfg, provenance=stamp(time.perf_counter() - started),
        constants={"f_read": list(parsed.f_read), "h_read": list(parsed.h_read)},
        numbers={"n": parsed.n, "m": parsed.m, "p": parsed.p, "q": parsed.q})
