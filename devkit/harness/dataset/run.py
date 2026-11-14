"""``run(config) -> Record``: collect training data from a plant, as ``run.py`` does. No printing,
no saving, no drawing. ``plant_of(config)``: the plant a dataset run collects from, at its step.
And ``dataset_of(path)``: a saved dataset run, checked against the code that reads it, for the
pieces that run on one.
"""

from __future__ import annotations

import os
import time

import numpy as np

from plants import PLANT_REGISTRY
from plants.base import PlantBase
from benchlab.record import stamp
from sim.bounds import input_bounds_of
from sim.config import build_deepc_config
from sim.data_generation import collect_data, episode_lengths

from .config import Config
from .record import Record

#: The saved runs already checked in this process, by (path, modification time, size).
_CHECKED: dict[tuple, Record] = {}


def plant_of(cfg: Config) -> PlantBase:
    """The plant a dataset run collects from, built at the run's step ``cfg.Ts`` in seconds (None:
    the plant's own), so the data and every closed loop on it step at one T_s (D1)."""
    return PLANT_REGISTRY[cfg.plant](**({} if cfg.Ts is None else {"Ts": cfg.Ts}))


def run(cfg: Config) -> Record:
    started = time.perf_counter()
    plant = plant_of(cfg)
    deepc_cfg = build_deepc_config(plant, T_data=cfg.T_data, noise_std_output=cfg.noise_std_output)
    u, y = collect_data(plant, deepc_cfg, seed=cfg.seed)
    lengths = episode_lengths(deepc_cfg.T_data, len(plant.get_data_collection_config().initial_conditions))
    numbers = {f"u_rms_{c}": float(np.sqrt(np.mean(u[:, j] ** 2))) for j, c in enumerate(plant.input_names)}
    numbers.update({f"y_rms_{c}": float(np.sqrt(np.mean(y[:, j] ** 2)))
                    for j, c in enumerate(plant.output_names)})
    return Record(
        name="default", config=cfg, provenance=stamp(time.perf_counter() - started),
        constants={"plant_Ts_s": plant.Ts, "T_data": deepc_cfg.T_data,
                   "noise_std_output": deepc_cfg.noise_std_output, "episode_lengths": lengths},
        inputs=list(plant.input_names), outputs=list(plant.output_names),
        input_units=list(plant.input_units), input_bounds=input_bounds_of(plant),
        output_units=list(plant.output_units), u=u, y=y,
        episode_starts=[int(s) for s in np.cumsum([0] + lengths[:-1])], numbers=numbers)


def dataset_of(path: str) -> Record:
    """The saved dataset run at ``path``, checked against the code that reads it: it is run again
    from the settings it saved and must hold what the file holds, value for value -- a file
    written by code this checkout no longer holds is refused, with the command that writes it
    again (BESS ``harness/battery/run.py``, ``battery_of``). Checked once per state of the file in
    a process, so a sweep pays it once."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no saved dataset run at {path}; the dataset piece writes one: "
                                f"{Record.produced_by} [--name NAME]")
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_mtime_ns, st.st_size)
    if key not in _CHECKED:
        saved = Record.load(path)
        again = run(saved.config)
        differ = [f for f in ("u", "y") if not np.array_equal(getattr(saved, f), getattr(again, f))]
        differ += [f for f in ("episode_starts", "inputs", "outputs")
                   if getattr(saved, f) != getattr(again, f)]
        if differ:
            raise ValueError(f"the dataset run at {path} does not run again to what it holds "
                             f"({', '.join(differ)} differ): code this checkout no longer holds "
                             f"wrote it; run it again ({Record.produced_by})")
        _CHECKED[key] = saved
    return _CHECKED[key]
