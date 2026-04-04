"""Abstract base class for plant models in the DeePC sandbox."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Constraints:
    """Named constraint specification from a plant.

    Keys are input/output channel names (matching input_names / output_names).
    Missing keys default to +/-inf (unconstrained).
    """

    u_lb: dict[str, float] = field(default_factory=dict)
    u_ub: dict[str, float] = field(default_factory=dict)
    du_min: dict[str, float] = field(default_factory=dict)
    du_max: dict[str, float] = field(default_factory=dict)
    y_lb: dict[str, float] = field(default_factory=dict)
    y_ub: dict[str, float] = field(default_factory=dict)


@dataclass
class DataCollectionConfig:
    """Plant-specific configuration for data collection.

    Bundles everything the generic collect_data routine needs from a plant:
    episode initial conditions, excitation signals, stabilizing controller,
    and nominal reference trajectory.
    """

    initial_conditions: list[dict[str, Any]]
    """One dict per episode.  Each dict is passed to the plant's ``reset``."""

    excitation_fn: Callable[[int, np.random.Generator], np.ndarray]
    """(T_episode, rng) -> (T_episode, m) array of excitation inputs."""

    stabilizing_controller: Callable[[np.ndarray], np.ndarray]
    """(errors,) -> (m,) stabilizing baseline input."""

    nominal_reference: Callable[[int, float], np.ndarray]
    """(step_index, Ts) -> reference waypoint for get_output."""


class PlantBase(ABC):
    """Abstract base class every plant model must implement."""

    # ── Required properties ───────────────────────────────────────────

    @property
    @abstractmethod
    def m(self) -> int:
        """Number of input channels."""

    @property
    @abstractmethod
    def p(self) -> int:
        """Number of output channels."""

    @property
    @abstractmethod
    def Ts(self) -> float:
        """Sampling time [s]."""

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Ordered names for input channels (length m)."""

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Ordered names for output channels (length p)."""

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """Current internal state vector."""

    # ── Required methods ──────────────────────────────────────────────

    @abstractmethod
    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance one time step given input *u*.  Return new state."""

    @abstractmethod
    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset to initial or given state."""

    @abstractmethod
    def get_output(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute p-dimensional output given state and a reference waypoint.

        The reference format is plant-defined (e.g. bicycle uses
        ``[x, y, heading, v]``).  Returns p output channels, typically
        tracking errors relative to the reference.
        """

    @abstractmethod
    def get_constraints(self) -> Constraints:
        """Return named constraint specification for this plant."""

    @abstractmethod
    def get_default_config_overrides(self) -> dict[str, Any]:
        """Return DeePCConfig field defaults appropriate for this plant.

        Keys are ``DeePCConfig`` field names, e.g.
        ``{'Q_diag': [10, 10, 1], 'R_diag': [0.1, 0.1], 'T_data': 400}``.
        """

    @abstractmethod
    def get_scenarios(self) -> dict[str, Callable[..., np.ndarray]]:
        """Return ``{name: generator}`` for reference scenarios.

        Each generator signature::

            (Tini: int, N: int, sim_steps: int, Ts: float) -> np.ndarray

        The returned array has shape ``(Tini + sim_steps + N, ref_dim)``
        where *ref_dim* matches what ``get_output`` expects as *reference*.
        """

    @abstractmethod
    def get_data_collection_config(self) -> DataCollectionConfig:
        """Return plant-specific data collection configuration."""

    @abstractmethod
    def make_episode_initial_state(
        self, condition: dict[str, Any], rng: np.random.Generator,
    ) -> np.ndarray:
        """Create an initial state for one data-collection episode.

        *condition* is one element from ``DataCollectionConfig.initial_conditions``.
        """

    # ── Optional hooks (defaults provided) ────────────────────────────

    def get_initial_state_for_scenario(
        self, first_waypoint: np.ndarray,
    ) -> np.ndarray:
        """Derive an initial state from the first reference waypoint.

        Default implementation returns *first_waypoint* as-is.
        """
        return first_waypoint

    def get_position_from_state(self, state: np.ndarray) -> np.ndarray | None:
        """Return ``[x, y]`` position for trajectory plotting.

        Return ``None`` if the plant has no spatial position concept.
        """
        return None

    def plot_training_data(
        self,
        u_data: np.ndarray,
        y_data: np.ndarray,
        Ts: float,
    ) -> str | None:
        """Optional custom Plotly HTML div for training data.

        Return ``None`` to use the generic auto-plot.
        """
        return None

    def plot_simulation_results(
        self,
        results: dict,
        config: Any,
    ) -> str | None:
        """Optional custom Plotly HTML div for simulation results.

        Return ``None`` to use the generic auto-plot.
        """
        return None

    def compute_custom_metrics(self, results: dict) -> dict[str, float]:
        """Optional extra metrics beyond generic per-channel RMSE."""
        return {}

    def get_tuning_objective_key(self) -> str:
        """Metric key to minimize during Bayesian tuning.

        Default: RMSE of the first output channel.
        """
        return f"rmse_{self.output_names[0]}"
