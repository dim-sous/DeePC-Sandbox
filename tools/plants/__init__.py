"""Plant models for DeePC experiments."""

from plants.base import Constraints, DataCollectionConfig, PlantBase
from plants.bicycle_model import BicycleModel
from plants.coupled_masses import CoupledMasses

__all__ = [
    "PlantBase", "Constraints", "DataCollectionConfig",
    "BicycleModel", "CoupledMasses", "PLANT_REGISTRY",
]

#: The plants a run can name (``--plant``), by name.
PLANT_REGISTRY: dict[str, type[PlantBase]] = {
    "bicycle": BicycleModel,
    "coupled_masses": CoupledMasses,
}
