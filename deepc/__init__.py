"""DeePC controller package."""

from deepc.deepc_controller import DeePCController
from deepc.hankel import build_data_matrices, build_hankel_matrix
from deepc.regularization import check_persistent_excitation

__all__ = [
    "DeePCController",
    "build_data_matrices",
    "build_hankel_matrix",
    "check_persistent_excitation",
]
