"""Regularization and data-quality utilities for DeePC."""

import numpy as np

from deepc.hankel import build_hankel_matrix


def check_persistent_excitation(
    u_data: np.ndarray,
    L: int,
    m: int,
    tol: float = 1e-6,
) -> tuple[int, int]:
    """Rank of the input Hankel matrix H_L(u) and the rank L * m it needs for the input
    to be persistently exciting of order L (full row rank).

    Args:
        u_data: Input trajectory, shape (T, m).
        L: Hankel depth (= Tini + N in DeePC).
        m: Number of input channels.
        tol: Tolerance for the rank computation.

    Returns:
        (rank of H_L(u), L * m). The caller records both; nothing is printed (D3).
    """
    if u_data.ndim == 1:
        u_data = u_data.reshape(-1, 1)

    Hu = build_hankel_matrix(u_data, L)
    return int(np.linalg.matrix_rank(Hu, tol=tol)), L * m
