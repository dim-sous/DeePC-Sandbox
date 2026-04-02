"""Regularization and data-quality utilities for DeePC."""

import numpy as np

from control.hankel import build_hankel_matrix


def check_persistent_excitation(
    u_data: np.ndarray,
    L: int,
    m: int,
    tol: float = 1e-6,
) -> bool:
    """Check whether input data is persistently exciting of order L.

    The input is persistently exciting of order L when the Hankel matrix
    H_L(u) has full row rank, i.e. rank >= L * m.

    Args:
        u_data: Input trajectory, shape (T, m).
        L: Hankel depth (= Tini + N in DeePC).
        m: Number of input channels.
        tol: Tolerance for the rank computation.

    Returns:
        True if the data satisfies the PE condition.
    """
    if u_data.ndim == 1:
        u_data = u_data.reshape(-1, 1)

    Hu = build_hankel_matrix(u_data, L)
    rank = np.linalg.matrix_rank(Hu, tol=tol)
    required_rank = L * m

    if rank < required_rank:
        print(
            f"WARNING: Hankel rank {rank} < required {required_rank}. "
            f"Data is NOT persistently exciting of order {L}."
        )
    return rank >= required_rank
