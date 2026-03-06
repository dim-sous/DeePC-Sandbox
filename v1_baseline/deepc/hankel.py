"""Hankel matrix construction for the DeePC framework."""

import numpy as np


def build_hankel_matrix(data: np.ndarray, L: int) -> np.ndarray:
    """Build a block-Hankel matrix from a data sequence.

    For a data matrix of shape (T, n) and depth L, the Hankel matrix has
    L*n rows and T-L+1 columns.  Column j contains the flattened block
    [data[j], data[j+1], ..., data[j+L-1]].

    Args:
        data: Data array of shape (T, n_channels).  A 1-D array is
              treated as a single channel.
        L: Number of block rows (Hankel depth).

    Returns:
        Hankel matrix of shape (L * n_channels, T - L + 1).

    Raises:
        ValueError: If the data length is shorter than L.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T, n = data.shape
    if T < L:
        raise ValueError(
            f"Data length {T} is shorter than Hankel depth {L}."
        )

    num_cols = T - L + 1
    H = np.empty((L * n, num_cols))

    for j in range(num_cols):
        H[:, j] = data[j : j + L].ravel()

    return H


def build_data_matrices(
    u_data: np.ndarray,
    y_data: np.ndarray,
    Tini: int,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the four partitioned Hankel blocks used by DeePC.

    Constructs H_L(u) and H_L(y) with L = Tini + N, then splits each
    into a *past* block (first Tini block-rows) and a *future* block
    (last N block-rows).

    Args:
        u_data: Input data, shape (T_data, m).
        y_data: Output data, shape (T_data, p).
        Tini: Length of the initialization (past) window.
        N: Prediction (future) horizon.

    Returns:
        Up: Past input block,   shape (Tini * m, cols).
        Yp: Past output block,  shape (Tini * p, cols).
        Uf: Future input block,  shape (N * m, cols).
        Yf: Future output block, shape (N * p, cols).
    """
    if u_data.ndim == 1:
        u_data = u_data.reshape(-1, 1)
    if y_data.ndim == 1:
        y_data = y_data.reshape(-1, 1)

    m = u_data.shape[1]
    p = y_data.shape[1]
    L = Tini + N

    Hu = build_hankel_matrix(u_data, L)
    Hy = build_hankel_matrix(y_data, L)

    Up = Hu[: Tini * m, :]
    Uf = Hu[Tini * m :, :]
    Yp = Hy[: Tini * p, :]
    Yf = Hy[Tini * p :, :]

    return Up, Yp, Uf, Yf
