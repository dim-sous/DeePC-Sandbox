"""Online sliding Hankel window for DeePC.

Starts with all offline Hankel columns. Appends new columns from
closed-loop I/O data each step. Once max_cols is reached, drops
the oldest column per step (sliding). The CVXPY problem is built
at max_cols dimension from the start (padded if needed) so g never
changes size — no rebuilds, warm-start preserved.

Reference: Online Reduced-Order DeePC (arXiv:2407.16066).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from control.hankel import build_hankel_matrix


class SlidingHankelWindow:
    """Append-then-slide Hankel window with online updates.

    Phase 1 (growing): online columns appended, replacing padding.
    Phase 2 (sliding): oldest column dropped per new column.
    """

    def __init__(
        self,
        u_data: np.ndarray,
        y_data: np.ndarray,
        Tini: int,
        N: int,
        m: int,
        p: int,
        max_cols: int = 0,
    ) -> None:
        self.Tini = Tini
        self.N = N
        self.L = Tini + N
        self.m = m
        self.p = p
        self._update_count = 0

        u_data = np.atleast_2d(u_data)
        y_data = np.atleast_2d(y_data)

        # Build full offline Hankel
        Hu_full = build_hankel_matrix(u_data, self.L)
        Hy_full = build_hankel_matrix(y_data, self.L)
        n_cols_offline = Hu_full.shape[1]

        # max_cols: 0 means use offline size + sim_steps headroom
        if max_cols <= 0:
            max_cols = n_cols_offline
        self._max_cols = max_cols

        # Initialize: all offline columns, pad to max_cols if needed
        if n_cols_offline >= max_cols:
            self._Hu = Hu_full[:, -max_cols:].copy()
            self._Hy = Hy_full[:, -max_cols:].copy()
            self._n_real = max_cols
        else:
            # Real columns on the left, padding on the right
            pad = max_cols - n_cols_offline
            self._Hu = np.column_stack(
                [Hu_full, np.tile(Hu_full[:, -1:], (1, pad))]
            )
            self._Hy = np.column_stack(
                [Hy_full, np.tile(Hy_full[:, -1:], (1, pad))]
            )
            self._n_real = n_cols_offline

        # Trajectory buffer for forming new columns
        self._u_buf: deque[np.ndarray] = deque(maxlen=max_cols + self.L)
        self._y_buf: deque[np.ndarray] = deque(maxlen=max_cols + self.L)

        # Seed with tail of offline data (L-1 samples needed for first column)
        seed_len = min(len(u_data), self.L - 1)
        for i in range(len(u_data) - seed_len, len(u_data)):
            self._u_buf.append(u_data[i].copy())
            self._y_buf.append(y_data[i].copy())

        self._partition()

    def _partition(self) -> None:
        self._Up = self._Hu[: self.Tini * self.m, :]
        self._Uf = self._Hu[self.Tini * self.m :, :]
        self._Yp = self._Hy[: self.Tini * self.p, :]
        self._Yf = self._Hy[self.Tini * self.p :, :]

    def update(self, u_new: np.ndarray, y_new: np.ndarray) -> dict:
        """Append one I/O sample. Grow or slide the window."""
        self._u_buf.append(np.asarray(u_new, dtype=float).ravel())
        self._y_buf.append(np.asarray(y_new, dtype=float).ravel())

        if len(self._u_buf) < self.L:
            return {"updated": False, "reason": "warmup"}

        # Form new column
        u_col = np.concatenate(list(self._u_buf)[-self.L:])
        y_col = np.concatenate(list(self._y_buf)[-self.L:])

        if self._n_real < self._max_cols:
            # Phase 1: replace next padding column
            self._Hu[:, self._n_real] = u_col
            self._Hy[:, self._n_real] = y_col
            self._n_real += 1
        else:
            # Phase 2: slide — drop oldest, append new
            self._Hu[:, :-1] = self._Hu[:, 1:]
            self._Hu[:, -1] = u_col
            self._Hy[:, :-1] = self._Hy[:, 1:]
            self._Hy[:, -1] = y_col

        self._partition()
        self._update_count += 1

        return {"updated": True, "n_updates": self._update_count}

    @property
    def Up(self) -> np.ndarray:
        return self._Up

    @property
    def Yp(self) -> np.ndarray:
        return self._Yp

    @property
    def Uf(self) -> np.ndarray:
        return self._Uf

    @property
    def Yf(self) -> np.ndarray:
        return self._Yf

    @property
    def n_cols(self) -> int:
        return self._max_cols

    @property
    def n_online_updates(self) -> int:
        return self._update_count

    def get_pe_rank(self) -> int:
        return int(np.linalg.matrix_rank(self._Hu, tol=1e-6))

    def get_sigma_min(self) -> float:
        H = np.vstack([self._Hu, self._Hy])
        S = np.linalg.svd(H, compute_uv=False)
        nonzero = S[S > 1e-10]
        return float(nonzero[-1]) if len(nonzero) > 0 else 0.0
