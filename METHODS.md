# Methods

What the code computes: the two plants, how the training data is collected, the Hankel matrices, the DeePC
quadratic program, the loop that runs it, its two optional variants, the metrics and the tuner.

**Read this first.** This document was written from the code on 2026-11-10 and checked against the papers
on 2026-11-11: the arXiv versions of [1]–[3], [5] and [6], whose section, equation and theorem numbers are
the ones used here. [4] is not on arXiv and has not been read. Each section states what the code does.
§3, §4, §6 and §7, the sections that cite a method, end with an **Against the sources** paragraph that quotes
what the papers state and names where the code differs; the other sections cite no paper. Line
pointers (`file.py:line`) were read at commit `3067d03` and re-pointed to the three-root layout by
matching each line's text. Where the code departs from a paper, from what it claims,
or from itself, the incident in [`docs/project/failures.md`](docs/project/failures.md) is named (D1, D2, …).

## 1. The plants

Both plants implement `PlantBase` (`tools/plants/base.py`) and are listed in `PLANT_REGISTRY` (`tools/plants/__init__.py:13`). Their
outputs are **tracking errors** against a reference waypoint $r$, not measurements of the state: the
controller is asked to drive the errors to zero.

### 1.1 Kinematic bicycle

`tools/plants/bicycle_model.py`. State $x = [p_x,\ p_y,\ \psi,\ v]$ (position in m, heading in rad, speed in m/s;
the code's `x, y, theta, v`), input $u = [\delta,\ a]$ (steering angle in rad, acceleration in m/s²). One
step of length $T_s$ (`:178-188`):

```math
\begin{aligned}
\bar\delta &= \mathrm{clip}(\delta, -\delta_{max}, \delta_{max}), \qquad \bar a = \mathrm{clip}(a, a_{min}, a_{max}), \\
p_x^+ &= p_x + T_s\, v \cos\psi, \qquad p_y^+ = p_y + T_s\, v \sin\psi, \\
\psi^+ &= \psi + T_s\, \frac{v}{L_{wb}} \tan\bar\delta, \qquad v^+ = \max(v + T_s\, \bar a,\ 0).
\end{aligned}
```

This is forward Euler; the reference point moves along the heading (no slip-angle term). No source for the
model is cited anywhere in the repository. Defaults (`:95-107`): $T_s$ = 0.1 s, $L_{wb}$ = 2.5 m,
$\delta_{max}$ = 0.5 rad, $a \in [-1, 1]$ m/s², reference speed $v_{ref}$ = 5 m/s, path amplitude
$A$ = 5 m, path frequency $f$ = 0.05 Hz.

**Outputs** (`:46-51`), against a waypoint $r = [r_x,\ r_y,\ r_\psi,\ r_v]$:

```math
e_{lat} = -(p_x - r_x)\sin r_\psi + (p_y - r_y)\cos r_\psi, \qquad
e_\psi = \mathrm{wrap}_{[-\pi,\pi)}(\psi - r_\psi), \qquad
e_v = v - r_v .
```

There is no along-track error among the outputs: an offset along the path's tangent changes none of the
three. The bicycle's position RMSE (§8), which is also the tuner's objective (§9), does include it.

**Scenarios** (`:285-327`), at $t_k = k T_s$:

- `default`: $r_x = v_{ref}\, t + A \sin(2\pi f t)$, $r_y = A \sin(4\pi f t)$;
- `lissajous`: $r_x = A \sin(2\pi f t)$, $r_y = A \sin(4\pi f t)$.

The heading and speed of each waypoint are forward differences of the positions,
$r_{\psi,k} = \mathrm{atan2}(\Delta r_y, \Delta r_x)$ and $r_{v,k} = \lVert \Delta r \rVert / T_s$, the last
waypoint repeating the one before. In both scenarios the reference speed $r_v$ therefore varies along the
path; it is not $v_{ref}$.

**Constraints** (`:208-218`): the input bounds equal the plant's own clipping bounds; the rate bounds are
per sample, $|\Delta\delta| \le 0.3$ rad and $|\Delta a| \le 0.5$ m/s² between consecutive samples, so their
value per second depends on $T_s$; output bounds sit on the speed error, $e_v \in [0, 10]$ (D7), and act only
when $\lambda_{out} > 0$ (§4).

### 1.2 Coupled two-mass spring-damper

`tools/plants/coupled_masses.py`. State $x = [x_1,\ v_1,\ x_2,\ v_2]$, input $u = [F_1,\ F_2]$. The code states no
units; SI is assumed here. The continuous dynamics (docstring and `:59-70`):

```math
\begin{aligned}
m_1 \ddot x_1 &= -k_1 x_1 - c_1 \dot x_1 + k_{12}(x_2 - x_1) + c_{12}(\dot x_2 - \dot x_1) + F_1, \\
m_2 \ddot x_2 &= -k_2 x_2 - c_2 \dot x_2 + k_{12}(x_1 - x_2) + c_{12}(\dot x_1 - \dot x_2) + F_2,
\end{aligned}
```

written as $\dot x = A_c x + B_c u$ and discretised by forward Euler, $A = I + T_s A_c$, $B = T_s B_c$
(`:73-74`). One step (`:111-112`): $x^+ = A x + B\, \mathrm{clip}(u, -F_{max}, F_{max})$. Defaults (`:38-48`):
$T_s$ = 0.05 s, $m_1 = m_2$ = 1, $k_1 = k_2$ = 2, $k_{12}$ = 1, $c_1 = c_2$ = 0.3, $c_{12}$ = 0.1,
$F_{max}$ = 5, and a rate bound of 2 per sample.

**Outputs** (`:127`): $e = x - r$, with $r = [r_{x_1},\ r_{v_1},\ r_{x_2},\ r_{v_2}]$.

**Scenarios**: `default` (`:218-242`), position steps held 5 s each, mass 1 through
$0, 1, -0.5, 1.5, -1, 0.5, -1.5, 1, 0$ and mass 2 through $0, -0.5, 1, -1, 0.5, 1.5, -0.5, 0$ starting
2.5 s later; `sinusoidal` (`:254-255`), $r_{x_1} = \sin(0.2\pi t)$, $r_{x_2} = 0.8\sin(0.3\pi t + \pi/3)$. In
both, the velocity references stay zero, so $e_{v_i} = v_i$ even while the position reference moves. The
`default` references are zero for the first 5 s.

**Constraints** (`:129-137`): $|F_i| \le F_{max}$ and $|\Delta F_i| \le 2$ per sample; no output bounds.

## 2. Collecting the training data

`devkit/sim/data_generation.py:18-70`. The plant's `get_data_collection_config()` names $E$ episodes (5 for both
plants), an excitation signal, a baseline controller $\kappa$ and a nominal reference $\bar r_k$. `T_data`
samples are split into $E$ episodes of $\lfloor T_{data}/E \rfloor$ samples, the first
$T_{data} \bmod E$ episodes one longer (`:33-40`). Each episode resets the plant to its own initial state
(`:44`) and runs, for $k = 0, \dots, T_e - 1$ (`:52-65`):

```math
u_k = \kappa\big(e(x_k, \bar r_k)\big) + w_k, \qquad x_{k+1} = f(x_k, u_k), \qquad
y_k = e(x_{k+1}, \bar r_k) + n_k, \qquad n_k \sim \mathcal N(0, \sigma_n^2 I),
```

with $w_k$ the excitation and $\sigma_n$ = `noise_std_output` (bicycle 0.01, masses 0.005). The recorded
output $y_k$ is the one after the step. The recorded input $u_k$ is the one before the plant's own clipping
(§1; `devkit/sim/data_generation.py:57-64`): where the plant clips, the data holds an input it did not apply
(D12). The baseline controller reads the error before the noise is added
(D4). The episodes are then stacked end to end into one record of `T_data` samples (`:70`; D9).

**Bicycle** (`tools/plants/bicycle_model.py:243-382`): each episode starts at $p = 0$, $\psi \sim U(-0.3, 0.3)$ rad,
$v = \max(v_{ref}(1 + \Delta), 0.5)$ for $\Delta \in \{0, -0.4, 0.4, -0.7, 0.7\}$. Nominal reference: the
straight line $\bar r_k = [(k+1)\, v_{ref} T_s,\ 0,\ 0,\ v_{ref}]$. Baseline:
$\kappa(e) = [-0.3\, e_{lat} - 1.0\, e_\psi,\ -0.5\, e_v]$. Excitation: steering
$\tfrac12$ PRBS(0.5 rad, minimum hold 3) $+ \tfrac12\, U(-0.5, 0.5)$; acceleration
$\tfrac12$ multisine(2.5 m/s², 10 frequencies) $+ \tfrac12\, U(-2.5, 2.5)$.

**Coupled masses** (`tools/plants/coupled_masses.py:162-282`): each episode starts at position offsets
$(x_1, x_2) \in \{(0,0), (0.5,-0.5), (-0.5,0.5), (0.3,0.3), (-0.3,-0.3)\}$ with $v_i \sim U(-0.2, 0.2)$. Nominal
reference: zero. Baseline: $F_i = -1.5\, e_{x_i} - 0.8\, e_{v_i}$. Excitation per channel:
$\tfrac12$ PRBS (amplitude $0.6 F_{max}$, minimum hold 3) $+\ \tfrac12$ multisine (amplitude $0.6 F_{max}$,
8 frequencies).

**The two signals.** What the code calls a PRBS (`tools/plants/excitation.py:9-26`) is a binary signal
$\pm A$ that starts at a random sign and flips after each hold, the hold drawn uniformly from
$\{m, \dots, 2m\}$ samples. It is not a maximum-length shift-register sequence. The multisine (`tools/plants/excitation.py:29-50`) is
$s(t) = \sum_i \sin(2\pi f_i t + \phi_i)$ with $f_i \sim U(0.01, 0.4) \cdot \frac{1}{2T_s}$ and
$\phi_i \sim U(0, 2\pi)$, scaled so that $\max |s| = A$. Frequencies and phases are drawn again for every
episode and channel.

## 3. Hankel matrices and persistency of excitation

**Block-Hankel matrix** (`src/deepc/hankel.py:6-37`). For a signal $w_0, \dots, w_{T-1} \in \mathbb R^{n_w}$
and a depth $L$, column $j$ of $H_L(w)$ stacks $L$ consecutive samples:

```math
H_L(w) = \begin{bmatrix} w_0 & w_1 & \cdots & w_{T-L} \\ w_1 & w_2 & \cdots & w_{T-L+1} \\
\vdots & \vdots & & \vdots \\ w_{L-1} & w_L & \cdots & w_{T-1} \end{bmatrix} \in \mathbb R^{L n_w \times (T-L+1)} .
```

DeePC uses $L = T_{ini} + N$ (`src/deepc/config.py:66`): the first $T_{ini}$ block rows are the *past* blocks
$U_p, Y_p$, the last $N$ the *future* blocks $U_f, Y_f$ (`src/deepc/online_hankel.py:86-90`, which the
controller uses; `src/deepc/hankel.py:40-81` does the same and nothing calls it, D8). At the plants' defaults
($T_{ini}, N$ = 3, 15 for the bicycle and 4, 20 for the masses), $H_L(u)$ is 36 × 383 and 48 × 577.

**Persistency of excitation** (`src/deepc/regularization.py:8-40`). The check is

```math
\mathrm{rank}\, H_L(u) \ge L\, m \qquad (\text{numerical rank, tolerance } 10^{-6}),
```

on the inputs only, of order $L$. When it fails, a warning is printed and the controller is built anyway
(D3).

**Against the sources.** [1, Definition 4.4] defines persistency of excitation of order $L$ as full row rank
of $H_L(u)$: the code's check. The fundamental lemma, as [1, Lemma 4.2] states it after Willems et al. [4]
(reference 37 of [1]), needs order $t + n(\mathcal B)$ for the columns of $H_t$ to span every length-$t$
trajectory, with $n(\mathcal B)$ the order of a minimal state-space representation. [1, §V-A] assumes order
$T_{ini} + N + n(\mathcal B)$ and $T \ge (m+1)(T_{ini} + N + n(\mathcal B)) - 1$; [6, Theorem 1] asks for
order $n + L$.

- **Differs:** the code checks order $L$, not $L + n$ (D13).
- **Several datasets:** [6, Definition 2, eq. (8)] combines them in a mosaic-Hankel matrix
  $[H_k(u^1)\ \cdots\ H_k(u^q)]$, one Hankel block per dataset, and [6, Theorem 2] extends the lemma to it
  under "collectively persistently exciting" inputs of order $n + L$. **Differs:** the code stacks the
  episodes into one signal before building one Hankel matrix (D9).
- Both statements are for linear time-invariant systems; the bicycle is not one.

## 4. The DeePC quadratic program

`src/deepc/controller.py:90-202`, solved once per step. Decision variables: $g \in \mathbb R^{n_c}$ (one weight
per Hankel column), the predicted inputs $u \in \mathbb R^{Nm}$ and outputs $y \in \mathbb R^{Np}$, the slack
$\sigma_y \in \mathbb R^{T_{ini} p}$ on the past outputs, and, only when $\lambda_{out} > 0$, a slack
$\sigma_{out} \in \mathbb R^{Np}_{\ge 0}$ on the output bounds.

```math
\begin{aligned}
\min_{g,\,u,\,y,\,\sigma_y,\,\sigma_{out}}\quad & \lVert y - y_{ref} \rVert_Q^2 + \lVert u \rVert_R^2
  + \lambda_g\, \rho_g(g) + \lambda_y\, \rho_y(\sigma_y) + \lambda_{out} \lVert \sigma_{out} \rVert_2^2 \\
\text{s.t.}\quad & U_p\, g = u_{ini}, \qquad Y_p\, g - y_{ini} = \sigma_y, \qquad U_f\, g = u, \qquad Y_f\, g = y, \\
& u_{min} \le u_k \le u_{max}, \qquad k = 0, \dots, N-1, \\
& \Delta u_{min} \le u_{k+1} - u_k \le \Delta u_{max}, \qquad k = 0, \dots, N-2, \\
& \Delta u_{min} \le u_0 - u_{prev} \le \Delta u_{max}, \\
& y_{min} + \tau - \sigma_{out} \le y_k \le y_{max} - \tau + \sigma_{out} \quad \text{(finite bounds only, when } \lambda_{out} > 0).
\end{aligned}
```

- $Q$ and $R$ are diagonal, `Q_diag` and `R_diag` repeated over the horizon (`:133-137`): the bicycle
  weights $[10, 10, 0]$ (no weight on $e_v$) and $[0.1, 0.1]$; the masses $[20, 1, 20, 1]$ and $[0.1, 0.1]$.
  $R$ weighs the inputs themselves, not their changes.
- $\rho$ is the 1-norm (`L1`) or the **squared** 2-norm (`L2`), chosen by `reg_norm_g` and `reg_norm_sigma_y`
  (`:139-147`). Both plants default to squared 2-norm on $g$ and 1-norm on $\sigma_y$, with $\lambda_g$ = 5
  (bicycle) or 10 (masses) and $\lambda_y = 10^4$; $\lambda_{out}$ = 0, so no output bound acts by default.
- $u_{ini}, y_{ini}$ are the last $T_{ini}$ applied inputs and outputs; $u_{prev}$ is the last applied input
  (`devkit/sim/simulation.py:123-128`); $\tau$ is the noise-adaptive tightening (§6), zero otherwise.
- The Hankel blocks, $u_{ini}$, $y_{ini}$, $y_{ref}$, $u_{prev}$, $\lambda_g$, $\lambda_y$ and $\tau$ are
  CVXPY parameters, so the problem is built once and re-solved with new values.
- Solver (`src/deepc/controller.py:245-266`): OSQP with `eps_abs` = `eps_rel` = $10^{-5}$, `max_iter` = 10000, polishing, warm start.
  Any exception, of any kind, re-solves with SCS, unrecorded (D3). With no solution, the controller returns
  $u = 0$ and the loop applies it (D3). Otherwise the first predicted input $u_0$ is applied (`src/deepc/controller.py:293`).

This is the formulation the old README states. The old README names Coulson et al. [1] as its source; the
code names none for the QP. The old README also lists Berberich et al. [5]; no code here implements it.

**Against the sources.** [1, eq. (6)] is the DeePC problem: the cost
$\sum_k \lVert y_k - r_{t+k} \rVert_Q^2 + \lVert u_k \rVert_R^2$, the equality
$\mathrm{col}(U_p, Y_p, U_f, Y_f)\, g = \mathrm{col}(u_{ini}, y_{ini}, u, y)$ and $u_k \in \mathcal U$,
$y_k \in \mathcal Y$. Its regularised form, [1, eq. (8)], adds $\lambda_g \lVert g \rVert_1 + \lambda_y
\lVert \sigma_y \rVert_1$, with $\sigma_y$ on the $Y_p$ row, and replaces the data matrix by "a low-rank matrix
approximation" of it.

- **As in [1]:** the cost weighs the inputs, not their changes; the slack $\sigma_y$ sits on the past
  outputs, with the 1-norm by default.
- **Differs:** the code's default regulariser on $g$ is the squared 2-norm, not [1]'s 1-norm, and the code
  makes no low-rank approximation (D14). [2, Remark 4.3]: "being robust in the trajectory space in the
  $\lVert\cdot\rVert_2$ sense requires regularizing with a $\lVert\cdot\rVert_2$" (the norm, not its square).
- **Added by the code**, in none of [1, eq. (6)–(8)]: the input-rate bounds and the soft output bounds.
- **Output timing:** [1, eq. (1)] has $y(t) = Cx(t) + Du(t)$. The output the code records after input $u_k$
  is, for the coupled masses, $y_k = A x_k + B\,\mathrm{clip}(u_k) - r$: while the clip does not act (it
  never does in the frozen training data, D12), the form of [1, eq. (1)] with $C = A$, $D = B$ and the
  reference subtracted. For the bicycle it has the form $y = h(x, u)$ of [1, eq. (7)].
- **Differs, the reference:** [1, eq. (6)] tracks a reference $r$ with the system's outputs. The code's
  outputs are errors against the scenario's reference, its cost reference is zero, and its training errors
  are taken against the nominal reference (§2, §5) (D15).

## 5. The closed loop, tracking on errors

`devkit/sim/simulation.py:25-200`. The plant starts at the first waypoint, $x_0 = r_0$ (`:61`;
`get_initial_state_for_scenario` returns the waypoint itself). The loop then runs:

1. **Warm-up**, $T_{ini}$ steps with $u = 0$ (`:78-85`), each recording $y_i = e(x_{i+1}, r_i)$.
2. **Then, for** $k = 0, \dots, K - 1$, with $K$ = `int(sim_duration / Ts)` (`src/deepc/config.py:62-63`, D1):
   - solve §4 with $y_{ref} = 0$ over the whole horizon (`devkit/sim/simulation.py:126`);
   - apply $u_0$;
   - record $y = e(x_{i+1}, r_i)$ with $i = k + T_{ini}$ (`devkit/sim/simulation.py:139-141`).

The loop adds no measurement noise (D4). The state after step $i$, at time $(i+1)T_s$, is compared with
waypoint $r_i$, at time $iT_s$. In the bicycle's data collection, the state after step $k$ is compared with
$\bar r_k$ at $(k+1) v_{ref} T_s$, the same instant (D10). The positions behind the bicycle's position RMSE are
compared at equal times (§8).

## 6. Noise-adaptive regularisation (`--noise-adaptive`)

`src/deepc/noise_estimator.py`, `devkit/sim/simulation.py:107-121`. From the second solve on, the residual between the
output recorded after a step and the output the previous solve predicted for it,
$\varrho_k = y_k - \hat y_{k|k-1}$, enters a window of the last $W$ = 10 residuals. Once it holds 3 or
more, the per-channel sample standard deviation $\hat\sigma$ (`ddof` = 1) gives

```math
s = \mathrm{clip}\!\left(\frac{\operatorname{mean}(\hat\sigma)}{\sigma_0},\ 1,\ k_{max}\right), \qquad
\lambda_g \leftarrow s\, \lambda_g^{(0)}, \qquad \lambda_y \leftarrow s\, \lambda_y^{(0)}, \qquad
\tau = c\, \hat\sigma \ \text{(repeated over the horizon)},
```

with $\sigma_0$ = `baseline_noise_std` = 0.01, $k_{max}$ = 10 and $c$ = 2 (`src/deepc/config.py:48-51`).
$\tau$ enters only the soft output bounds, which exist only when $\lambda_{out} > 0$ and only on the outputs
with a finite bound (`src/deepc/controller.py:182-199`). At every plant's default it enters no constraint, and
for the coupled masses, which have no output bounds, at no $\lambda_{out}$ (D11). The code cites
arXiv:1903.06804 [2] for scaling the regularisation with the estimated noise.

**Against the sources.** [2, eq. (8)] regularises with
$\lambda_{ini} \lVert \hat Y_p g - \hat y_{ini} \rVert_1 + \lambda_g \lVert g \rVert_1$. After
[2, Theorem 4.2]: "$\lambda_g$ in (8) plays the role of either $\epsilon \cdot \sup_{\theta \in \Theta_2}
\lVert\theta\rVert_\infty$ or $\epsilon\lambda_{ini}$", with $\epsilon$ the radius of a Wasserstein ball around
the data's empirical distribution. [2, Remark 4.1] says to "choose the radius $\epsilon$ of the Wasserstein
ball in a data-driven fashion (see Section V)", and in [2, §V] "A Wasserstein radius $\epsilon$ was fixed and
the cost $f(u, y)$ accumulated over the 20 second horizon of the simulation was computed. The cost was
averaged over 15 random simulations for each $\epsilon$."

- **Differs:** [2] fixes $\epsilon$ for a run and picks it by comparing runs. The code rescales $\lambda_g$
  and $\lambda_y$ at every step from an online estimate of its prediction residuals, and tightens the output
  bounds. The text of [2] mentions neither an online noise estimate nor an output tightening (D16).

## 7. Online Hankel window (`--online-hankel`)

`src/deepc/online_hankel.py`, `devkit/sim/simulation.py:144-148`.

- **Size.** The window holds `max_cols` columns: `hankel_window_size`, or the number of offline columns when
  that is 0, the default (`:54-55`). A larger window is padded with copies of the last offline column
  (`:59-70`).
- **First columns.** A buffer is seeded with the last $L - 1$ training samples (`:79-82`).
- **Each step from `hankel_warmup_steps` on** (default 100), the step's input and output are appended. A
  column is formed from the last $L$ buffered samples (`:92-119`). It replaces the next padding column while
  any is left; otherwise every column shifts left by one and the new one takes the last place.
- **Into the QP.** The four blocks are re-split and passed to the QP's parameters (`src/deepc/controller.py:204-212`).
- **What the first columns hold.** The first $L - 1$ online columns join the last training samples to the
  loop's first ones, which do not follow each other in time (D9).
- **The gate.** An update is made only while the mean noise estimate of §6 is below 1.0 (D5).

The code cites arXiv:2407.16066 [3].

**Against the sources.** [3, Algorithm 1] keeps every column it accepts. From step $K$ on, it appends the
latest length-$K$ trajectory to a mosaic-Hankel matrix [3, eq. (11)] and keeps it only if the matrix's
minimum non-zero singular value stays at or above a threshold; otherwise it removes it. [3, §III-B] then
reduces the growing matrix's order with a numerical SVD.

The sliding window the code implements is the one [3, §I] attributes to Baros et al. (*Automatica* 138,
109926, 2022, as [3] cites it, its reference 18; not read here). In it, "the first column (the oldest data
point) is discarded from the Hankel matrix, all columns are shifted back by one step, and the most recent
real-time data is added as the last column", and [3] says this method "requires a PE control input in
real-time".

- **Differs:** the code cites [3] for Baros et al.'s window, gates its updates on the noise estimate rather
  than on a singular value, and makes no order reduction (D17).

## 8. Metrics

`run.py:58-99` and each plant's `compute_custom_metrics`:

- **RMSE per output channel**, over all $T_{ini} + K$ samples, warm-up included.
- **Bicycle** (`tools/plants/bicycle_model.py:269-281`): the RMSE of $p_x - r_x$, of $p_y - r_y$ and of their
  Euclidean norm (`rmse_position`). Each position is compared with the waypoint of the same time.
- **Coupled masses** (`tools/plants/coupled_masses.py:198-205`): the RMSE of $(e_{x_1}, e_{x_2})$ jointly.
- **Solve times**: mean, maximum and total wall-clock time of the `solve` call.
- **The "optimal" share**: the share of solves whose status contains the word "optimal", so OSQP's
  `optimal_inaccurate` counts (D2).
- **The slack**: the mean and maximum of $\lVert \sigma_y \rVert_2$, whichever norm its penalty uses.

## 9. Tuning (`tune.py`)

Optuna's TPE sampler, seed 42 (`tune.py:342-345`), minimises one metric per trial: `rmse_position` for the
bicycle, `rmse_e_x1` for the masses. If that metric is missing, the mean of the RMSEs is used. Each trial
collects its data at seed 42 and runs the closed loop for `--sim-duration` (default 10 s) on the `--scenario`
(default `default`). Search space (`tune.py:60-70`):

| setting | range | scale |
|---|---|---|
| `Ts` | 0.02 – 0.2 s | log (never reaches the plant, D1) |
| `Tini` | 1 – 10 | integer |
| `N` | 3 – 30 | integer |
| `T_data` | 100 – 1000, step 50 | integer |
| `lambda_g` | 0.01 – 200 | log |
| `lambda_y` | 1 – 10⁶ | log |
| `reg_norm_g`, `reg_norm_sigma_y` | L1, L2 | choice |

A trial that raises, or returns a non-finite metric, scores $10^6$ (D3).

## References

Checked on 2026-11-11 against the arXiv version named: title and authors from the arXiv record and the
PDF's title page, section and equation numbers from that version. The PDFs are kept in `references/`, which
is git-ignored. [4] is not on arXiv and has not been read.

1. J. Coulson, J. Lygeros, F. Dörfler, "Data-Enabled Predictive Control: In the Shallows of the DeePC",
   arXiv:1811.05890v2, 15 Mar 2019. Neither the arXiv record nor the PDF names a venue. The old README gave
   *European Journal of Control*, 2019, which is not confirmed.
2. J. Coulson, J. Lygeros, F. Dörfler, "Regularized and Distributionally Robust Data-Enabled Predictive
   Control", arXiv:1903.06804v2, 1 Nov 2019. No venue on the arXiv record.
3. A. Vahidi-Moghaddam, K. Zhang, X. Yin, V. Srivastava, Z. Li, "Online Reduced-Order Data-Enabled
   Predictive Control", arXiv:2407.16066v1, 22 Jul 2024.
4. J. C. Willems, P. Rapisarda, I. Markovsky, B. L. De Moor, "A note on persistency of excitation",
   *Systems & Control Letters* 54(4), 325–329, 2005: the whole entry as [1] cites it (its reference 37).
   Not on arXiv; not read.
5. J. Berberich, A. Koch, C. W. Scherer, F. Allgöwer, "Robust data-driven state-feedback design", in *Proc.
   American Control Conference*, 2020, pp. 1532–1538 (the arXiv record's journal reference);
   arXiv:1909.04314v3, 26 Oct 2020. The PDF's title page names A. Koch; the arXiv record lists that author
   as Anne Romer. No code here implements it.
6. H. J. van Waarde, C. De Persis, M. K. Camlibel, P. Tesi, "Willems' Fundamental Lemma for State-space
   Systems and its Extension to Multiple Datasets", arXiv:2002.01023v2, 7 May 2020. No venue on the arXiv
   record.

Software, at the versions `uv.lock` pins: CVXPY 1.8.1, OSQP 1.1.1, SCS 3.2.11, Optuna 4.8.0, NumPy 2.4.2,
SciPy 1.17.1, Plotly 6.6.0.
