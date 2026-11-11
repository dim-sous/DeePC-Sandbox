# Methods

What the code computes: the two plants, how the training data is collected, the Hankel matrices, the DeePC
quadratic program, the loop that runs it, its two optional variants, the metrics and the tuner.

**Read this first.** This document was written from the code, on 2026-11-10, with no access to the papers
it cites. It states what the code does, not what the literature says. **No citation below has been checked
against its source**; each section ends with what to check once the papers are read. Line pointers
(`file.py:line`) are at commit `3067d03`. Where the code departs from what it claims, or from itself, the
incident in [`docs/project/failures.md`](docs/project/failures.md) is named (D1, D2, …).

## 1. The plants

Both plants implement `PlantBase` (`plants/base.py`) and are listed in `PLANT_REGISTRY` (`run.py:40`). Their
outputs are **tracking errors** against a reference waypoint $r$, not measurements of the state: the
controller is asked to drive the errors to zero.

### 1.1 Kinematic bicycle

`plants/bicycle_model.py`. State $x = [p_x,\ p_y,\ \psi,\ v]$ (position in m, heading in rad, speed in m/s;
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

`plants/coupled_masses.py`. State $x = [x_1,\ v_1,\ x_2,\ v_2]$, input $u = [F_1,\ F_2]$. The code states no
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

`sim/data_generation.py:62-114`. The plant's `get_data_collection_config()` names $E$ episodes (5 for both
plants), an excitation signal, a baseline controller $\kappa$ and a nominal reference $\bar r_k$. `T_data`
samples are split into $E$ episodes of $\lfloor T_{data}/E \rfloor$ samples, the first
$T_{data} \bmod E$ episodes one longer (`:77-84`). Each episode resets the plant to its own initial state
(`:88`) and runs, for $k = 0, \dots, T_e - 1$ (`:96-109`):

```math
u_k = \kappa\big(e(x_k, \bar r_k)\big) + w_k, \qquad x_{k+1} = f(x_k, u_k), \qquad
y_k = e(x_{k+1}, \bar r_k) + n_k, \qquad n_k \sim \mathcal N(0, \sigma_n^2 I),
```

with $w_k$ the excitation and $\sigma_n$ = `noise_std_output` (bicycle 0.01, masses 0.005). The recorded
output $y_k$ is the one after the step. The recorded input $u_k$ is the one before the plant's own clipping
(§1; `sim/data_generation.py:101-108`): where the plant clips, the data holds an input it did not apply
(D12). The baseline controller reads the error before the noise is added
(D4). The episodes are then stacked end to end into one record of `T_data` samples (`:114`; D9).

**Bicycle** (`plants/bicycle_model.py:243-382`): each episode starts at $p = 0$, $\psi \sim U(-0.3, 0.3)$ rad,
$v = \max(v_{ref}(1 + \Delta), 0.5)$ for $\Delta \in \{0, -0.4, 0.4, -0.7, 0.7\}$. Nominal reference: the
straight line $\bar r_k = [(k+1)\, v_{ref} T_s,\ 0,\ 0,\ v_{ref}]$. Baseline:
$\kappa(e) = [-0.3\, e_{lat} - 1.0\, e_\psi,\ -0.5\, e_v]$. Excitation: steering
$\tfrac12$ PRBS(0.5 rad, minimum hold 3) $+ \tfrac12\, U(-0.5, 0.5)$; acceleration
$\tfrac12$ multisine(2.5 m/s², 10 frequencies) $+ \tfrac12\, U(-2.5, 2.5)$.

**Coupled masses** (`plants/coupled_masses.py:162-282`): each episode starts at position offsets
$(x_1, x_2) \in \{(0,0), (0.5,-0.5), (-0.5,0.5), (0.3,0.3), (-0.3,-0.3)\}$ with $v_i \sim U(-0.2, 0.2)$. Nominal
reference: zero. Baseline: $F_i = -1.5\, e_{x_i} - 0.8\, e_{v_i}$. Excitation per channel:
$\tfrac12$ PRBS (amplitude $0.6 F_{max}$, minimum hold 3) $+\ \tfrac12$ multisine (amplitude $0.6 F_{max}$,
8 frequencies).

**The two signals.** What the code calls a PRBS (`sim/data_generation.py:18-35`) is a binary signal
$\pm A$ that starts at a random sign and flips after each hold, the hold drawn uniformly from
$\{m, \dots, 2m\}$ samples. It is not a maximum-length shift-register sequence. The multisine (`:38-59`) is
$s(t) = \sum_i \sin(2\pi f_i t + \phi_i)$ with $f_i \sim U(0.01, 0.4) \cdot \frac{1}{2T_s}$ and
$\phi_i \sim U(0, 2\pi)$, scaled so that $\max |s| = A$. Frequencies and phases are drawn again for every
episode and channel.

## 3. Hankel matrices and persistency of excitation

**Block-Hankel matrix** (`control/hankel.py:6-37`). For a signal $w_0, \dots, w_{T-1} \in \mathbb R^{n_w}$
and a depth $L$, column $j$ of $H_L(w)$ stacks $L$ consecutive samples:

```math
H_L(w) = \begin{bmatrix} w_0 & w_1 & \cdots & w_{T-L} \\ w_1 & w_2 & \cdots & w_{T-L+1} \\
\vdots & \vdots & & \vdots \\ w_{L-1} & w_L & \cdots & w_{T-1} \end{bmatrix} \in \mathbb R^{L n_w \times (T-L+1)} .
```

DeePC uses $L = T_{ini} + N$ (`control/config.py:71`): the first $T_{ini}$ block rows are the *past* blocks
$U_p, Y_p$, the last $N$ the *future* blocks $U_f, Y_f$ (`control/online_hankel.py:86-90`, which the
controller uses; `control/hankel.py:40-81` does the same and nothing calls it, D8). At the plants' defaults
($T_{ini}, N$ = 3, 15 for the bicycle and 4, 20 for the masses), $H_L(u)$ is 36 × 383 and 48 × 577.

**Persistency of excitation** (`control/regularization.py:8-40`). The check is

```math
\mathrm{rank}\, H_L(u) \ge L\, m \qquad (\text{numerical rank, tolerance } 10^{-6}),
```

on the inputs only, of order $L$. When it fails, a warning is printed and the controller is built anyway
(D3). The old README names Willems et al. [4] as the foundation.

To check against the sources: the order of persistency of excitation the fundamental lemma asks for [4],
and how several datasets may be combined into one matrix (D9).

## 4. The DeePC quadratic program

`control/controller.py:90-202`, solved once per step. Decision variables: $g \in \mathbb R^{n_c}$ (one weight
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
  (`sim/simulation.py:123-128`); $\tau$ is the noise-adaptive tightening (§6), zero otherwise.
- The Hankel blocks, $u_{ini}$, $y_{ini}$, $y_{ref}$, $u_{prev}$, $\lambda_g$, $\lambda_y$ and $\tau$ are
  CVXPY parameters, so the problem is built once and re-solved with new values.
- Solver (`:245-266`): OSQP with `eps_abs` = `eps_rel` = $10^{-5}$, `max_iter` = 10000, polishing, warm start.
  Any exception, of any kind, re-solves with SCS, unrecorded (D3). With no solution, the controller returns
  $u = 0$ and the loop applies it (D3). Otherwise the first predicted input $u_0$ is applied (`:293`).

This is the formulation the old README states. The old README names Coulson et al. [1] as its source; the
code names none for the QP.
The old README also lists Berberich et al. [5]; no code here implements it.

To check against [1]: whether its regulariser on $g$ is a norm or a squared norm; the norm on $\sigma_y$;
whether the cost weighs $u$ or its changes; and its convention for when an output is sampled relative to its
input (here $y_k$ is the output after $u_k$ is applied, §2 and §5).

## 5. The closed loop, tracking on errors

`sim/simulation.py:25-200`. The plant starts at the first waypoint, $x_0 = r_0$ (`:61`;
`get_initial_state_for_scenario` returns the waypoint itself). The loop then runs:

1. **Warm-up**, $T_{ini}$ steps with $u = 0$ (`:78-85`), each recording $y_i = e(x_{i+1}, r_i)$.
2. **Then, for** $k = 0, \dots, K - 1$, with $K$ = `int(sim_duration / Ts)` (`control/config.py:67-68`, D1):
   - solve §4 with $y_{ref} = 0$ over the whole horizon (`:126`);
   - apply $u_0$;
   - record $y = e(x_{i+1}, r_i)$ with $i = k + T_{ini}$ (`:139-141`).

The loop adds no measurement noise (D4). The state after step $i$, at time $(i+1)T_s$, is compared with
waypoint $r_i$, at time $iT_s$. In the bicycle's data collection, the state after step $k$ is compared with
$\bar r_k$ at $(k+1) v_{ref} T_s$, the same instant (D10). The positions behind the bicycle's position RMSE are
compared at equal times (§8).

## 6. Noise-adaptive regularisation (`--noise-adaptive`)

`control/noise_estimator.py`, `sim/simulation.py:107-121`. From the second solve on, the residual between the
output recorded after a step and the output the previous solve predicted for it,
$\varrho_k = y_k - \hat y_{k|k-1}$, enters a window of the last $W$ = 10 residuals. Once it holds 3 or
more, the per-channel sample standard deviation $\hat\sigma$ (`ddof` = 1) gives

```math
s = \mathrm{clip}\!\left(\frac{\operatorname{mean}(\hat\sigma)}{\sigma_0},\ 1,\ k_{max}\right), \qquad
\lambda_g \leftarrow s\, \lambda_g^{(0)}, \qquad \lambda_y \leftarrow s\, \lambda_y^{(0)}, \qquad
\tau = c\, \hat\sigma \ \text{(repeated over the horizon)},
```

with $\sigma_0$ = `baseline_noise_std` = 0.01, $k_{max}$ = 10 and $c$ = 2 (`control/config.py:53-56`).
$\tau$ enters only the soft output bounds, which exist only when $\lambda_{out} > 0$ and only on the outputs
with a finite bound (`control/controller.py:182-199`). At every plant's default it enters no constraint, and
for the coupled masses, which have no output bounds, at no $\lambda_{out}$ (D11). The code cites
arXiv:1903.06804 [2] for scaling the regularisation with the estimated noise.

To check against [2]: whether it scales $\lambda_g, \lambda_y$ with an estimated noise level, and by what
rule.

## 7. Online Hankel window (`--online-hankel`)

`control/online_hankel.py`, `sim/simulation.py:144-148`.

- **Size.** The window holds `max_cols` columns: `hankel_window_size`, or the number of offline columns when
  that is 0, the default (`:54-55`). A larger window is padded with copies of the last offline column
  (`:59-70`).
- **First columns.** A buffer is seeded with the last $L - 1$ training samples (`:79-82`).
- **Each step from `hankel_warmup_steps` on** (default 100), the step's input and output are appended. A
  column is formed from the last $L$ buffered samples (`:92-119`). It replaces the next padding column while
  any is left; otherwise every column shifts left by one and the new one takes the last place.
- **Into the QP.** The four blocks are re-split and passed to the QP's parameters (`control/controller.py:204-212`).
- **What the first columns hold.** The first $L - 1$ online columns join the last training samples to the
  loop's first ones, which do not follow each other in time (D9).
- **The gate.** An update is made only while the mean noise estimate of §6 is below 1.0 (D5).

The code cites arXiv:2407.16066 [3].

To check against [3]: how it forms and drops columns, and what it requires of them.

## 8. Metrics

`run.py:65-106` and each plant's `compute_custom_metrics`:

- **RMSE per output channel**, over all $T_{ini} + K$ samples, warm-up included.
- **Bicycle** (`plants/bicycle_model.py:269-281`): the RMSE of $p_x - r_x$, of $p_y - r_y$ and of their
  Euclidean norm (`rmse_position`). Each position is compared with the waypoint of the same time.
- **Coupled masses** (`plants/coupled_masses.py:198-205`): the RMSE of $(e_{x_1}, e_{x_2})$ jointly.
- **Solve times**: mean, maximum and total wall-clock time of the `solve` call.
- **The "optimal" share**: the share of solves whose status contains the word "optimal", so OSQP's
  `optimal_inaccurate` counts (D2).
- **The slack**: the mean and maximum of $\lVert \sigma_y \rVert_2$, whichever norm its penalty uses.

## 9. Tuning (`tune.py`)

Optuna's TPE sampler, seed 42 (`tune.py:343-346`), minimises one metric per trial: `rmse_position` for the
bicycle, `rmse_e_x1` for the masses. If that metric is missing, the mean of the RMSEs is used. Each trial
collects its data at seed 42 and runs the closed loop for `--sim-duration` (default 10 s) on the `--scenario`
(default `default`). Search space (`tune.py:61-71`):

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

None of these has been checked against its source in this repository. Each entry is given as the
repository's own text gives it.

1. J. Coulson, J. Lygeros, F. Dörfler, *Data-Enabled Predictive Control: In the Shallows of the DeePC*.
   Given by the old README as *European Journal of Control*, 2019; venue, volume and pages not checked.
2. arXiv:1903.06804, cited in `control/noise_estimator.py` as "Coulson/Lygeros/Dörfler" for the
   "Wasserstein robustness radius"; title not given in the code, not checked.
3. arXiv:2407.16066, cited in `control/online_hankel.py` as "Online Reduced-Order DeePC"; authors not given
   in the code, not checked.
4. J. C. Willems, P. Rapisarda, I. Markovsky, B. L. M. De Moor, *A note on persistency of excitation*,
   *Systems & Control Letters*, 2005 (as the old README gives it; volume and pages not checked).
5. J. Berberich, A. Koch, C. W. Scherer, F. Allgöwer, *Robust data-driven state-feedback design*, ACC 2020
   (as the old README gives it). No code in this repository implements it.

Software, at the versions `uv.lock` pins: CVXPY 1.8.1, OSQP 1.1.1, SCS 3.2.11, Optuna 4.8.0, NumPy 2.4.2,
SciPy 1.17.1, Plotly 6.6.0.
