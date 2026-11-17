# Methods

What the code computes: the Hankel matrices, the DeePC quadratic program, the closed loop that runs it and
its numbers (§3–§5, §8), the typed system and the open-loop simulation that makes the controller's data
(§10, §11), and the two variants whose code no piece reaches any more (§6, §7). §1, §2 and §9 described the
plant class, its data collection and the tuner, deleted on 2026-11-17; each keeps a note of what went.

**Read this first.** This document was written from the code on 2026-11-10 and checked against the papers
on 2026-11-11: the arXiv versions of [1]–[3], [5] and [6], whose section, equation and theorem numbers are
the ones used here. [4] is not on arXiv and has not been read. Each section states what the code does.
§3, §4, §6 and §7, the sections that compare the code with a method, end with an **Against the sources** paragraph that quotes
what the papers state and names where the code differs; the other sections cite no paper. Line
pointers (`file.py:line`) were read at commit `3067d03` and re-pointed to the three-root layout by
matching each line's text. Where the code departs from a paper, from what it claims,
or from itself, the incident in [`docs/project/failures.md`](docs/project/failures.md) is named (D1, D2, …).

## 1. The plant (deleted 2026-11-17)

The coupled masses were a hand-written plant class (`c7f4569:tools/plants/coupled_masses.py`), stepped by
forward Euler at $T_s$, its outputs the **tracking errors** $e = x - r$ against a scenario's waypoints
(D15). The same system is now typed as rows, the `system` piece's default, $	heta = [k, k_{12}, c,
c_{12}] = [2, 1, 0.3, 0.1]$ with unit masses, integrated by RK45 (§10); its outputs are the states
themselves, $y = x$. The old class's actuator bound $|F_i| \le 5$ is the open-loop simulation's default
(§11); its rate bound of 2 per sample, its `default` steps and its SI units have no counterpart. Its
`sinusoidal` scenario, $r_{x_1} = \sin(0.2\pi t)$, $r_{x_2} = 0.8\sin(0.3\pi t + \pi/3)$, the velocity
references zero, is the closed loop's default reference, typed (§5).

## 2. Collecting the training data (deleted 2026-11-17)

The old dataset piece (`c7f4569:devkit/sim/data_generation.py`) ran several episodes, each from its own
reset state, under a proportional baseline controller plus excitation, and stacked them end to end; it
recorded the output after each step with noise and the input before the plant's clip (D4, D9, D12). The
controller's data is now one open-loop run (§11): one trajectory, no baseline controller, the applied
input recorded, $y_k = h(x_k)$ plus noise.

## 3. Hankel matrices and persistency of excitation

**Block-Hankel matrix** (`src/deepc/hankel.py:6-37`). For a signal $w_0, \dots, w_{T-1} \in \mathbb R^{n_w}$
and a depth $L$, column $j$ of $H_L(w)$ stacks $L$ consecutive samples:

```math
H_L(w) = \begin{bmatrix} w_0 & w_1 & \cdots & w_{T-L} \\ w_1 & w_2 & \cdots & w_{T-L+1} \\
\vdots & \vdots & & \vdots \\ w_{L-1} & w_L & \cdots & w_{T-1} \end{bmatrix} \in \mathbb R^{L n_w \times (T-L+1)} .
```

DeePC uses $L = T_{ini} + N$ (`src/deepc/config.py:66`): the first $T_{ini}$ block rows are the *past* blocks
$U_p, Y_p$, the last $N$ the *future* blocks $U_f, Y_f$ (`src/deepc/online_hankel.py:86-90`, which the
controller uses; `src/deepc/hankel.py:40-81` does the same, called only by a test of the fundamental lemma,
`tests/test_closed_loop.py`). At the closed loop's defaults ($T_{ini}, N$ = 4, 20) on the default open-loop
run ($T$ = 600, $m$ = 2), $H_L(u)$ is 48 × 577.

**Persistency of excitation** (`src/deepc/regularization.py:8-30`). The check is

```math
\mathrm{rank}\, H_L(u) \ge L\, m \qquad (\text{numerical rank, tolerance } 10^{-6}),
```

on the inputs only, of order $L$. The rank and $L\,m$ are kept on the controller and recorded by the
closed-loop piece (`constants`, `excitation_rank`); the controller is built either way (D3).

**Against the sources.** [1, Definition 4.4] defines persistency of excitation of order $L$ as full row rank
of $H_L(u)$: the code's check. The fundamental lemma, as [1, Lemma 4.2] states it after Willems et al. [4]
(reference 37 of [1]), needs order $t + n(\mathcal B)$ for the columns of $H_t$ to span every length-$t$
trajectory, with $n(\mathcal B)$ the order of a minimal state-space representation. [1, §V-A] assumes order
$T_{ini} + N + n(\mathcal B)$ and $T \ge (m+1)(T_{ini} + N + n(\mathcal B)) - 1$; [6, Theorem 1] asks for
order $n + L$.

- **Differs:** the code checks order $L$, not $L + n$ (D13).
- **Several datasets:** [6, Definition 2, eq. (8)] combines them in a mosaic-Hankel matrix
  $[H_k(u^1)\ \cdots\ H_k(u^q)]$, one Hankel block per dataset, and [6, Theorem 2] extends the lemma to it
  under "collectively persistently exciting" inputs of order $n + L$. The closed loop builds its Hankel
  matrix from one open-loop run, one trajectory, so no column joins two (D9, closed 2026-11-17).
- Both statements are for linear time-invariant systems.

## 4. The DeePC quadratic program

`src/deepc/controller.py:88-200`, solved once per step. Decision variables: $g \in \mathbb R^{n_c}$ (one weight
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

- $Q$ and $R$ are diagonal, `Q_diag` and `R_diag` repeated over the horizon (`:131-135`): the closed
  loop's defaults are $[20, 1, 20, 1]$ and $[0.1, 0.1]$, the old coupled masses' (§5).
  $R$ weighs the inputs themselves, not their changes.
- $\rho$ is the 1-norm (`L1`) or the **squared** 2-norm (`L2`), chosen by `reg_norm_g` and `reg_norm_sigma_y`
  (`:137-145`). The closed loop defaults to squared 2-norm on $g$ and 1-norm on $\sigma_y$, with
  $\lambda_g$ = 10 and $\lambda_y = 10^4$; it sets no $\lambda_{out}$ (the controller's default 0), so no
  output bound acts, and hands the controller $\pm\infty$ as the rate bounds.
- $u_{ini}, y_{ini}$ are the last $T_{ini}$ applied inputs and measured outputs, samples $k - T_{ini}, \dots,
  k - 1$; $u_{prev}$ is the last applied input (`devkit/harness/closed_loop/run.py`, §5); $\tau$ is the
  noise-adaptive tightening (§6), zero in every run since no piece reaches §6.
- The Hankel blocks, $u_{ini}$, $y_{ini}$, $y_{ref}$, $u_{prev}$, $\lambda_g$, $\lambda_y$ and $\tau$ are
  CVXPY parameters, so the problem is built once and re-solved with new values.
- Solver (`src/deepc/controller.py:243-266`): OSQP with `eps_abs` = `eps_rel` = $10^{-5}$, `max_iter` = 10000, polishing, warm start.
  A `cvxpy.SolverError` re-solves with SCS, and the solve names the solver that answered; any other exception
  reaches the caller (D3). With the closed loop's $\pm\infty$ rate bounds, SCS answered
  `unbounded_inaccurate` with no solution where OSQP found one (D18). With no solution, the controller
  returns $u = 0$, flagged on the solve; the closed loop stops on the flag and applies nothing (§5). The
  statuses and solvers are counted per run (`devkit/harness/closed_loop/run.py`, `solve_counts`; D2, D3).
  Otherwise the first predicted input $u_0$ is applied (`src/deepc/controller.py:296`).

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
- **Output timing:** [1, eq. (1)] has $y(t) = Cx(t) + Du(t)$. The data and the loop both measure
  $y_k = h(x_k)$ plus noise, from the state at the same sample (§5, §11); with $D = 0$, since $h$ names no
  input.
- **As in [1, eq. (6)], the reference:** the loop tracks a typed reference $r$ with the system's outputs,
  $y_{ref} = (r_k, \dots, r_{k+N-1})$ (§5). Until 2026-11-17 the outputs were errors and the cost reference
  zero (D15, closed).

## 5. The closed loop

`devkit/harness/closed_loop/run.py`, on a saved open-loop run (§11), which it runs again and checks
(`devkit/harness/open_loop/run.py`, `open_loop_of`). The run's inputs and measured outputs are the
controller's data (§3, §4). The loop's plant is the run's plant: the same system, $T_s$, actuator bounds,
$\sigma$, $\sigma_d$ and integrator tolerances, sampled by the same two functions, `measure` and `step`
(`tools/system/step.py`), so the output the data holds and the one the loop gives the controller come from
one formula (D4, D10, D12, closed 2026-11-17). The reference is typed as one row per output in $t$
(`tools/system/parse.py`, `parse_signal`: the functions of §10 and `Heaviside`, $\tfrac12$ at 0). From
$x_0$ (none: the open-loop run's), for $k = 0, \dots, T_{ini} + K - 1$, $K$ = `int(duration / T_s)`:

```math
y_k = h(x_k) + n_k, \qquad
u_k = \mathrm{clip}(\hat u_k, u_{min}, u_{max}), \qquad x_{k+1} = \Phi_{T_s}(x_k, u_k, d_k),
```

with $n_k$ and $d_k$ drawn as in §11 from the loop's own generator, seeded by its own `seed` (default 1,
not the data's 42; the developer, 2026-11-17), the noise then the disturbance. For $k < T_{ini}$,
$\hat u_k = 0$ (the developer, 2026-11-17). After that $\hat u_k$ is the first input of §4, solved with
$u_{ini}, y_{ini}$ the samples $k - T_{ini}, \dots, k - 1$ and $y_{ref} = (r(t_k), \dots, r(t_{k+N-1}))$,
$t_k = k T_s$. The past window ends at $k - 1$ because a Hankel column pairs $u_j$ with $y_j = h(x_j)$ and
$u_k$ is not yet chosen; [1, Algorithm 1] is not re-read for this (the papers are not in every checkout). A
solve that leaves no solution stops the run, naming the sample, the status and the solver (the developer,
2026-11-17). The controller is used as it is (the developer, 2026-11-17: no maths edits yet).

## 6. Noise-adaptive regularisation (`noise_adaptive`)

**No piece reaches this since 2026-11-17:** the closed loop does not offer it (the developer's 3d,
`docs/project/design.md`); its code stays in `src/deepc/` until the maths phase (D8). As the old loop ran
it: `src/deepc/noise_estimator.py`, `c7f4569:devkit/sim/simulation.py:113-127`. From the second solve on, the residual between the
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
with a finite bound (`src/deepc/controller.py:180-197`). At every plant's default it enters no constraint, and
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

## 7. Online Hankel window (`online_hankel`)

**No piece reaches this since 2026-11-17** (as §6). As the old loop ran it: `src/deepc/online_hankel.py`,
`c7f4569:devkit/sim/simulation.py:150-154`.

- **Size.** The window holds `max_cols` columns: `hankel_window_size`, or the number of offline columns when
  that is 0, the default (`:54-55`). A larger window is padded with copies of the last offline column
  (`:59-70`).
- **First columns.** A buffer is seeded with the last $L - 1$ training samples (`:79-82`).
- **Each step from `hankel_warmup_steps` on** (default 100), the step's input and output are appended. A
  column is formed from the last $L$ buffered samples (`:85-110`). It replaces the next padding column while
  any is left; otherwise every column shifts left by one and the new one takes the last place.
- **Into the QP.** The four blocks are re-split and passed to the QP's parameters (`src/deepc/controller.py:202-210`).
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

`devkit/harness/closed_loop/run.py`, each a reduction of an array on the record
(`devkit/harness/closed_loop/record.py`):

- **RMSE per output**, of $h(x_k) - r_k$, the plant's output before the noise, over all $T_{ini} + K$
  samples, warm-up included.
- **Solves by outcome** (`solve_counts`): per status, matched exactly, and per solver that answered (D2, D3).
- **Solve time**: mean, standard deviation and maximum wall-clock time of the `solve` call.
- **The slack**: the mean and maximum of $\lVert \sigma_y \rVert_2$, whichever norm its penalty uses.
- **Clipped inputs**: the samples where the plant's clip changed the controller's input.

The controller's excitation rank and the rank it needs (§3) are kept with the run's constants.

## 9. Tuning (deleted 2026-11-17)

`c7f4569:tune.py` searched the old controller's settings with Optuna's TPE sampler, scoring one metric of
the old loop. Deleted with it (the developer's 5a, `docs/project/design.md`: a sweep in the Lab covers
"different configurations").

## 10. The typed system (`system`, the rewrite's first piece)

`tools/system/`, the first piece of the rewrite in `docs/project/design.md`; it replaces the plant class of
§1.

**What is typed** (`tools/system/parse.py`). One row per state equation, $\dot x_i = f_i(x, u, d,
\theta)$, one per output equation, $y_j = h_j(x, \theta)$, and the values of the parameter vector
$\theta$. A row is an expression in the states `x1…xn`, the inputs `u1…um`, the parameters
`theta1…thetaq`, in a state row the disturbance inputs `d1…dr` (where the unmodelled disturbance
enters; none is allowed in an output row), and the functions `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `tanh`, `abs` and the
constant `pi`; `^` is a power. The sizes: $n$ the number of state rows, $p$ the number of output
rows, $q$ the number of $\theta$ values, $m$ the highest input index, $r$ the highest disturbance
index (0 for none); every input and disturbance must act in a state row as SymPy reads it. Refused, in words: any other
name or character, a `.` outside a number, an empty row, a state index above $n$, a parameter index
above $q$, a $\theta$ value that is not finite or acts in no row as SymPy reads it, an input in an
output row, an input that acts in no state row as SymPy reads it, a row that reads as no
expression (`()`), as not real (`sqrt(-1)`) or with a constant too large for a float. Each row is checked token by token before SymPy's
`parse_expr` reads it (which evaluates Python); every number is read as a float, so nothing is
computed exactly; `lambdify` makes NumPy functions $f(x, u)$, $h(x)$ at the $\theta$ values. The
record keeps the rows as SymPy read them, with $\theta$ as symbols, and the sizes. Nothing is
simulated here: the initial state, the input bounds and the sampling time are the open-loop
simulation's (`docs/project/design.md` §2).

**The analysis at a requested point** (`tools/system/linearize.py`, `devkit/harness/system/run.py`;
the developer, 2026-11-11: "the linearization is not fixed but defined by the user during plot
request"). At $(x^*, u^*)$ (default the origin) and $d = 0$, the system is linearised exactly,
$A = \partial f/\partial x$, $B = \partial f/\partial u$, $C = \partial h/\partial x$ at the point, by
SymPy's derivatives; $D = 0$, since $h$ names no input. The point need not be an equilibrium; the
residual $\lVert f(x^*, u^*, 0) \rVert_2$ says how far it is. For the chosen pair $u_j \to y_i$,
$G(s) = C_i (sI - A)^{-1} B_j$, python-control (0.10.2, by function) computes: the poles and their
damping ratios (`damp`), the frequency response over its own range of $\omega$
(`frequency_response`; the phase unwrapped and shown in degrees), the step response over its own
time span (`step_response`), the impulse and ramp ($u_j - u_j^* = t$) responses over that span
(`impulse_response`, `forced_response`), the DC gain $G(0)$ (`dcgain`) and the gain and phase
margins (`stability_margins`). A number with no finite answer (an infinite margin, the DC gain of
an integrator) is left out, the reason recorded.

**Integration** (`tools/system/integrate.py`, used by the open-loop simulation, branch 2). Over a
step the input is held constant (zero-order hold) and the state integrated by
`scipy.integrate.solve_ivp`, method RK45, the Dormand–Prince 5(4) pair [7], at rtol $10^{-9}$ and
atol $10^{-12}$ (the developer's decision, 2026-11-11). A failed solve raises with the solver's
message.

**Checked against** (`tests/test_system.py`): the exact zero-order-hold sampling of a linear
system, $x_{k+1} = A_d x_k + B_d u_k$ with $\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} =
\exp\!\left(\begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} T_s\right)$ [8], on the two masses (§1)
typed with $\theta = [k, k_{12}, c, c_{12}]$; and the closed forms of $\dot x_1 = -\theta_1 x_1$ and
$\dot x_1 = u_1$.

## 11. The open-loop simulation (`open_loop`, the rewrite's second piece)

`devkit/harness/open_loop/`, on a saved system run (§10), which it reads again and checks
(`devkit/harness/system/run.py`, `system_of`). From $x_0$, for $k = 0, \dots, T - 1$
(`devkit/harness/open_loop/run.py`):

```math
y_k = h(x_k) + n_k, \qquad n_k \sim \mathcal N(0, \sigma^2 I), \qquad
w_k \sim \mathcal U(u_{exc,min}, u_{exc,max}), \qquad
u_k = \mathrm{clip}(w_k, u_{min}, u_{max}), \qquad
d_k \sim \mathcal N(0, \mathrm{diag}(\sigma_d^2)), \qquad x_{k+1} = \Phi_{T_s}(x_k, u_k, d_k),
```

with $\Phi_{T_s}$ the integration of $\dot x = f(x, u_k, d_k, \theta)$ over one sampling time with
$u_k$ and $d_k$ held (§10, `tools/system/step.py`, the same `measure` and `step` the closed loop
samples its plant by, §5); $d_k$ is the unmodelled disturbance, entering
where the system's rows name it (the developer, 2026-11-11). **Departure, declared:** $d_k$ has a
fixed standard deviation per sample and is held over $T_s$; it is not the sampling of a
continuous-time white noise of fixed intensity, whose per-sample variance scales with $1/T_s$, so
the same $\sigma_d$ pushes the plant differently at another $T_s$. The output at sample $k$ is taken from the state at sample
$k$, as [1, eq. (1)]: $y(t) = Cx(t) + Du(t)$. The excitation $w_k$ is drawn i.i.d. uniform per
input on its own range, as [1, §VI-B] draws "the input sequence from a uniformly distributed
random variable" (its range is not stated there; both read 2026-11-11 in the arXiv HTML rendering
of v2, not the PDF). The actuator bounds $u_{min}, u_{max}$ ($\pm\infty$ for none) clip it; the
record holds the input the plant applied, $u_k$, and counts the samples the clip changed. No
baseline controller acts (the developer, 2026-11-11). The excitation and the noise come from one
generator seeded by `seed`: the excitation, then the measurement noise, then the disturbance. The true states $x_k$ are recorded for the
charts; the controller never reads them, and the record also keeps the system as read (rows,
θ, sizes), not only its file's path. The time per sampling step's integration is recorded: mean,
standard deviation and maximum.

**Its numbers.** The PE order of $u$: the largest $L$ at which $H_L(u)$ (§3) has full row rank $Lm$,
the input persistently exciting of order $L$ [1, Def. 4.4], by the controller's own check
(`src/deepc/regularization.py`, `check_persistent_excitation`, numerical rank at tolerance
$10^{-6}$), bisected over $L \le \lfloor (T+1)/(m+1) \rfloor$, where $H_L(u)$ has at least as many
columns as rows (full row rank at $L$ implies it at $L - 1$). The SNR of each output: the variance
of the noise-free $h(x_k)$ over the samples, divided by $\sigma^2$; left out when $\sigma = 0$. The
clipped input samples; the step time.

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
7. J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta formulae", *Journal of Computational and
   Applied Mathematics* 6(1), 1980: as SciPy's `solve_ivp` documentation names it for RK45. Not read.
8. C. F. Van Loan, "Computing integrals involving the matrix exponential", *IEEE Transactions on
   Automatic Control* 23(3), 1978. Not read.

Software, at the versions `uv.lock` pins: CVXPY 1.8.1, OSQP 1.1.1, SCS 3.2.11, NumPy 2.4.2, SciPy 1.17.1.
