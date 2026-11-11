# Target design — from the scope, not from today's code

Draft, 2026-11-11, for the developer to edit. Each choice they made is marked **Decided**, with
the date; none is open. Today's code is not a
constraint: anything not in the scope may be thrown out or rewritten.

## The scope (the developer, 2026-11-11)

> define plant → simulate plant with chosen configurations → (attempt to) control the plant with
> deepc under different configuration. Literally everything else can be thrown out the window.

Three pieces, each a Lab form that saves one record, each reading the one before:

```
system  ──>  open_loop  ──>  closed_loop
(define)     (simulate)      (control with DeePC)
```

The middle piece is today's `dataset`, renamed **Open loop** when branch 2 rewrites it (the
developer, 2026-11-11: "dataset sounds weird").

## Fixed directions (the developer's words)

1. **Typed ODEs in the Lab**: $\dot x = f(x, u, \theta)$, $y = h(x, \theta)$, one row per
   equation, with a parameter vector $\theta$ (2026-11-11, after seeing branch 1: "rows for each
   state equation and each observation equation ... the things that get defined in the system
   tab are x, y, u, theta").
2. **Track the state, not an error**: the controller sees the measured outputs $y$ and a
   reference $r$ on them, as in Coulson et al. [1, eq. (6)]: cost $\lVert y_k - r_{t+k} \rVert_Q^2$.
3. **Names**: inputs (manipulated variables) are $u$, outputs (measurements) are $y$, true states
   are $x$ — channels `u1, u2, …`, `y1, …`, `x1, …` — everywhere: axis labels, chart names,
   buttons, records. **No units anywhere.**
4. Order of work: system generator → state tracking → words (here they fold into one rewrite; see
   the last section).

## 1. `system` — define the plant

**The form** (decided 2026-11-11). Labelled rows with + and ✕: one per state equation (`x1' =`,
…), one per output equation (`y1 =`, …), one per parameter value (`θ1 =`, …). The rows name
`x1…`, `u1…`, `theta1…`; $n, m, p, q$ are read off them. **θ's values are set here**, so a system
run is a complete, fixed plant. No initial state, no input bounds, no sampling time, no chart: those
are the open-loop simulation's (§2).

**The record.** The rows as SymPy reads them (θ as symbols) and the sizes $n, m, p, q$.

**Parsing.** SymPy (`parse_expr`, then `lambdify` to NumPy). `parse_expr` evaluates Python, so
each row is checked token by token first: the symbols, a list of functions, numbers, operators;
anything else is refused in words. Every number is read as a float (an exact integer power such as
`9^9^9` would not finish).

**Integration** (used from §2 on). The input held constant over each step (zero-order hold), $x$
integrated over $[kT_s, (k+1)T_s]$ by `scipy.integrate.solve_ivp`, RK45 (Dormand–Prince 5(4)).

Measured 2026-11-11 (scratch script, not in the repo): the coupled masses' linear dynamics
($m_i = 1$, $k_i = 2$, $k_{12} = 1$, $c_i = 0.3$, $c_{12} = 0.1$), $T_s = 0.05$, 600 steps, $x_0 =
[0.5, 0, -0.5, 0]$, inputs i.i.d. uniform on $[-5, 5]$ held per step, seed 42, against the exact
zero-order-hold sampled system (Van Loan 1978, augmented matrix exponential). Largest state error
over the run (states range −1.7 … 2.3) and time per step:

| integrator | max error | time per step, mean ± std |
|---|---|---|
| forward Euler at $T_s$ (today's plants) | 0.54 | — |
| RK45, rtol 1e-3, atol 1e-6 (SciPy's defaults) | 1.2e-8 | 0.18 ± 0.18 ms |
| RK45, rtol 1e-6, atol 1e-9 | 2.3e-9 | 0.11 ± 0.13 ms |
| RK45, rtol 1e-9, atol 1e-12 | 3.4e-10 | 0.15 ± 0.09 ms |

**Decided (1a)**, the developer, 2026-11-11: default tolerances rtol 1e-9, atol 1e-12, both
settings of the open-loop simulation.

## 2. `open_loop` (today's `dataset`, renamed) — simulate the plant

**The form.** The system run; $x_0$; the input bounds $u_{min}, u_{max}$ (the actuator limits,
which clip $u$ before it is held); $T_s$; $T$ samples; the excitation; the measurement noise; the
seed; the integrator tolerances (the developer, 2026-11-11: "Bounds, initial conditions, sampling
rates etc live in the open loop tab").

**What it does.** From $x_0$, clip $u_k$ to the bounds, hold it, integrate one step, record
$y_k = h(x_{k+1}) + n_k$,
$n_k \sim \mathcal N(0, \sigma^2 I)$. Records $u$, $y$, and the true $x$ (for the charts only —
the controller never reads $x$).

**Decided (2a)**, 2026-11-11: one trajectory from $x_0$. Several episodes stacked end to end would
put columns across a reset into the Hankel matrix (D9); episodes come later, combined as the
mosaic-Hankel matrix of [6, eq. (8)].

**Decided (2b)**, 2026-11-11: the excitation is i.i.d. uniform within the input bounds, its
amplitude a fraction of them (a setting). What [1] used in its simulations is read before this is
built (the papers are not on the developer's computer; `references/` is git-ignored).

**Decided (2c)**, 2026-11-11: no baseline controller — open loop, excitation only. An unstable
system's data diverges and the record says so; a typed feedback $u = -K y +$ excitation can come
later on its own branch.

## 3. `closed_loop` — control it with DeePC

**The form.** The open-loop run; the reference $r(t)$; the duration; $x_0$ of the loop (default:
the open-loop run's); DeePC's settings: $T_{ini}$, $N$, $Q$, $R$, $\lambda_g$, $\lambda_y$.

**The controller.** Regularised DeePC as [1, eq. (8)] states it: minimise
$\sum_k \lVert y_k - r_{t+k} \rVert_Q^2 + \lVert u_k \rVert_R^2 + \lambda_g \lVert g \rVert_1 +
\lambda_y \lVert \sigma_y \rVert_1$ subject to the Hankel equality with slack $\sigma_y$ on the past
outputs and $u_k$ within the input bounds. The loop adds the same measurement noise as the data
(D4 gone by construction). Records $u$, $y$, $r$, true $x$, and every solve's status and time.

**Decided (3a)**, 2026-11-11: the reference is typed, $r_i(t)$ one expression per output, with the
system's parser (a step is `Heaviside(t - 5)`).

**Decided (3b)**, 2026-11-11: the low-rank approximation of the data matrix in [1, eq. (8)] is left
out, declared in the code as a departure (`# MISMATCH`); added later on its own branch.

**Decided (3c, 3d)**, 2026-11-11: dropped — the input-rate bounds and soft output bounds, which [1]
does not have (D14), and the two variants, noise-adaptive regularisation and the online Hankel
window, which depart from the papers they cite (D16, D17). Any of them comes back on its own
branch, with a source.

## 4. The Lab

Three pieces, one form each. Charts per run: each $y_i$ with its $r_i$, each $u_j$ with its bounds,
each true $x_i$, solve time. Numbers: RMSE of $y_i - r_i$ per output, solves by status, mean and max
solve time. Names `u1`, `y1`, `x1`; no units.

## 5. What goes

The plant classes and `PlantBase`, the per-plant scenarios, data recipes, default settings, custom
metrics, channel names and units; the error outputs; `tune.py` (**decided (5a)**, 2026-11-11:
dropped — a sweep in the Lab covers "different configurations"); the old safety net, whose frozen
numbers belong to the old pipeline.

**Tests instead of the old frozen numbers**, each against a known answer:
- the integrator against the exact zero-order-hold sampled system, for a typed linear system;
- the parser: what it accepts and refuses;
- DeePC's prediction on noise-free data of a typed linear system: with $g$ solving the Hankel
  equality, the predicted $y$ equals the model's (the fundamental lemma [1, Lemma 4.2]; [6]);
- then a new frozen safety net, made once the three pieces exist.

## 6. How to get there

A rewrite in three branches, each a vertical slice the Lab shows, each judged alone; the new names
and no units from the first line, so the "words" step folds in:

1. **`system` + integrator** — the new first piece (rows, θ), the Lab's row editor, the
   integrator and its tests; the units rule changed on the developer's word. Old pipeline untouched.
2. **`open_loop` on a system run** — the new simulation (x0, bounds, T_s, excitation, noise,
   tolerances); the old `dataset` piece replaced and the name with it.
3. **`closed_loop` on [1, eq. (8)], tracking $y$** — the new controller loop; the old plants,
   scenarios, variants, `tune.py` and safety net deleted; the new safety net made.

**Decided (6a)**, 2026-11-11 (the developer approved the plan of branch 1): the three branches.
Branch 1, `system-generator`, was reworked the same evening after the developer saw it: rows and
θ, no free response, x0 and bounds moved to branch 2.

## References

Numbered as in `METHODS.md`, checked there against the arXiv versions:
[1] J. Coulson, J. Lygeros, F. Dörfler, "Data-Enabled Predictive Control: In the Shallows of the
DeePC", arXiv:1811.05890v2; [6] H. J. van Waarde, C. De Persis, M. K. Camlibel, P. Tesi, "Willems'
Fundamental Lemma for State-space Systems and its Extension to Multiple Datasets",
arXiv:2002.01023v2.

Not yet checked against their source: C. F. Van Loan, "Computing integrals involving the matrix
exponential", IEEE Transactions on Automatic Control 23(3), 1978; J. R. Dormand, P. J. Prince, "A
family of embedded Runge-Kutta formulae", Journal of Computational and Applied Mathematics 6(1),
1980 (the RK45 that SciPy's `solve_ivp` documentation names).
