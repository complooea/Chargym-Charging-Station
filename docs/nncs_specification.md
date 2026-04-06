# NNCS Specification: EV Charging Station Control

This document specifies a neural network control system (NNCS) for a
multi-slot EV charging station, written in the canonical plant / neural
controller / disturbance form used by tools such as Sherlock, Verisig 2.0,
ReachNN\*, NNV, JuliaReach, OVERT, and RINO.

---

## 1. Overview

The system controls charging and discharging (V2G) of up to 10 electric
vehicles at a station over a 24-hour horizon. At each hourly step a neural
controller reads a 28-dimensional observation — current and forecast solar
radiation and electricity price (all normalized to $[0,1]$), plus per-slot
state-of-charge and time-to-departure — and outputs a normalized power
command for each slot. The plant then updates the battery states and
decrements the time-to-departure counters; the exogenous signals advance
along a fixed, deterministic per-episode trajectory.

All uncertainty is placed on the plant side:

- **Exogenous signals** (radiation, price, 3-step-ahead forecasts) are
  *deterministic within an episode*. Across-episode uncertainty is
  captured by a set $\bar{\mathcal{E}}$ of admissible exogenous
  trajectories, which may be finite or infinite.
- **Arrival SOC** of a newly plugged-in vehicle is a bounded sample in
  $[0.20, 0.49]$ — this is the only genuine per-step stochastic input.
- **Presence** is not an independent signal: it is derived from the
  time-to-departure state.

The neural controller $N$ is a deterministic function of its 28-dim input.

---

## 2. Signals

### 2.1 State

$$
x_k \in [0,1]^{8} \times [0,1]^{10} \times [0, \tfrac{9}{24}]^{10}
\subset \mathbb{R}^{28},
\qquad
x_k = \begin{bmatrix} e_k \\ s_k \\ \tau_k \end{bmatrix},
$$

with the partition

| Block     | Dims     | Symbol   | Range           | Meaning                                    |
|-----------|----------|----------|-----------------|--------------------------------------------|
| Exogenous | 1        | $e_k^{(1)}$ | $[0,1]$ | current solar radiation, $\text{Radiation}[t]/1000$  |
| Exogenous | 2        | $e_k^{(2)}$ | $[0,1]$ | current electricity price, $\text{Price}[t]/0.1$     |
| Exogenous | 3        | $e_k^{(3)}$ | $[0,1]$ | radiation forecast 1h ahead, $\text{Radiation}[t{+}1]/1000$ |
| Exogenous | 4        | $e_k^{(4)}$ | $[0,1]$ | radiation forecast 2h ahead, $\text{Radiation}[t{+}2]/1000$ |
| Exogenous | 5        | $e_k^{(5)}$ | $[0,1]$ | radiation forecast 3h ahead, $\text{Radiation}[t{+}3]/1000$ |
| Exogenous | 6        | $e_k^{(6)}$ | $[0,1]$ | price forecast 1h ahead, $\text{Price}[t{+}1]/0.1$          |
| Exogenous | 7        | $e_k^{(7)}$ | $[0,1]$ | price forecast 2h ahead, $\text{Price}[t{+}2]/0.1$          |
| Exogenous | 8        | $e_k^{(8)}$ | $[0,1]$ | price forecast 3h ahead, $\text{Price}[t{+}3]/0.1$          |
| SOC       | 9–18     | $s_k^i$, $i=1..10$ | $[0,1]$ | state-of-charge per slot                |
| Time      | 19–28    | $\tau_k^i$, $i=1..10$ | $[0, \tfrac{9}{24}]$ | normalized time-to-departure |

So $e_k \in [0,1]^8$, $s_k \in [0,1]^{10}$, $\tau_k \in
[0,\tfrac{9}{24}]^{10}$. The layout is **block-contiguous, not
interleaved**: dim 1 is current radiation, dim 2 is current price, dims
3–5 are the 1/2/3-hour radiation forecasts, and dims 6–8 are the
1/2/3-hour price forecasts. (In 0-indexed form matching the reference
implementation: index 0 = radiation, index 1 = price, indices 2–4 =
radiation lookahead, indices 5–7 = price lookahead.) All are perfect
forecasts — the exact future values, not noisy predictions.

All exogenous coordinates are normalized: raw solar radiation (in W/m²)
is divided by 1000, and raw electricity price is divided by 0.1. The
$[0,1]$ upper bound on radiation is a hard physical limit for any
weather trace in the target regime; the $[0,1]$ bound on price holds
for the four hard-coded price flags used by the reference environment
(peak prices reach exactly 0.10, normalizing to 1.0). If a future price
schedule exceeds 0.10 raw the bound should be relaxed accordingly.

**Derived presence** (not a state, used only for case splits):

$$
\pi_k^i \;\triangleq\; \mathbb{1}[\tau_k^i > 0].
$$

### 2.2 Control input

$$
u_k \in [-1,1]^{10}, \qquad u_k = N(x_k),
$$

with $u_k^i \ge 0$ interpreted as charging and $u_k^i < 0$ as discharging.

### 2.3 Exogenous disturbance

In the reference environment the exogenous evolution is **fully
deterministic within an episode**: solar radiation comes from a weather
trace loaded once at `reset()`, and electricity price comes from a
hard-coded 24-hour profile selected by a `price_flag`. Neither is
resampled per step, and the two signals are generated independently.
The 3-step-ahead forecasts exposed in the observation are the *true*
future values (perfect foresight), not noisy predictions. So from the
perspective of verification, the exogenous block is an **episode
parameter**, not a per-step disturbance.

Formally, each episode is parameterized by a fixed exogenous trajectory

$$
\bar{e} = (\bar e_0, \bar e_1, \dots, \bar e_{T-1}) \in
\big([0,1]^8\big)^T,
$$

where $\bar e_k^{(1)} = \text{Radiation}[k]/1000$, $\bar e_k^{(2)} =
\text{Price}[k]/0.1$, and the remaining components are the perfect
3-step lookaheads into the same profiles (dims 3–5 for radiation,
dims 6–8 for price; see §2.1).
For verification, $\bar{\mathcal{E}}$ denotes the chosen set of
admissible episode trajectories. It may be a finite catalog of traces
or an infinite family defined by bounds or other constraints.

The only per-step, genuinely stochastic quantity is the **arrival-SOC
sample** drawn when a car plugs into an empty slot. We keep that as the
disturbance:

$$
\beta_k \in [0.20,\,0.49]^{10},
$$

where $\beta_k^i$ is consumed only when $\alpha_k^i = 1$ and is ignored
otherwise. (In the reference env the draw is uniform on the 30-element
grid $\{0.20, 0.21, \dots, 0.49\}$; for set-based reachability it is
sound and standard to over-approximate this by the continuous interval
$[0.20, 0.49]$.)

Putting the two together, the exogenous coordinates of the state follow
the deterministic recursion

$$
e_{k+1} = \bar e_{k+1},
$$

with $\bar e$ drawn from $\bar{\mathcal{E}}$ at episode start, while the
SOC block takes $\beta_k$ as its only disturbance input at the per-step
level.

### 2.4 Schedule parameters

Per car $i$ and per step $k$, known from the problem instance:

- $\alpha_k^i \in \{0,1\}$ — **arrival indicator** for the $k \to k{+}1$
  transition. Well-posedness: $\alpha_k^i = 1 \Rightarrow \tau_k^i = 0$.
  In the reference environment, arrivals are only attempted for
  $k \in \{0,\dots,20\}$, so $\alpha_k^i = 0$ for $k \ge 21$.
- $\delta_k^i \in \{\tfrac{4}{24}, \tfrac{5}{24}, \dots, \tfrac{9}{24}\}$
  — normalized stay duration assigned when $\alpha_k^i = 1$; unused
  otherwise. This matches the reference implementation, which samples a
  departure hour with `randint(hour + 4, min(hour + 10, 25))`, so the
  realized stay is at least 4 hours and at most 9 hours. More precisely,
  the admissible support is time-dependent near the end of the horizon:
  for $k \le 16$, $\delta_k^i \in \{\tfrac{4}{24},\dots,\tfrac{9}{24}\}$;
  for $k = 17$, $\delta_k^i \in \{\tfrac{4}{24},\dots,\tfrac{7}{24}\}$;
  for $k = 18$, $\delta_k^i \in \{\tfrac{4}{24},\tfrac{5}{24},\tfrac{6}{24}\}$;
  for $k = 19$, $\delta_k^i \in \{\tfrac{4}{24},\tfrac{5}{24}\}$; and
  for $k = 20$, $\delta_k^i = \tfrac{4}{24}$.

These are the only exogenous schedule quantities, because presence and
departure are both recoverable from $\tau$.

### 2.5 Constants

| Symbol     | Value  | Meaning                          |
|------------|--------|----------------------------------|
| $C$        | 30 kWh | EV battery capacity              |
| $P_{\max}$ | 10 kW  | charger power rating (kWh/hour)  |
| $T$        | 24     | horizon length (hours)           |
| $N_\text{cars}$ | 10 | number of charging slots        |

---

## 3. Neural controller

The controller is a deterministic feed-forward neural network

$$
N : \mathbb{R}^{28} \longrightarrow [-1,1]^{10},
\qquad u_k = N(x_k).
$$

All uncertainty enters the closed loop through the plant-side
disturbance $\beta_k$, the episode trajectory $\bar e$, and the
schedule parameters $(\alpha_k,\delta_k)$; the controller itself has no
uncertain inputs. Standard NN verification backends (Sherlock,
Verisig 2.0, $\beta$-CROWN, NNV star sets, OVERT relaxations) apply to
$N$ unchanged.

---

## 4. Plant dynamics

The plant map has the form

$$
x_{k+1} = f\!\left(x_k,\, u_k,\, \beta_k;\; \bar e_{k+1},\, \alpha_k,\, \delta_k\right),
$$

and decomposes into three independent blocks.

### 4.1 Exogenous block (dims 1–8)

$$
e_{k+1} = \bar e_{k+1}.
$$

The next exogenous vector is the fixed, deterministic value from the
episode's exogenous trajectory $\bar e \in \bar{\mathcal{E}}$, with
$\bar e_{k+1} \in [0,1]^8$. It has no dependence on $x_k$, $u_k$, or
any disturbance. Episode-level uncertainty is handled by running
reachability once per element of $\bar{\mathcal{E}}$.

### 4.2 State-of-charge block (dims 9–18)

Define the rate-limited charge/discharge branch

$$
\phi(s,u) =
\begin{cases}
\min\!\big(P_{\max},\, C(1-s)\big), & u \ge 0, \\[1mm]
\min\!\big(P_{\max},\, C s\big),     & u < 0.
\end{cases}
$$

Then, for each slot $i \in \{1,\dots,10\}$,

$$
s_{k+1}^i =
\begin{cases}
s_k^i + \dfrac{1}{C}\, u_k^i\, \phi(s_k^i, u_k^i),
& \tau_k^i > 0 \quad \text{(present: charge/discharge)} \\[3mm]
\beta_k^i,
& \tau_k^i = 0 \;\wedge\; \alpha_k^i = 1 \quad \text{(arrival)} \\[2mm]
0,
& \tau_k^i = 0 \;\wedge\; \alpha_k^i = 0 \quad \text{(absent)}
\end{cases}
$$

**Departure-hour observation (one-step lag).** The SOC is *not* frozen
after departure, but the departure hour itself carries a one-step lag
that is worth spelling out. Consider a car present during hours
$h,\dots,d-1$:

- At $k = d-1$ (last present step, $\tau_k^i = \tfrac{1}{24}$): the
  charging branch fires and writes $s_d^i = s_{d-1}^i + \tfrac{1}{C}
  u_{d-1}^i \phi(s_{d-1}^i, u_{d-1}^i)$. Simultaneously $\tau_d^i = 0$.
- At $k = d$ (departure hour, first absent step): the observation reads
  the just-written $s_d^i$ (the post-charging value), but because
  $\tau_k^i = 0$ the absent branch fires and writes $s_{d+1}^i = 0$. The
  controller's action at $k=d$ has no effect on the state.
- At $k \ge d+1$: $s_k^i = 0$ for the rest of the horizon (until any
  later arrival into the same slot).

So the post-charging SOC is visible to the controller for exactly one
step after departure, then drops to $0$. This matches the reference
implementation, where the `BOC` matrix is zero-initialized and columns
past departure are never written.

The "present" branch covers the last present step uniformly with the
rest of the stay; the transition to the absent branch happens at
$k = d$, driven by $\tau_d^i = 0$ and $\alpha_d^i = 0$ (by
well-posedness, $\alpha_k^i = 0$ whenever $\tau_k^i > 0$, and the
arrival of a *new* car into the same slot cannot coincide with the
departure hour).

### 4.3 Time-to-departure block (dims 19–28)

For each slot $i$,

$$
\tau_{k+1}^i =
\begin{cases}
\tau_k^i - \dfrac{1}{24},
& \tau_k^i > 0 \quad \text{(count down; hits 0 at departure)} \\[3mm]
\delta_k^i,
& \tau_k^i = 0 \;\wedge\; \alpha_k^i = 1 \quad \text{(arrival)} \\[2mm]
0,
& \tau_k^i = 0 \;\wedge\; \alpha_k^i = 0 \quad \text{(absent)}
\end{cases}
$$

By construction, while a car is present $\tau_k^i \in
\{\tfrac{1}{24},\dots,\tfrac{9}{24}\}$, with newly assigned stays in
$\{\tfrac{4}{24},\dots,\tfrac{9}{24}\}$. The countdown then reaches
$\tfrac{1}{24}$ at the last present step and drops to $0$ at departure.

---

## 5. Closed-loop NNCS

$$
\boxed{\;
\begin{aligned}
x_{k+1} &= f\!\left(x_k,\, u_k,\, \beta_k;\; \bar e_{k+1},\, \alpha_k,\, \delta_k\right) \\[1mm]
u_k     &= N(x_k) \\[1mm]
\beta_k &\in [0.20,\,0.49]^{10} \\[1mm]
\bar e  &\in \bar{\mathcal{E}} \quad \text{(fixed per episode)} \\[1mm]
x_0     &\in \mathcal{X}_0(\bar e) \\[1mm]
k       &= 0, 1, \dots, T-1
\end{aligned}
\;}
$$

### 5.1 Initial set $\mathcal{X}_0(\bar e)$

Given a chosen episode trajectory $\bar e \in \bar{\mathcal{E}}$:

- $e_0 = \bar e_0$ (a single point, not a set: current radiation and
  price plus perfect 3-step lookaheads, all in $[0,1]$).
- $s_0^i = 0$ for slots with no car present at $k=0$;
  $s_0^i \in [0.20, 0.49]$ for slots that start occupied (this is the
  only genuine set component of $x_0$, coming from the initial-SOC
  sample of a car that is plugged in at $k=0$).
- $\tau_0^i = 0$ when empty; in the reference environment, an occupied
  slot at $k=0$ arises only from an arrival sampled at hour $0$, so
  $\tau_0^i \in \{\tfrac{4}{24},\dots,\tfrac{9}{24}\}$ for occupied slots.

### 5.2 Per-step set propagation

At each $k$:

1. Evaluate the NN on the current reachable set:
   $U_k = N(X_k)$, using any NN reachability backend.
2. Push through the plant:
   $X_{k+1} = f(X_k, U_k, [0.20,0.49]^{10};\, \bar e_{k+1}, \alpha_k,
   \delta_k)$, with the three blocks handled independently per slot.
3. Optionally intersect with invariants (e.g. $s_k^i \in [0,1]$,
   $e_k \in [0,1]^8$) to tighten.

The exogenous block is a pure point assignment $e_{k+1} \leftarrow \bar
e_{k+1}$ independent of $X_k$, so it introduces no wrapping error. The
SOC block contains the only nonlinearity (through $\phi$) and is the
main driver of conservatism.

---

## 6. Verification properties

Typical safety / performance queries on this NNCS, evaluated per chosen
episode $\bar e \in \bar{\mathcal{E}}$:

1. **Physical feasibility.** For all $k$ and all $i$,
   $s_k^i \in [0,1]$ and $\tau_k^i \in [0, \tfrac{9}{24}]$.
2. **Departure-SOC guarantee.** For every slot $i$ and its scheduled
   departure step $d_i$, $s_{d_i}^i \geq s_{\min}$ for all admissible
   arrival-SOC samples $\beta \in \prod_k [0.20,0.49]^{10}$.
3. **Penalty bound.** The episode penalty
   $\sum_i \big(2(1 - s_{d_i}^i)\big)^2$ stays below a threshold for all
   admissible $\beta$.
4. **Action saturation avoidance** (optional): the controller does not
   saturate at $\pm 1$ more than a given fraction of steps in reach.

A global guarantee across exogenous conditions is obtained by running
the above for every $\bar e \in \bar{\mathcal{E}}$.

---

## 7. Structural notes for tool integration

**All uncertainty is plant-side.** The NN has no uncertain inputs. The
remaining uncertainty splits into two layers: the per-step arrival-SOC
disturbance $\beta_k \in [0.20, 0.49]^{10}$, and the episode-level
exogenous trajectory $\bar e \in \bar{\mathcal{E}}$ together with the
schedule parameters $(\alpha_k, \delta_k)$. This matches the canonical
NNCS form and keeps the NN reachability problem as small as possible.

**Deterministic exogenous evolution.** Within an episode, the exogenous
block evolves as a pure point sequence $e_{k+1} = \bar e_{k+1}$ during
verification. Cross-episode uncertainty is represented by the set
$\bar{\mathcal{E}}$ of admissible deterministic trajectories rather than
by enlarging the per-step disturbance set. This is much tighter than a
set-valued-per-step formulation would be.

**Normalized, bounded state.** All 28 state components are bounded:
$e_k \in [0,1]^8$, $s_k \in [0,1]^{10}$, $\tau_k \in
[0,\tfrac{9}{24}]^{10}$. These bounds can be used as invariants to
intersect reachable sets and tighten propagation.

**State-derived presence.** $\pi_k^i = \mathbb{1}[\tau_k^i > 0]$ is not
stored or tracked separately. Under set propagation, $\tau$ is the single
source of truth: presence can never drift out of sync with the state, and
the mode-switch guards are affine conditions on a single state coordinate
($\tau_k^i > 0$, $\tau_k^i = 0$, $\tau_k^i = \tfrac{1}{24}$).

**Per-car decoupling.** Conditional on $(\alpha_k, \delta_k)$, the $s$ and
$\tau$ updates for different slots are independent. Only the controller
$N$ couples them. This factoring helps reachability backends avoid
spurious cross-slot dependencies.

**Piecewise-linear nonlinearity.** The only nonlinear component of $f$ is
$\phi(s,u)$, which is a two-level $\min$ of affine expressions gated by
the sign of $u$. OVERT abstracts this class of $\min / \max$ switches
natively into sound piecewise-linear relaxations; MILP-based backends
encode each branch as a mode.

**Fixed vs. robust schedule.** As written, $(\alpha_k, \delta_k)_k$ is a
fixed instance parameter: one verification run per schedule. For the
reference environment, admissible schedules should also respect the
generation logic used at `reset()`: an arrival can occur only while the
slot is absent, and new arrivals are attempted only for hours
$k \in \{0,\dots,20\}$. To verify over a *set* of schedules, lift
$\alpha_k^i$ (and optionally $\delta_k^i$) into the disturbance as
integer-valued non-deterministic inputs and let the reachability engine
branch over modes, while preserving those support constraints. Presence
remains derived from $\tau$ and does not need to be lifted, because it
was never independent to begin with.
