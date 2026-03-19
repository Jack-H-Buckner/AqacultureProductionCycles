# Profit-Coverage Indemnity: Model and Algorithm

## Overview

This document describes the model and numerical procedure for computing
the profit-coverage indemnity $I(\tau)$ and the optimal rotation policy
under risk aversion for the seasonal aquaculture model.

The indemnity compensates the firm for accumulated production costs plus a
fraction $\xi$ of the opportunity cost of losing the production cycle, where
opportunity costs are evaluated under a breakeven insurance baseline.

The full solution requires four stages, solved in sequence. The first two
stages establish the breakeven baseline, and the second two solve for the
optimal policy under profit coverage.

1. **Stage A** — Solve the risk-averse model with breakeven insurance
   ($\xi = 0$) to determine the baseline optimal policy.
2. **Stage B** — Compute the dollar continuation value $W(t)$ under the
   baseline policy with breakeven insurance.
3. **Stage C** — Compute the profit-coverage indemnity $I(\tau)$ via a
   sequence of three ODE solves.
4. **Stage D** — Re-solve the risk-averse model using the profit-coverage
   payoffs, iterating with Stage C until the policy converges.

---

## Stage A: Baseline risk-averse model (breakeven insurance)

Run `solve_seasonal_model(p)` with breakeven insurance ($\xi = 0$). This is
the existing solver implemented in `03_continuation_value_solver.jl`, using
the breakeven indemnity ODE from Section 9 of the writeup.

### Output

- $V^0(t)$: the continuation value in utility terms (periodic spline),
- $\tilde{V}^0(t_0)$: the cycle value at optimal stocking,
- $\tau^{*0}(t_0)$: the optimal rotation length (periodic spline),
- $d^{*0}(t)$: the optimal fallow duration at each calendar time,
- $t_0^{*0}(t)$: the optimal stocking time after a cycle ends at $t$.

This solution provides the initial policy for all subsequent stages and the
breakeven cash flows needed to compute opportunity costs.

---

## Stage B: Dollar continuation value under breakeven insurance

### Goal

Compute $W(t)$, the expected NPV in dollars of the production system at
calendar time $t$, assuming the firm follows the baseline policy from
Stage A and holds breakeven insurance coverage.

### Breakeven indemnity at each node

For each stocking-date node $t_0$ in the grid, solve the breakeven
indemnity ODE on $[t_0, T^{*0}(t_0)]$:

$$I_0'(\tau) = \left(\frac{\lambda(\tau)}{1-Q} + \delta_b\right) I_0(\tau)
  + \phi(\tau) + \frac{c_I}{1-Q} - \delta_b c_2,
  \qquad I_0(t_0) = c_s + c_2.$$

From $I_0$ compute the breakeven cash flows:

$$Y_H^0(T^*) = v(T^* - t_0) - \Pi_0(T^*, t_0) - \Phi(T^*, t_0)
  - c_s e^{\delta_b(T^* - t_0)} - c_h,$$

$$Y_L^0(s) = I_0(s) - \Pi_0(s, t_0) - \Phi(s, t_0)
  - c_s e^{\delta_b(s - t_0)} - c_2.$$

### Dollar cycle value

The dollar-valued cycle value at stocking date $t_0$ decomposes as

$$\tilde{W}(t_0) = f_W(t_0) + g(t_0) \cdot W(T^*),$$

where:

- $f_W(t_0)$: the same integrals as the existing `compute_Vtilde_decomposed`
  but with $u(Y_H^0) \to Y_H^0$ and $u(Y_L^0) \to Y_L^0$,
- $g(t_0)$: the probability-weighted discount factor (unchanged from the
  existing code — does not involve the utility function or cash flows),
- $W(T^*)$: evaluated from the current $W$ spline.

The continuation value links across the fallow period:

$$W(t) = e^{-\delta\, d^{*0}(t)}\, \tilde{W}(t_0^{*0}(t)).$$

### Algorithm

This mirrors the iterative structure of `solve_seasonal_model` with two
modifications:

1. **Replace $u(\cdot)$ with the identity** in the $f$ component.
2. **Skip all FOC solves** — use the fixed policy $(\tau^{*0}, d^{*0})$
   from Stage A.

Each iteration:

1. Compute $\tilde{W}(t_0)$ at $2N+1$ nodes using breakeven cash flows.
2. Build and solve the linear system
   $$(I - \text{diag}(\alpha \cdot g) \cdot \mathcal{W})\, \mathbf{w}
     = \alpha \cdot \mathbf{f}_W,$$
   where $\alpha_i = e^{-\delta d_i^{*0}}$ and $\mathcal{W}$ is the spline
   interpolation weight matrix.
3. Apply damping and check convergence.

Optionally refine with Bellman fixed-point iterations (Phase 2), evaluating
$W(s)$ at every quadrature point in the loss integral.

### Output

A periodic spline $W(t)$ representing the expected dollar NPV of the
production system under breakeven insurance. This is independent of the
coverage fraction $\xi$ and only needs to be computed once.

---

## Stage C: Profit-coverage indemnity

### Goal

For a given cycle $[t_0, T]$, solve the indemnity

$$I(\tau) = C(\tau) + \xi \cdot OC(\tau),$$

where $C(\tau)$ is the accumulated cost of the production cycle (including
profit-coverage premiums) and $OC(\tau)$ is the opportunity cost of losing
the cycle at time $\tau$, evaluated under breakeven insurance.

### Definitions

The opportunity cost is

$$OC(\tau) = \bar{W}_0(\tau) - W(\tau),$$

where $\bar{W}_0(\tau)$ is the expected dollar value of completing the
current cycle from time $\tau$ onward (conditional on survival to $\tau$),
evaluated with breakeven cash flows, and $W(\tau)$ is the dollar
continuation value from Stage B (the value of starting fresh).

### Step 1: Breakeven indemnity (forward ODE)

Solve on $[t_0, T]$:

$$I_0'(\tau) = \left(\frac{\lambda(\tau)}{1-Q} + \delta_b\right) I_0(\tau)
  + \phi(\tau) + \frac{c_I}{1-Q} - \delta_b c_2,
  \qquad I_0(t_0) = c_s + c_2.$$

Compute $Y_H^0(T)$ and $Y_L^0(\tau)$ from the breakeven solution.

Accumulate the breakeven premium integral as an auxiliary ODE state:

$$P_0'(\tau) = \frac{\lambda(\tau) I_0(\tau) + c_I}{1-Q} + \delta_b P_0,
  \qquad P_0(t_0) = 0,$$

so that $\Pi_0(\tau, t_0) = P_0(\tau)$.

### Step 2: Conditional continuation value (backward ODE)

Solve on $[t_0, T]$ integrating **backward** from $T$:

$$\bar{W}_0'(\tau) = (\delta + \lambda(\tau))\, \bar{W}_0(\tau)
  - \lambda(\tau)\left[Y_L^0(\tau) + W(\tau)\right],
  \qquad \bar{W}_0(T) = Y_H^0(T) + W(T).$$

Here $W(\tau)$ is evaluated from the Stage B spline and $Y_L^0(\tau)$ comes
from Step 1. This can be integrated as a forward ODE in the reversed time
variable $\sigma = T - \tau$:

$$\hat{W}'(\sigma) = -(\delta + \lambda(T - \sigma))\, \hat{W}(\sigma)
  + \lambda(T - \sigma)\left[Y_L^0(T - \sigma) + W(T - \sigma)\right],$$

with $\hat{W}(0) = Y_H^0(T) + W(T)$, integrated forward in $\sigma$
from $0$ to $T - t_0$.

Set $OC(\tau) = \bar{W}_0(\tau) - W(\tau)$.

### Step 3: Profit-coverage indemnity (forward ODE)

Solve on $[t_0, T]$:

$$I'(\tau) = \left(\frac{\lambda(\tau)}{1-Q} + \delta_b\right) I(\tau)
  + \phi(\tau) + \frac{c_I}{1-Q} - \delta_b c_2
  + \xi\left[OC'(\tau) - \delta_b \cdot OC(\tau)\right],$$

$$I(t_0) = c_s + c_2 + \xi \cdot OC(t_0).$$

The term $OC'(\tau) = \bar{W}_0'(\tau) - W'(\tau)$ is known from Step 2
and from differentiating the $W$ spline from Stage B.

This is a **single linear ODE** with known, precomputed coefficients.
No iteration on $Y_H$ or any other quantity is required.

Accumulate the profit-coverage premium integral as an auxiliary state:

$$P'(\tau) = \frac{\lambda(\tau) I(\tau) + c_I}{1-Q} + \delta_b P,
  \qquad P(t_0) = 0.$$

### Output

For a given cycle $[t_0, T]$:

- $I(\tau)$: the profit-coverage indemnity,
- $OC(\tau)$: the opportunity cost of a loss at time $\tau$,
- $Y_H(T) = v(T - t_0) - P(T) - \Phi(T, t_0) - c_s e^{\delta_b(T - t_0)} - c_h$:
  the harvest income under profit coverage,
- $Y_L(\tau) = I(\tau) - P(\tau) - \Phi(\tau, t_0) - c_s e^{\delta_b(\tau - t_0)} - c_2$:
  the loss payoff under profit coverage.

---

## Stage D: Optimal policy under profit coverage

### Goal

Re-solve the risk-averse model using the profit-coverage payoffs from
Stage C, so that the continuation value $V(t)$ and optimal policy
$(\tau^*, d^*)$ reflect the higher indemnity and the associated premium
costs.

### How Stage C integrates into Stage A's iterative solver

The existing solver (`solve_seasonal_model`) evaluates $Y_H$ and $Y_L(\tau)$
at each node and quadrature point by solving the breakeven indemnity ODE
within each cycle evaluation. Stage D replaces this with the Stage C
three-ODE sequence, which produces the profit-coverage $Y_H$ and $Y_L$.

The key practical question is how often the Stage C ODEs must be solved.
The indemnity depends on the cycle interval $[t_0, T]$, which changes as
the harvest FOC solver evaluates different candidate rotation lengths.
However, the opportunity cost $OC(\tau)$ depends on $T$ primarily through
the terminal condition of the backward ODE, and this sensitivity is weak
for candidates near the current $\tau^*$.

**Approximation: compute $OC$ once per node, reuse across FOC candidates.**

At each stocking-date node $t_0$, run Stage C once using the current best
estimate $T^{\text{ref}} = t_0 + \tau^*(t_0)$. Store $OC(\tau)$ and the
profit-coverage indemnity $I(\tau)$ on $[t_0, T^{\text{ref}}]$. When the
harvest FOC solver evaluates a candidate $T$:

- If $T \leq T^{\text{ref}}$: truncate the stored profiles.
- If $T > T^{\text{ref}}$: extend $I(\tau)$ by integrating the ODE a few
  more steps; extrapolate $OC(\tau)$ or use breakeven $Y_L$ beyond
  $T^{\text{ref}}$.

Since $Y_H(T)$ — the dominant driver of the FOC — is computed exactly for
each candidate $T$ (using the stored premium integral), the rotation
decision remains accurate. The loss integral uses the approximate $OC$,
but this is a correction weighted by $\lambda$, which is typically small.

### Iterative algorithm

The solver follows the same structure as the existing `solve_seasonal_model`,
with the cycle evaluation modified to use profit-coverage payoffs.

**Initialisation.** Use the Stage A solution as the starting point:

- $V^{(0)}(t) = V^0(t)$ (breakeven continuation value),
- $\tau^{*(0)}(t_0) = \tau^{*0}(t_0)$ (breakeven optimal rotation).

**Each iteration $k$:**

1. **Precompute indemnity profiles.** At each of the $2N+1$ stocking-date
   nodes $t_0$, run Stage C using $T^{\text{ref}} = t_0 + \tau^{*(k-1)}(t_0)$:
   - Breakeven ODE forward → $I_0(\tau)$, $Y_H^0$, $Y_L^0(\tau)$
   - Backward ODE → $\bar{W}_0(\tau)$, $OC(\tau)$
   - Profit-coverage ODE forward → $I(\tau)$, $Y_H$, $Y_L(\tau)$

   Store $Y_H(t_0)$ and $Y_L(\tau; t_0)$ as interpolation objects.

2. **Solve harvest FOC at nodes.** For each node $t_0$, find $T^*$ such
   that the FOC

   $$(∂Y_H/∂T) \cdot u'(Y_H(T)) = \delta(V(T) + u(Y_H(T)))
     + \lambda(T)(u(Y_H(T)) - u(Y_L(T))) - V'(T)$$

   is satisfied. Evaluate $Y_H(T)$ exactly for each candidate $T$ using
   the stored premium integral. Evaluate $Y_L(s)$ from the precomputed
   profile. This produces $\tau^{*(k)}(t_0)$.

3. **Compute $\tilde{V}(t_0)$ at nodes.** Evaluate the cycle value

   $$\tilde{V}(t_0) = S(T^*, t_0)\, e^{-\delta(T^* - t_0)}
     \left[u(Y_H(T^*)) + V(T^*)\right]
     + \int_{t_0}^{T^*} S(s, t_0)\, \lambda(s)\, e^{-\delta(s - t_0)}
     \left[u(Y_L(s)) + V(s)\right] ds$$

   using the precomputed $Y_H(t_0)$ and $Y_L(\tau; t_0)$.

4. **Solve stocking FOC and update $V(t)$.** Follow the existing procedure:
   solve the stocking FOC using the $\tilde{V}$ spline, then update $V(t)$
   via the f/g decomposition and linear system solve.

5. **Damped update and convergence check.** Blend:
   $V^{(k)} = \alpha_{\text{damp}} \cdot V_{\text{new}}
   + (1 - \alpha_{\text{damp}}) \cdot V^{(k-1)}$.
   Converge when $\|V^{(k)} - V^{(k-1)}\|_\infty < \varepsilon$.

**Refinement (optional).** If $\tau^*$ has shifted significantly from the
reference used in Step 1, re-run Stage C at the updated rotation lengths
and perform a few more iterations to ensure consistency.

### Continuation homotopy for difficult convergence

If direct iteration at the target $\xi$ fails to converge (e.g., the
profit-coverage premiums are large enough to substantially alter the
rotation decision), use a **$\xi$-continuation** strategy:

1. Choose a sequence of coverage fractions
   $0 = \xi_0 < \xi_1 < \xi_2 < \cdots < \xi_K = \xi_{\text{target}}$.
   A geometric or linear spacing works (e.g., $\xi_j = j \cdot \xi_{\text{target}} / K$
   with $K = 5$).

2. For $j = 0, 1, \ldots, K$:
   - Run Stage D at $\xi_j$, using the converged solution from $\xi_{j-1}$
     as the initial guess for $V(t)$ and $\tau^*(t_0)$.
   - Stages A and B are **not re-run** — the breakeven policy and $W(t)$
     are fixed throughout the homotopy.
   - Only Stage C (three ODE solves per node) and the Stage D iteration
     loop are repeated at each step.

3. The final solution at $\xi_K = \xi_{\text{target}}$ is the converged
   policy under profit coverage.

This works because each step starts close to the solution, ensuring the
iterative solver remains in the basin of convergence. The cost is modest:
Stage C is cheap (three linear ODEs per node), and the Stage D iteration
typically converges in a few steps when initialised near the solution.

---

## Full pipeline summary

```
Stage A:  solve_seasonal_model(p) with ξ = 0
            → V⁰(t), τ*⁰(t₀), d*⁰(t)               [existing code]

Stage B:  Dollar continuation value (breakeven insurance)
            For each node: solve breakeven ODE → Y_H⁰, Y_L⁰
            Iterate W(t) using f/g decomposition      [adapted from Stage A]
            → W(t) periodic spline

Stage C:  Profit-coverage indemnity (three sequential ODEs per cycle)
            Step 1: breakeven ODE forward              → I₀(τ), Y_H⁰, Y_L⁰
            Step 2: W̄₀ backward ODE                   → W̄₀(τ), OC(τ)
            Step 3: profit-coverage ODE forward        → I(τ), Y_H, Y_L
            [no iteration — each step feeds the next]

Stage D:  Re-solve risk-averse model with profit-coverage payoffs
            Initialise from Stage A solution
            Each iteration:
              1. Precompute indemnity profiles at each node (Stage C)
              2. Solve harvest FOC → τ*(t₀)
              3. Compute Ṽ(t₀) with profit-coverage Y_H, Y_L
              4. Solve stocking FOC → d*(t)
              5. Update V(t), check convergence
            → V(t), τ*(t₀), d*(t), I(τ)

            If convergence is difficult, use ξ-continuation:
              ξ₀=0 → ξ₁ → ξ₂ → ... → ξ_target
              re-running Stages C+D at each step
```

---

## Properties

- **Stages A and B are solved once.** The breakeven policy and dollar
  continuation value $W(t)$ are independent of $\xi$ and provide the
  fixed baseline for opportunity cost calculations.

- **No iteration within Stage C.** The opportunity cost is precomputed
  under breakeven insurance, eliminating the circularity between $I$ and
  $OC$. The profit-coverage ODE is linear with known coefficients.

- **Stage C is called once per node per Stage D iteration.** The indemnity
  profile is computed at the current best $\tau^*$ and reused across all
  FOC candidates at that node. The outer iteration corrects for any
  approximation error.

- **$\xi$-continuation for robustness.** When the target $\xi$ is large,
  gradually increasing coverage from the breakeven baseline ensures the
  iterative solver stays near convergence at each step.

- **Consistent baselines.** Both $W(t)$ and $\bar{W}_0(\tau)$ are
  evaluated under breakeven insurance, ensuring the opportunity cost
  measures the value of lost production time against a consistent
  reference point.

---

## Practical notes

- **$OC'(\tau)$ computation.** The derivative $\bar{W}_0'(\tau)$ is known
  analytically from the backward ODE right-hand side. The derivative
  $W'(\tau)$ comes from the Stage B spline: piecewise constant for linear
  splines, or fit a cubic spline to the $W$ nodal values if smoother
  derivatives are needed. An adaptive ODE solver (e.g. `Tsit5()`) handles
  the resulting mild discontinuities without difficulty.

- **Backward ODE integration.** To integrate $\bar{W}_0$ backward from
  $T$, substitute $\sigma = T - \tau$ and define
  $\hat{W}(\sigma) = \bar{W}_0(T - \sigma)$. Then

  $$\hat{W}'(\sigma) = -(\delta + \lambda(T - \sigma))\, \hat{W}(\sigma)
    + \lambda(T - \sigma)\left[Y_L^0(T - \sigma) + W(T - \sigma)\right],$$

  with $\hat{W}(0) = Y_H^0(T) + W(T)$, integrated forward in $\sigma$
  from $0$ to $T - t_0$.

- **Bundling ODE solves.** Steps 1 and 3 of Stage C can share the same
  solver call by augmenting the state vector. Step 2 must be solved
  separately (backward direction) but is a single linear ODE and very
  cheap.

- **Sensitivity analysis.** To compare coverage levels, hold Stages A
  and B fixed and loop over $\xi$ values, re-running Stages C and D.
  If using $\xi$-continuation, the solutions at intermediate $\xi$ values
  come for free.

- **Validation.** Under constant hazard and $\xi = 0$, Stage D should
  reproduce the Stage A solution exactly. Under constant hazard with
  $\xi > 0$, the indemnity ODE has constant coefficients and can be solved
  analytically, providing a benchmark for Stage C. The $OC(\tau)$ profile
  can be checked against direct numerical integration of the $\bar{W}_0$
  definition.