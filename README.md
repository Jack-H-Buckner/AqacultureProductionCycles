# Optimal Aquaculture Production Cycles Under Seasonal Mortality Risk

## Overview

This project implements a numerical solver for an optimal rotation model in aquaculture that extends the classical Reed (1984) forestry framework to incorporate **seasonal (inhomogeneous) mortality risk**, **continuous feed costs**, **risk aversion**, and **indemnity insurance**. The model determines when an aquaculture operator should stock and harvest fish to maximize expected utility over an infinite sequence of production cycles, given that catastrophic loss events (disease, storms, algal blooms) follow a time-varying seasonal pattern.

## Notation and Definitions

The model is built on the following primitives, defined in Sections 2–5 of the writeup.

### Timing

Each production cycle is defined by two calendar dates: the **stocking date** t₀ (when fish are placed in the facility) and the **planned harvest date** T. The rotation length is T − t₀. After a cycle ends the operator may leave the site fallow before restocking.

### Growth and Harvest Value

The gross value of the standing stock, conditional on survival, is v(T − t₀) — a function of age alone in the base model (Section 2.2). We assume v(0) = 0, v' > 0, and eventual concavity as growth slows toward biological maturity. In the time-dependent growth extension (Section 10) the value becomes v(t, t₀) = n(t)·f(L(t)), where n and L evolve according to seasonal mortality and von Bertalanffy dynamics respectively.

### Cost Structure

| Symbol | Name | When paid | Description |
|--------|------|-----------|-------------|
| c_s | Stocking cost | At t₀ | Fingerlings / seed stock |
| c_h | Harvest cost | At T | Labor, equipment, transport, processing |
| c₂ | Clearing cost | At loss event | Site preparation after catastrophic mortality |
| φ(v) = η·v | Feed cost rate | Continuously while stock is alive | Instantaneous cost proportional to stock value (η is the feed cost rate) |

### Seasonal Mortality Risk

Catastrophic mortality events (disease, harmful algal blooms, storms, temperature extremes) arrive as an **inhomogeneous Poisson process** with instantaneous hazard rate λ(t), a periodic function with period 365 days. The key derived quantities are:

**Cumulative hazard** from calendar time t to T:
```
m(t, T) = ∫[t to T] λ(s) ds
```

**Survival function** — the probability the stock survives from t to T without a catastrophic event:
```
S(t, T) = exp(−m(t, T)) = exp(−∫[t to T] λ(s) ds)
```

Under the base assumption of total destruction (θ = 0), a loss event wipes out the entire stock with no salvage value.

### Accumulated Cost Terms

Because the operator borrows against end-of-cycle revenue at rate δ_b, flow costs are compounded forward to the end of the cycle. The two accumulated cost integrals that appear throughout the model are:

**Accumulated insurance premiums** (compounded to time T):
```
Π(T, t₀) = ∫[t₀ to T] π(s)·e^{δ_b·(T − s)} ds
```

**Accumulated feed costs** (compounded to time T):
```
Φ(T, t₀) = ∫[t₀ to T] φ(s)·e^{δ_b·(T − s)} ds
```

where φ(s) = η·v(s − t₀) is the instantaneous feed cost at time s and π(s) is the instantaneous insurance premium. Each integral compounds past costs forward at the borrowing rate so they can be subtracted from end-of-cycle revenue.

### Discounting and Continuation Value

Future cash flows are discounted at a constant rate δ > 0. The continuation value V(T) is the expected present value of all future cycles, evaluated at the moment the current cycle ends. Because mortality risk is seasonal, V depends on the calendar date (t mod 365) — the central departure from the time-homogeneous Reed (1984) model.

## Model Description

The core model (Sections 7–10 of the writeup) builds on an earlier risk-neutral formulation by introducing three key extensions: a von Neumann–Morgenstern utility function to capture risk aversion, an insurance market that offers indemnity contracts, and time-dependent biological growth.

### Income and Utility (Section 7)

The operator finances each production cycle by borrowing against end-of-cycle revenue. All costs — stocking, feed, and insurance premiums — accrue interest at a borrowing rate δ_b until the cycle concludes. Terminal income therefore depends on whether the cycle ends with a successful harvest or a catastrophic loss:

- **Harvest income** Y_H(T): gross harvest value v(T − t₀) minus the accumulated (with interest) stocking cost, feed costs Φ(T, t₀), insurance premiums Π(T, t₀), and harvest cost c_h.
- **Loss income** Y_L(τ): the indemnity payment I(τ) minus accumulated stocking cost, feed costs, insurance premiums, and clearing cost c₂, evaluated at the time of the loss event τ.

The expected present utility of a production cycle combines:
1. The utility from a successful harvest, weighted by the survival probability S(T, t₀) and discounted at rate δ.
2. An integral over all possible loss times, weighting the utility from insurance payouts plus the continuation value V(s) by the hazard rate λ(s) and survival probability.

### Optimal Harvest Condition (Section 7, FOC)

The optimal harvest date T* satisfies a first-order condition that balances the marginal utility gain from delaying harvest against three opportunity costs:

```
(∂Y_H/∂T) · u'(Y_H(T)) = δ(V(T) + u(Y_H(T)))
                         + λ(T)(u(Y_H(T)) − u(Y_L(T)))
                         − V'(T)
```

The left side is the marginal utility of additional growth (net of the marginal cost of feed, insurance, and interest). The right side includes: (i) the discount-rate cost of tying up site value and harvest utility, (ii) a risk-aversion term that weights the utility gap between harvest and loss outcomes by the instantaneous hazard rate, and (iii) a calendar-time effect from shifting future cycles.

### Insurance Premiums and Coverage (Sections 8–9)

Insurance premiums are priced using the pure premium method. The instantaneous premium rate is:

```
π(t) = (λ(t)·I(t) + c_I) / (1 − Q)
```

where c_I captures administrative costs and Q is the insurer's profit margin.

The baseline coverage rule sets the indemnity so the firm breaks even in the event of a loss. This leads to a first-order linear ODE for the indemnity level I(t):

```
I'(t) = (λ(t)/(1−Q) + δ_b)·I(t) + φ(t) + c_I/(1−Q) − δ_b·c₂
I(t₀) = c_s + c₂
```

An optional minimum-income margin Y_MIN can be added to the clearing cost to guarantee the firm a positive return even after a loss.

### Optimal Stocking Condition (Section 10)

The optimal stocking date t₀* after a cycle ends at time t satisfies:

```
Ṽ'(t₀) = δ · Ṽ(t₀)
```

where Ṽ(t₀) = J(T*, t₀, t₀) is the expected present utility of the cycle evaluated at stocking. The operator delays restocking as long as the rate of increase in cycle value exceeds the opportunity cost of waiting.

### Time-Dependent Growth (Section 10, second part)

Stock value v(t) = n(t)·f(L(t)) is driven by seasonal mortality on numbers (ṅ = −m(t)·n) and von Bertalanffy growth on mean length (L̇ = k(t)·(L∞ − L)). This makes terminal income depend on absolute calendar dates, not just cycle duration, enriching the time dependence of the continuation values.

## Numerical Procedure (Section 11)

The model's coupled system of equations — harvest FOC, stocking FOC, value linkage, and the value function — cannot be solved in closed form because the continuation value V(t) is a periodic function of calendar time. The numerical solution proceeds as follows.

### Representation

All periodic unknowns are approximated as truncated Fourier (harmonic) series with period 365 days:

| Function | Description |
|----------|-------------|
| V(t) | Continuation value of the facility at end-of-cycle time t |
| T*(t₀) | Optimal harvest date given stocking date t₀ |
| t₀*(T) | Optimal stocking date given end-of-cycle time T |

Each is represented as a linear combination of sine and cosine terms with periods ≤ 365 days. The Fourier coefficients are the unknowns to be determined.

### Iterative Algorithm

1. **Initialize** V(t) (e.g., constant or informed guess from the risk-neutral solution).
2. **Solve for optimal controls**: Given the current V(t), evaluate the harvest FOC and stocking FOC at the nodes of the harmonic series to obtain T*(t₀) and t₀*(T).
3. **Update continuation values**: Using the optimal controls, compute Ṽ(t₀) from the full objective function J(T*, t₀, t₀) at each node, then update V(t) via the value linkage V(t) = e^{−δ(t₀* − t)} · Ṽ(t₀*).
4. **Iterate** steps 2–3 until the Fourier coefficients converge.

### Key Equations Used at Each Iteration

**Harvest FOC** (to find T* for each t₀):
```
(∂Y_H/∂T)·u'(Y_H) = δ(V(T) + u(Y_H)) + λ(T)(u(Y_H) − u(Y_L)) − V'(T)
```

**Stocking FOC** (to find t₀* for each t):
```
Ṽ'(t₀) = δ · Ṽ(t₀)
```

**Value function** (to compute Ṽ):
```
Ṽ(t₀) = S(T*, t₀)·e^{−δ(T*−t₀)}·(Y_H(T*) + V(T*))
       + ∫[t₀ to T*] S(s, t₀)·λ(s)·e^{−δ(s−t₀)}·(u(Y_L(s)) + V(s)) ds
```

**Value linkage** (to update V from Ṽ):
```
V(t) = S(T*, t₀*)·e^{−δ(T*−t)}·(Y_H(T*) + V(T*))
     + ∫[t₀* to T*] S(s, t₀*)·λ(s)·e^{−δ(s−t)} ·(u(Y_L(s)) + V(s)) ds
```

**Insurance ODE** (solved within each cycle evaluation):
```
I'(t) = (λ(t)/(1−Q) + δ_b)·I(t) + φ(t) + c_I/(1−Q) − δ_b·(c₂ + Y_MIN)
I(t₀) = Y_MIN + c_s + c₂
```

## References

- Reed, W. J. (1984). The effects of the risk of fire on the optimal rotation of a forest. *Journal of Environmental Economics and Management*, 11, 180–190.
- Bjørndal, T. (1988). Optimal harvesting of farmed fish. *Marine Resource Economics*, 5, 139–159.
- Loisel, P., Brunette, M., & Couture, S. (2020). Insurance and Forest Rotation Decisions Under Storm Risk. *Environmental and Resource Economics*, 76, 347–367.