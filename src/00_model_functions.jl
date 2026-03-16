"""
    00_model_functions.jl

Model function definitions for the aquaculture bioeconomic model.

Provides all functional inputs required by the numerical solver described in
README.md §§ 2–10. Functions fall into five groups:

1. **Seasonal primitives** — hazard rate λ(t), natural mortality m(t), and
   von Bertalanffy growth rate k(t). Each is a positive, 365-day periodic
   function parameterised as exp(Fourier series).
2. **Survival and cumulative hazard** — S(t,T) and cumulative hazard integral.
3. **Growth and stock value** — individual fish length L(t), numbers n(t),
   and harvest value v(t, t₀).
4. **Cost and income** — feed cost φ, accumulated cost integrals Φ and Π,
   harvest income Y_H, loss income Y_L.
5. **Insurance** — premium rate π(t) and indemnity ODE for I(t).
6. **Utility** — von Neumann–Morgenstern utility u and its derivative u'.
"""

using QuadGK          # numerical integration
using OrdinaryDiffEq  # ODE solver for indemnity

# ──────────────────────────────────────────────────────────────────────────────
# 1. Seasonal primitive functions (period = 365, strictly positive)
#
#    Each is parameterised as  exp(a₀ + Σₖ [aₖ sin(2πk t/365) + bₖ cos(2πk t/365)])
#    so the output is always positive.  The coefficient vectors are stored in
#    a NamedTuple or struct; the functions below evaluate the series.
# ──────────────────────────────────────────────────────────────────────────────

"""
    positive_periodic(t, coeffs)

Evaluate a strictly-positive 365-day periodic function at calendar day `t`.

# Arguments
- `t`      : calendar time (days)
- `coeffs` : `(a0, a, b)` where `a0` is the intercept, `a = [a₁,…,aₖ]` and
             `b = [b₁,…,bₖ]` are sine and cosine coefficient vectors.

# Returns
`exp(a0 + Σₖ aₖ sin(2πk t/365) + bₖ cos(2πk t/365))`
"""
function positive_periodic(t, coeffs)
    (; a0, a, b) = coeffs
    ω = 2π / 365.0
    s = a0
    for k in eachindex(a)
        s += a[k] * sin(k * ω * t) + b[k] * cos(k * ω * t)
    end
    return exp(s)
end

"""
    λ(t, p)

Instantaneous catastrophic hazard rate (inhomogeneous Poisson intensity).
See README § "Seasonal Mortality Risk".

# Arguments
- `t` : calendar day
- `p` : parameter set (must contain `p.λ_coeffs`)
"""
λ(t, p) = positive_periodic(t, p.λ_coeffs)

"""
    m_rate(t, p)

Instantaneous natural (background) mortality rate on fish numbers.
Drives the numbers ODE  ṅ = −m(t)·n  (README § 10, time-dependent growth).

# Arguments
- `t` : calendar day
- `p` : parameter set (must contain `p.m_coeffs`)
"""
m_rate(t, p) = positive_periodic(t, p.m_coeffs)

"""
    k_growth(t, p)

Von Bertalanffy growth-rate parameter k(t).
Drives the length ODE  L̇ = k(t)·(L∞ − L)  (README § 10).

# Arguments
- `t` : calendar day
- `p` : parameter set (must contain `p.k_coeffs`)
"""
k_growth(t, p) = positive_periodic(t, p.k_coeffs)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cumulative hazard and survival
# ──────────────────────────────────────────────────────────────────────────────

"""
    cumulative_hazard(t, T, p)

Cumulative catastrophic hazard  m(t,T) = ∫_t^T λ(s) ds.
"""
function cumulative_hazard(t, T, p)
    val, _ = quadgk(s -> λ(s, p), t, T)
    return val
end

"""
    S(t, T, p)

Survival probability from time `t` to `T`:  S(t,T) = exp(−m(t,T)).
"""
S(t, T, p) = exp(-cumulative_hazard(t, T, p))

"""
    solve_cumulative_hazard(t₀, T, p)

Solve the cumulative hazard ODE  Λ'(t) = λ(t), Λ(t₀) = 0.
Enables O(1) survival lookups: S(t₀, s) = exp(−Λ(s)) for any s ∈ [t₀, T].
"""
function solve_cumulative_hazard(t₀, T, p)
    dΛdt(Λ, params, t) = λ(t, params)
    prob = ODEProblem(dΛdt, 0.0, (t₀, T), p)
    return solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. Growth and stock value
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_length(t₀, T, L₀, p)

Solve the von Bertalanffy length ODE  L̇ = k(t)·(L∞ − L)  from `t₀` to `T`
with initial length `L₀`.

# Returns
An ODE solution object that can be evaluated at any `t ∈ [t₀, T]`.
"""
function solve_length(t₀, T, L₀, p)
    function dLdt(L, params, t)
        return k_growth(t, params) * (params.L∞ - L)
    end
    prob = ODEProblem(dLdt, L₀, (t₀, T), p)
    sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
    return sol
end

"""
    solve_numbers(t₀, T, n₀, p)

Solve the numbers ODE  ṅ = −m(t)·n  from `t₀` to `T` with initial
count `n₀`.

# Returns
An ODE solution object that can be evaluated at any `t ∈ [t₀, T]`.
"""
function solve_numbers(t₀, T, n₀, p)
    function dndt(n, params, t)
        return -m_rate(t, params) * n
    end
    prob = ODEProblem(dndt, n₀, (t₀, T), p)
    sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
    return sol
end

"""
    W_weight(L, p)

Allometric weight-length relationship:  W(L) = ω · L^β.

# Arguments
- `L` : mean body length (cm)
- `p` : parameter set (must contain `p.ω` and `p.β`)
"""
W_weight(L, p) = p.ω * L^p.β

"""
    f_value(L, p)

Value per individual fish as a function of body length, using a sigmoid
price-per-gram that rises from near zero for small fish to near 1 for
large fish:

  W = ω · L^β                             (allometric weight)
  σ = 1 / (1 + exp(−(W − W₅₀) / s))      (sigmoid price factor)
  f = W · σ

The two parameters `W₅₀` (midpoint weight) and `s` (scale) control where
and how steeply the per-gram price increases.

# Arguments
- `L` : mean body length (cm)
- `p` : parameter set containing:
  - `ω`   : weight-length scalar
  - `β`   : weight-length exponent (typically ≈ 3)
  - `W₅₀` : sigmoid midpoint weight (g) — price is 0.5 per g at this weight
  - `s`   : sigmoid scale parameter (g) — controls steepness
"""
function f_value(L, p)
    W = W_weight(L, p)
    σ = 1.0 / (1.0 + exp(-(W - p.W₅₀) / p.s))
    return W * σ
end

"""
    v(t, t₀, L_sol, n_sol, p)

Gross stock value at calendar time `t` for a cycle stocked at `t₀`:
  v(t, t₀) = n(t) · f(L(t))
where `L_sol` and `n_sol` are precomputed ODE solutions from `solve_length`
and `solve_numbers`.

See README § 10 (time-dependent growth).
"""
function v(t, t₀, L_sol, n_sol, p)
    L_t = L_sol(t)
    n_t = n_sol(t)
    return n_t * f_value(L_t, p)
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cost and income functions
# ──────────────────────────────────────────────────────────────────────────────

"""
    φ(t, t₀, L_sol, n_sol, p)

Instantaneous feed cost rate:  φ(t) = η · n(t) · W(L(t)).
Feed costs are proportional to total biomass (numbers × weight per fish).
See README § "Cost Structure".
"""
φ(t, t₀, L_sol, n_sol, p) = p.η * n_sol(t) * W_weight(L_sol(t), p)

"""
    Φ_accumulated(T, t₀, L_sol, n_sol, p)

Accumulated feed costs compounded to time `T` at borrowing rate δ_b:
  Φ(T, t₀) = ∫_{t₀}^{T} φ(s) · exp(δ_b · (T − s)) ds
"""
function Φ_accumulated(T, t₀, L_sol, n_sol, p)
    integrand(s) = φ(s, t₀, L_sol, n_sol, p) * exp(p.δ_b * (T - s))
    val, _ = quadgk(integrand, t₀, T)
    return val
end

"""
    solve_accumulated_feed(t₀, T, L_sol, n_sol, p)

Solve the accumulated feed cost ODE  Φ'(t) = φ(t) + δ_b·Φ(t), Φ(t₀) = 0,
where φ(t) = η·n(t)·W(L(t)). The solution Φ_sol(s) equals the quadgk-based
`Φ_accumulated(s, t₀, ...)` but is O(1) to evaluate at any s ∈ [t₀, T].
"""
function solve_accumulated_feed(t₀, T, L_sol, n_sol, p)
    function dΦdt(Φ, params, t)
        return params.η * n_sol(t) * W_weight(L_sol(t), params) + params.δ_b * Φ
    end
    prob = ODEProblem(dΦdt, 0.0, (t₀, T), p)
    return solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
end

"""
    Π_accumulated(T, t₀, I_sol, p)

Accumulated insurance premiums compounded to time `T` at borrowing rate δ_b:
  Π(T, t₀) = ∫_{t₀}^{T} π(s) · exp(δ_b · (T − s)) ds

`I_sol` is the precomputed indemnity ODE solution.
"""
function Π_accumulated(T, t₀, I_sol, p)
    integrand(s) = π_premium(s, I_sol, p) * exp(p.δ_b * (T - s))
    val, _ = quadgk(integrand, t₀, T)
    return val
end

"""
    solve_accumulated_premium(t₀, T, I_sol, p)

Solve the accumulated premium ODE  Π'(t) = π(t) + δ_b·Π(t), Π(t₀) = 0.
The solution Π_sol(s) equals the quadgk-based `Π_accumulated(s, t₀, ...)`
but is O(1) to evaluate at any s ∈ [t₀, T].
"""
function solve_accumulated_premium(t₀, T, I_sol, p)
    function dΠdt(Π, params, t)
        π_t = (λ(t, params) * I_sol(t) + params.c_I) / (1 - params.Q)
        return π_t + params.δ_b * Π
    end
    prob = ODEProblem(dΠdt, 0.0, (t₀, T), p)
    return solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
end

"""
    Y_H(T, t₀, L_sol, n_sol, I_sol, p)

Harvest income at planned harvest date `T`:
  Y_H = v(T, t₀) − c_s·exp(δ_b·(T−t₀)) − Φ(T,t₀) − Π(T,t₀) − c_h

See README § 7 ("Income and Utility").
"""
function Y_H(T, t₀, L_sol, n_sol, I_sol, p)
    v_T = v(T, t₀, L_sol, n_sol, p)
    stocking_compounded = p.c_s * exp(p.δ_b * (T - t₀))
    feed = Φ_accumulated(T, t₀, L_sol, n_sol, p)
    premium = Π_accumulated(T, t₀, I_sol, p)
    return v_T - stocking_compounded - feed - premium - p.c_h
end

"""
    Y_L(τ, t₀, I_sol, L_sol, n_sol, p)

Loss income at catastrophic event time `τ`:
  Y_L = I(τ) − c_s·exp(δ_b·(τ−t₀)) − Φ(τ,t₀) − Π(τ,t₀) − c₂

See README § 7 ("Income and Utility").
"""
function Y_L(τ, t₀, I_sol, L_sol, n_sol, p)
    I_τ = I_sol(τ)
    stocking_compounded = p.c_s * exp(p.δ_b * (τ - t₀))
    feed = Φ_accumulated(τ, t₀, L_sol, n_sol, p)
    premium = Π_accumulated(τ, t₀, I_sol, p)
    return I_τ - stocking_compounded - feed - premium - p.c₂
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. Insurance — premium rate and indemnity ODE
# ──────────────────────────────────────────────────────────────────────────────

"""
    π_premium(t, I_sol, p)

Instantaneous insurance premium rate:
  π(t) = (λ(t)·I(t) + c_I) / (1 − Q)

See README § 8.
"""
function π_premium(t, I_sol, p)
    return (λ(t, p) * I_sol(t) + p.c_I) / (1 - p.Q)
end

"""
    solve_indemnity(t₀, T, L_sol, n_sol, p)

Solve the indemnity ODE from `t₀` to `T`:
  I'(t) = (λ(t)/(1−Q) + δ_b)·I(t) + φ(t) + c_I/(1−Q) − δ_b·(c₂ + Y_MIN)
  I(t₀) = Y_MIN + c_s + c₂

See README § 8–9 ("Insurance Premiums and Coverage").

# Returns
An ODE solution object that can be evaluated at any `t ∈ [t₀, T]`.
"""
function solve_indemnity(t₀, T, L_sol, n_sol, p)
    I₀ = p.Y_MIN + p.c_s + p.c₂

    function dIdt(I, params, t)
        λ_t = λ(t, params)
        φ_t = φ(t, t₀, L_sol, n_sol, params)
        return (λ_t / (1 - params.Q) + params.δ_b) * I +
               φ_t +
               params.c_I / (1 - params.Q) -
               params.δ_b * (params.c₂ + params.Y_MIN)
    end

    prob = ODEProblem(dIdt, I₀, (t₀, T), p)
    sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
    return sol
end


# ──────────────────────────────────────────────────────────────────────────────
# 6. Utility
# ──────────────────────────────────────────────────────────────────────────────

"""
    u(Y, p)

CRRA utility:  u(Y) = Y^(1−γ)/(1−γ)  for γ ≠ 1,  log(Y) for γ = 1.

# Arguments
- `Y` : income (must be positive)
- `p` : parameter set (must contain `p.γ`, the coefficient of relative risk aversion)
"""
function u(Y, p)
    γ = p.γ
    if γ ≈ 1.0
        return log(Y)
    else
        return Y^(1 - γ) / (1 - γ)
    end
end

"""
    u_prime(Y, p)

Marginal CRRA utility:  u'(Y) = Y^(−γ).

# Arguments
- `Y` : income (must be positive)
- `p` : parameter set (must contain `p.γ`)
"""
function u_prime(Y, p)
    return Y^(-p.γ)
end


# ──────────────────────────────────────────────────────────────────────────────
# 7. Fourier series helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    fourier_eval(t, coeffs)

Evaluate a general (possibly negative) 365-day periodic Fourier series:
  f(t) = a₀ + Σₖ [aₖ sin(2πk t/365) + bₖ cos(2πk t/365)]

Used to represent V(t), T*(t₀), and t₀*(T).

# Arguments
- `t`      : calendar day
- `coeffs` : `(a0, a, b)` — intercept plus sine/cosine coefficient vectors
"""
function fourier_eval(t, coeffs)
    (; a0, a, b) = coeffs
    ω = 2π / 365.0
    s = a0
    for k in eachindex(a)
        s += a[k] * sin(k * ω * t) + b[k] * cos(k * ω * t)
    end
    return s
end

"""
    fourier_derivative(t, coeffs)

Evaluate the time derivative of a 365-day periodic Fourier series:
  f'(t) = Σₖ k·(2π/365)·[aₖ cos(2πk t/365) − bₖ sin(2πk t/365)]

Used for V'(t) in the harvest FOC and Ṽ'(t₀) in the stocking FOC.
"""
function fourier_derivative(t, coeffs)
    (; a0, a, b) = coeffs
    ω = 2π / 365.0
    s = 0.0
    for k in eachindex(a)
        s += k * ω * (a[k] * cos(k * ω * t) - b[k] * sin(k * ω * t))
    end
    return s
end


# ──────────────────────────────────────────────────────────────────────────────
# 8. Periodic linear spline helpers
#
#    Used to represent V(t), Ṽ(t₀), and τ*(t₀) as piecewise-linear periodic
#    functions on [0, 365). Unlike Fourier series, linear splines can represent
#    discontinuities and sharp features without Gibbs ringing.
#
#    Each spline is stored as a NamedTuple (nodes, values) where `nodes` are
#    sorted positions in [0, 365) and `values` are the function values there.
# ──────────────────────────────────────────────────────────────────────────────

"""
    make_spline(nodes, values)

Create a periodic linear spline NamedTuple from nodes and values.
"""
make_spline(nodes, values) = (nodes = collect(Float64, nodes), values = collect(Float64, values))

"""
    spline_eval(t, coeffs)

Evaluate a periodic linear spline on [0, 365) at calendar day `t`.

# Arguments
- `t`      : calendar time (days)
- `coeffs` : `(nodes, values)` where `nodes` are sorted positions in [0, 365)
              and `values` are the function values at those nodes.
"""
function spline_eval(t, coeffs)
    nodes = coeffs.nodes
    vals = coeffs.values
    t_mod = mod(t, 365.0)
    n = length(nodes)

    idx = searchsortedlast(nodes, t_mod)

    if idx == 0 || idx == n
        t_lo = nodes[end]
        t_hi = nodes[1] + 365.0
        v_lo = vals[end]
        v_hi = vals[1]
        t_eff = idx == 0 ? t_mod + 365.0 : t_mod
    else
        t_lo = nodes[idx]
        t_hi = nodes[idx + 1]
        v_lo = vals[idx]
        v_hi = vals[idx + 1]
        t_eff = t_mod
    end

    frac = (t_eff - t_lo) / (t_hi - t_lo)
    return v_lo + frac * (v_hi - v_lo)
end

"""
    spline_derivative(t, coeffs)

Evaluate the derivative (piecewise constant slope) of a periodic linear spline.
At node boundaries, returns the slope of the interval containing `t`.

# Arguments
- `t`      : calendar time (days)
- `coeffs` : `(nodes, values)` periodic spline
"""
function spline_derivative(t, coeffs)
    nodes = coeffs.nodes
    vals = coeffs.values
    t_mod = mod(t, 365.0)
    n = length(nodes)

    idx = searchsortedlast(nodes, t_mod)

    if idx == 0 || idx == n
        t_lo = nodes[end]
        t_hi = nodes[1] + 365.0
        v_lo = vals[end]
        v_hi = vals[1]
    else
        t_lo = nodes[idx]
        t_hi = nodes[idx + 1]
        v_lo = vals[idx]
        v_hi = vals[idx + 1]
    end

    return (v_hi - v_lo) / (t_hi - t_lo)
end

"""
    spline_interp_weights(t, nodes)

Return interpolation indices and weight for a periodic linear spline evaluation.

# Returns
`(idx_lo, idx_hi, weight)` such that:
    spline(t) = (1 - weight) * values[idx_lo] + weight * values[idx_hi]
"""
function spline_interp_weights(t, nodes)
    t_mod = mod(t, 365.0)
    n = length(nodes)

    idx = searchsortedlast(nodes, t_mod)

    if idx == 0 || idx == n
        t_lo = nodes[end]
        t_hi = nodes[1] + 365.0
        idx_lo = n
        idx_hi = 1
        t_eff = idx == 0 ? t_mod + 365.0 : t_mod
    else
        t_lo = nodes[idx]
        t_hi = nodes[idx + 1]
        idx_lo = idx
        idx_hi = idx + 1
        t_eff = t_mod
    end

    weight = (t_eff - t_lo) / (t_hi - t_lo)
    return (idx_lo = idx_lo, idx_hi = idx_hi, weight = weight)
end
