"""
    01_homogeneous_case.jl

Analytical and semi-analytical solutions for the homogeneous (constant hazard)
aquaculture model, following the validation benchmarks in README.md
§ "Validation Against the Reed (1984) Analytical Solution".

Provides four nested validation cases with increasing complexity:
1. Classical Reed — risk-neutral, no feed/insurance costs
2. Risk aversion — CRRA utility, constant loss income Y_L = Y_MIN
3. Feed costs — η > 0, compounded feed cost integral
4. Insurance — breakeven coverage, constant-coefficient indemnity ODE

All cases use constant parameters (λ, k, m, δ, δ_b) so that seasonal
Fourier structure collapses to scalars. Growth follows von Bertalanffy
dynamics with allometric value, yielding closed-form stock value v(τ).
"""

using Roots
using QuadGK

include("00_model_functions.jl")

# ──────────────────────────────────────────────────────────────────────────────
# Bracket-finding utility
# ──────────────────────────────────────────────────────────────────────────────

"""
    find_positive_bracket(income_fn, p; T_min=1.0, T_max=3000.0, n_pts=500)

Find a bracket `(T_lo, T_hi)` where `income_fn > 0` at both endpoints,
suitable for root-finding on the FOC with CRRA utility.

Scans for the interval where income is positive and returns the first and
last T values with positive income.
"""
function find_positive_bracket(income_fn, p; T_min=1.0, T_max=3000.0, n_pts=500)
    Ts = range(T_min, T_max, length=n_pts)
    positive_Ts = [T for T in Ts if income_fn(T, p) > 0]
    isempty(positive_Ts) && error("Harvest income never becomes positive in [$T_min, $T_max]")
    return (first(positive_Ts), last(positive_Ts))
end

"""
    find_foc_bracket(foc_fn, p; T_min=1.0, T_max=3000.0, n_pts=2000)

Find the bracket around the last positive→negative sign change of a FOC
residual function. This selects the economically meaningful root (where
marginal growth no longer justifies delay) rather than an early transient
crossing.

Returns `(T_lo, T_hi)` where `foc_fn(T_lo) > 0` and `foc_fn(T_hi) < 0`.
"""
function find_foc_bracket(foc_fn, p; T_min=1.0, T_max=3000.0, n_pts=2000)
    Ts = range(T_min, T_max, length=n_pts)
    last_bracket = nothing
    prev_val = foc_fn(first(Ts), p)
    for i in 2:length(Ts)
        curr_val = foc_fn(Ts[i], p)
        if prev_val > 0 && curr_val < 0
            last_bracket = (Ts[i-1], Ts[i])
        end
        prev_val = curr_val
    end
    isnothing(last_bracket) && error("No positive→negative FOC crossing found in [$T_min, $T_max]")
    return last_bracket
end

# ──────────────────────────────────────────────────────────────────────────────
# Closed-form growth under constant parameters
# ──────────────────────────────────────────────────────────────────────────────

"""
    v_homogeneous(τ, p)

Stock value at age τ (= T − t₀) under constant growth rate k and constant
natural mortality m:
  L(τ) = L∞ − (L∞ − L₀)·exp(−k·τ)
  n(τ) = n₀·exp(−m·τ)
  v(τ) = n(τ) · f_value(L(τ), p)

where f_value includes weight-based pricing with a threshold premium.

# Arguments
- `τ` : age (days since stocking)
- `p` : parameter set (must contain `p.k_const`, `p.m_const`, `p.L∞`, `p.L₀`,
        `p.n₀`, and value function parameters)
"""
function v_homogeneous(τ, p)
    L = p.L∞ - (p.L∞ - p.L₀) * exp(-p.k_const * τ)
    n = p.n₀ * exp(-p.m_const * τ)
    return n * f_value(L, p)
end

"""
    w_homogeneous(τ, p)

Total stock biomass at age τ under constant parameters:
  w(τ) = n(τ) · W(L(τ))

Used for feed cost calculations (feed costs are proportional to biomass,
not market value).

# Arguments
- `τ` : age (days since stocking)
- `p` : parameter set
"""
function w_homogeneous(τ, p)
    L = p.L∞ - (p.L∞ - p.L₀) * exp(-p.k_const * τ)
    n = p.n₀ * exp(-p.m_const * τ)
    return n * W_weight(L, p)
end

"""
    v_homogeneous_prime(τ, p)

Time derivative of stock value dv/dτ under constant parameters, computed
via the product rule on v = n · f(L):

  dv/dτ = dn/dτ · f(L)  +  n · f'(L) · dL/dτ
        = −m · n · f(L)  +  n · f'(L) · k · (L∞ − L)

where f'(L) = df/dL is computed via the chain rule through W(L):
  f = W · σ(W)    where σ = 1/(1+exp(−(W−W₅₀)/s))
  df/dW = σ + W · σ · (1−σ) / s
  dW/dL = ω · β · L^(β−1)
"""
function v_homogeneous_prime(τ, p)
    L = p.L∞ - (p.L∞ - p.L₀) * exp(-p.k_const * τ)
    n = p.n₀ * exp(-p.m_const * τ)
    dLdτ = p.k_const * (p.L∞ - L)

    W = W_weight(L, p)
    dWdL = p.ω * p.β * L^(p.β - 1)

    # df/dL via chain rule: f = W·σ, df/dW = σ + W·σ(1−σ)/s
    σ = 1.0 / (1.0 + exp(-(W - p.W₅₀) / p.s))
    dfdW = σ + W * σ * (1 - σ) / p.s
    dfdL = dfdW * dWdL

    f_L = f_value(L, p)
    return n * (-p.m_const * f_L + dfdL * dLdτ)
end


# ──────────────────────────────────────────────────────────────────────────────
# Case 1: Classical Reed (1984)
#   Risk-neutral (u(x) = x), no stocking cost, no feed costs, no insurance
#   Y_H(T) = v(T) − c_h
#   FOC:  v'(T) = (δ + λ) · v(T) / (1 − e^{−(λ+δ)T})
# ──────────────────────────────────────────────────────────────────────────────

"""
    reed_foc(T, p)

Residual of the classical Reed (1984) FOC for optimal rotation:
  v'(T) − (δ + λ)·v(T) / (1 − exp(−(λ+δ)·T))

Returns zero at the optimal rotation length T*.

# Arguments
- `T` : rotation length (days)
- `p` : parameter set (must contain `p.λ_const`, `p.δ`)
"""
function reed_foc(T, p)
    δλ = p.δ + p.λ_const
    return v_homogeneous_prime(T, p) - δλ * v_homogeneous(T, p) / (1 - exp(-δλ * T))
end

"""
    solve_reed(p; T_bracket=nothing)

Solve for the optimal rotation T* under the classical Reed model using
a root finder on the FOC. If no bracket is given, automatically finds
the economically meaningful (last positive→negative) crossing.

# Returns
`T*` — optimal rotation length in days.
"""
function solve_reed(p; T_bracket=nothing)
    if isnothing(T_bracket)
        T_bracket = find_foc_bracket(reed_foc, p)
    end
    f(T) = reed_foc(T, p)
    return find_zero(f, T_bracket, Bisection())
end

"""
    reed_value(T, p)

Analytical continuation value V under classical Reed with constant hazard:
  V = (δ + λ) · e^{−(δ+λ)T} · (v(T) − c_h) / (δ · (1 − e^{−(δ+λ)T}))

Derived from the general formula V = E[e^{−δX}·Y] / (1 − E[e^{−δX}])
with risk-neutral utility, no stocking/feed/insurance costs.
"""
function reed_value(T, p)
    δλ = p.δ + p.λ_const
    S_T = exp(-δλ * T)
    Y_H = v_homogeneous(T, p) - p.c_h
    return (δλ * S_T * Y_H) / (p.δ * (1 - S_T))
end


# ──────────────────────────────────────────────────────────────────────────────
# Case 2: Risk aversion with constant loss income
#   CRRA utility, constant λ, Y_L(s) = Y_MIN (breakeven insurance)
#   Y_H(T) = v(T) − c_s·exp(δ_b·T) − c_h
#   FOC:  Y_H'(T)·u'(Y_H) = (δ+λ)·u(Y_H) / (1 − e^{−(δ+λ)T})
# ──────────────────────────────────────────────────────────────────────────────

"""
    Y_H_homogeneous(T, p)

Harvest income under constant hazard with stocking cost (no feed, no insurance):
  Y_H = v(T) − c_s·exp(δ_b·T) − c_h
"""
function Y_H_homogeneous(T, p)
    return v_homogeneous(T, p) - p.c_s * exp(p.δ_b * T) - p.c_h
end

"""
    Y_H_homogeneous_prime(T, p)

Derivative of harvest income:
  Y_H' = v'(T) − c_s·δ_b·exp(δ_b·T)
"""
function Y_H_homogeneous_prime(T, p)
    return v_homogeneous_prime(T, p) - p.c_s * p.δ_b * exp(p.δ_b * T)
end

"""
    risk_aversion_foc(T, p)

Residual of the harvest FOC with CRRA utility and constant loss income:
  Y_H'(T)·u'(Y_H) − (δ+λ)·u(Y_H) / (1 − e^{−(δ+λ)T})

Returns +Inf when Y_H ≤ 0 (not yet profitable, should keep growing).
See README § "Intermediate Benchmarks", case 1.
"""
function risk_aversion_foc(T, p)
    δλ = p.δ + p.λ_const
    yh = Y_H_homogeneous(T, p)
    yh ≤ 0 && return Inf
    yh_prime = Y_H_homogeneous_prime(T, p)
    return yh_prime * u_prime(yh, p) - δλ * u(yh, p) / (1 - exp(-δλ * T))
end

"""
    solve_risk_aversion(p; T_bracket=nothing)

Solve for T* under CRRA utility with constant hazard and constant loss income.
If no bracket is given, automatically finds the last positive→negative FOC crossing.
"""
function solve_risk_aversion(p; T_bracket=nothing)
    if isnothing(T_bracket)
        T_bracket = find_foc_bracket(risk_aversion_foc, p)
    end
    f(T) = risk_aversion_foc(T, p)
    return find_zero(f, T_bracket, Bisection())
end

"""
    risk_aversion_value(T, p)

Analytical continuation value V with CRRA utility, constant hazard,
and constant loss income Y_MIN:

  V = (δ+λ) · [∫₀ᵀ λ·e^{−(δ+λ)s}·u(Y_MIN) ds + e^{−(δ+λ)T}·u(Y_H(T))]
      / (δ · (1 − e^{−(δ+λ)T}))
"""
function risk_aversion_value(T, p)
    δλ = p.δ + p.λ_const
    S_T = exp(-δλ * T)

    # Loss branch integral: ∫₀ᵀ λ·e^{−(δ+λ)s}·u(Y_MIN) ds
    u_loss = u(p.Y_MIN, p)
    loss_integral = p.λ_const * u_loss * (1 - S_T) / δλ

    # Harvest branch
    harvest_term = S_T * u(Y_H_homogeneous(T, p), p)

    # Full objective
    numerator = δλ * (loss_integral + harvest_term)
    denominator = p.δ * (1 - S_T)
    return numerator / denominator
end


# ──────────────────────────────────────────────────────────────────────────────
# Case 3: Add feed costs (η > 0, constant hazard)
#   Y_H(T) = v(T) − c_s·exp(δ_b·T) − Φ(T) − c_h
#   Φ(T) = ∫₀ᵀ η·w(s)·exp(δ_b·(T−s)) ds  where w = n·W (total biomass)
# ──────────────────────────────────────────────────────────────────────────────

"""
    Φ_homogeneous(T, p)

Accumulated feed costs compounded to harvest time T under constant parameters:
  Φ(T) = ∫₀ᵀ η·w(s)·exp(δ_b·(T−s)) ds

where w(s) = n(s)·W(L(s)) is total biomass.
"""
function Φ_homogeneous(T, p)
    integrand(s) = p.η * w_homogeneous(s, p) * exp(p.δ_b * (T - s))
    val, _ = quadgk(integrand, 0.0, T)
    return val
end

"""
    Φ_homogeneous_prime(T, p)

Derivative of accumulated feed costs with respect to T:
  dΦ/dT = η·w(T) + δ_b·Φ(T)

(By Leibniz rule: the boundary term η·w(T)·exp(0) plus the integral of
δ_b times the integrand.)
"""
function Φ_homogeneous_prime(T, p)
    return p.η * w_homogeneous(T, p) + p.δ_b * Φ_homogeneous(T, p)
end

"""
    Y_H_feed(T, p)

Harvest income with feed costs:
  Y_H = v(T) − c_s·exp(δ_b·T) − Φ(T) − c_h
"""
function Y_H_feed(T, p)
    return v_homogeneous(T, p) - p.c_s * exp(p.δ_b * T) - Φ_homogeneous(T, p) - p.c_h
end

"""
    Y_H_feed_prime(T, p)

Derivative of harvest income with feed costs:
  Y_H' = v'(T) − c_s·δ_b·exp(δ_b·T) − Φ'(T)
"""
function Y_H_feed_prime(T, p)
    return v_homogeneous_prime(T, p) - p.c_s * p.δ_b * exp(p.δ_b * T) - Φ_homogeneous_prime(T, p)
end

"""
    feed_cost_foc(T, p)

Residual of the harvest FOC with CRRA utility, feed costs, and constant hazard:
  Y_H'(T)·u'(Y_H) − (δ+λ)·u(Y_H) / (1 − e^{−(δ+λ)T})

Returns +Inf when Y_H ≤ 0.
"""
function feed_cost_foc(T, p)
    δλ = p.δ + p.λ_const
    yh = Y_H_feed(T, p)
    yh ≤ 0 && return Inf
    yh_prime = Y_H_feed_prime(T, p)
    return yh_prime * u_prime(yh, p) - δλ * u(yh, p) / (1 - exp(-δλ * T))
end

"""
    solve_feed_cost(p; T_bracket=nothing)

Solve for T* with CRRA utility, feed costs, and constant hazard.
If no bracket is given, automatically finds the last positive→negative FOC crossing.
"""
function solve_feed_cost(p; T_bracket=nothing)
    if isnothing(T_bracket)
        T_bracket = find_foc_bracket(feed_cost_foc, p)
    end
    f(T) = feed_cost_foc(T, p)
    return find_zero(f, T_bracket, Bisection())
end

"""
    feed_cost_value(T, p)

Continuation value V with CRRA utility, feed costs, constant hazard,
and constant loss income Y_MIN:

  V = (δ+λ) · [∫₀ᵀ λ·e^{−(δ+λ)s}·u(Y_MIN) ds + e^{−(δ+λ)T}·u(Y_H(T))]
      / (δ · (1 − e^{−(δ+λ)T}))
"""
function feed_cost_value(T, p)
    δλ = p.δ + p.λ_const
    S_T = exp(-δλ * T)

    u_loss = u(p.Y_MIN, p)
    loss_integral = p.λ_const * u_loss * (1 - S_T) / δλ
    harvest_term = S_T * u(Y_H_feed(T, p), p)

    numerator = δλ * (loss_integral + harvest_term)
    denominator = p.δ * (1 - S_T)
    return numerator / denominator
end


# ──────────────────────────────────────────────────────────────────────────────
# Case 4: Add insurance (constant hazard, breakeven coverage)
#   Indemnity ODE has constant coefficients when λ is constant:
#     I'(t) = (λ/(1−Q) + δ_b)·I + η·w(t) + c_I/(1−Q) − δ_b·(c₂+Y_MIN)
#     I(0) = Y_MIN + c_s + c₂
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_indemnity_homogeneous(T, p)

Solve the indemnity ODE under constant hazard. Although the ODE has a
constant-coefficient linear part, the forcing term η·w(τ) is nonlinear
(von Bertalanffy growth), so we integrate numerically.

# Returns
A function `I(τ)` that returns the indemnity at age τ ∈ [0, T].
"""
function solve_indemnity_homogeneous(T, p)
    I₀ = p.Y_MIN + p.c_s + p.c₂
    r = p.λ_const / (1 - p.Q) + p.δ_b
    c_term = p.c_I / (1 - p.Q) - p.δ_b * (p.c₂ + p.Y_MIN)

    function dIdt(I, params, τ)
        return r * I + params.η * w_homogeneous(τ, params) + c_term
    end

    prob = ODEProblem(dIdt, I₀, (0.0, T), p)
    sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
    return sol
end

"""
    π_homogeneous(τ, I_sol, p)

Insurance premium rate under constant hazard:
  π(τ) = (λ·I(τ) + c_I) / (1 − Q)
"""
function π_homogeneous(τ, I_sol, p)
    return (p.λ_const * I_sol(τ) + p.c_I) / (1 - p.Q)
end

"""
    Π_homogeneous(T, I_sol, p)

Accumulated insurance premiums compounded to time T:
  Π(T) = ∫₀ᵀ π(s)·exp(δ_b·(T−s)) ds
"""
function Π_homogeneous(T, I_sol, p)
    integrand(s) = π_homogeneous(s, I_sol, p) * exp(p.δ_b * (T - s))
    val, _ = quadgk(integrand, 0.0, T)
    return val
end

"""
    Y_H_insurance(T, I_sol, p)

Harvest income with feed costs and insurance:
  Y_H = v(T) − c_s·exp(δ_b·T) − Φ(T) − Π(T) − c_h
"""
function Y_H_insurance(T, I_sol, p)
    return v_homogeneous(T, p) - p.c_s * exp(p.δ_b * T) -
           Φ_homogeneous(T, p) - Π_homogeneous(T, I_sol, p) - p.c_h
end

"""
    Y_L_insurance(τ, I_sol, p)

Loss income at time τ with insurance:
  Y_L = I(τ) − c_s·exp(δ_b·τ) − Φ(τ) − Π(τ) − c₂
"""
function Y_L_insurance(τ, I_sol, p)
    Φ_τ = Φ_homogeneous(τ, p)
    Π_τ = Π_homogeneous(τ, I_sol, p)
    return I_sol(τ) - p.c_s * exp(p.δ_b * τ) - Φ_τ - Π_τ - p.c₂
end

"""
    insurance_value(T, I_sol, p)

Continuation value V with CRRA utility, feed costs, insurance, and
constant hazard:

  V = (δ+λ) · [∫₀ᵀ λ·e^{−(δ+λ)s}·u(Y_MIN) ds + e^{−(δ+λ)T}·u(Y_H(T))]
      / (δ · (1 − e^{−(δ+λ)T}))

Under breakeven insurance, Y_L(s) = Y_MIN by construction of the indemnity
ODE, so the loss integral simplifies to a closed-form expression.
"""
function insurance_value(T, I_sol, p)
    δλ = p.δ + p.λ_const
    S_T = exp(-δλ * T)

    # Loss branch: Y_L = Y_MIN by breakeven insurance construction
    u_loss = u(p.Y_MIN, p)
    loss_integral = p.λ_const * u_loss * (1 - S_T) / δλ

    # Harvest branch
    harvest_term = S_T * u(Y_H_insurance(T, I_sol, p), p)

    numerator = δλ * (loss_integral + harvest_term)
    denominator = p.δ * (1 - S_T)
    return numerator / denominator
end

"""
    Π_homogeneous_prime(T, I_sol, p)

Derivative of accumulated insurance premiums with respect to T:
  dΠ/dT = π(T) + δ_b·Π(T)

(Leibniz rule: boundary term π(T)·exp(0) plus δ_b times the integral.)
"""
function Π_homogeneous_prime(T, I_sol, p)
    return π_homogeneous(T, I_sol, p) + p.δ_b * Π_homogeneous(T, I_sol, p)
end

"""
    Y_H_insurance_prime(T, I_sol, p)

Derivative of harvest income with feed costs and insurance:
  Y_H' = v'(T) − c_s·δ_b·exp(δ_b·T) − Φ'(T) − Π'(T)
"""
function Y_H_insurance_prime(T, I_sol, p)
    return v_homogeneous_prime(T, p) - p.c_s * p.δ_b * exp(p.δ_b * T) -
           Φ_homogeneous_prime(T, p) - Π_homogeneous_prime(T, I_sol, p)
end

"""
    insurance_foc(T, p)

Residual of the harvest FOC with CRRA utility, feed costs, insurance,
and constant hazard. Solves the indemnity ODE internally for each T.

  Y_H'(T)·u'(Y_H) − (δ+λ)·u(Y_H) / (1 − e^{−(δ+λ)T})

Returns +Inf when Y_H ≤ 0.
"""
function insurance_foc(T, p)
    δλ = p.δ + p.λ_const
    I_sol = solve_indemnity_homogeneous(T, p)
    yh = Y_H_insurance(T, I_sol, p)
    yh ≤ 0 && return Inf
    yh_prime = Y_H_insurance_prime(T, I_sol, p)
    return yh_prime * u_prime(yh, p) - δλ * u(yh, p) / (1 - exp(-δλ * T))
end

"""
    solve_insurance(p; T_bracket=nothing)

Solve for T* with CRRA utility, feed costs, insurance, and constant hazard.
If no bracket is given, automatically finds the last positive→negative FOC crossing.
"""
function solve_insurance(p; T_bracket=nothing)
    if isnothing(T_bracket)
        T_bracket = find_foc_bracket(insurance_foc, p)
    end
    f(T) = insurance_foc(T, p)
    return find_zero(f, T_bracket, Bisection())
end


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build homogeneous parameter set from default_params
# ──────────────────────────────────────────────────────────────────────────────

"""
    make_homogeneous_params(p; λ_const=nothing, k_const=nothing, m_const=nothing)

Create a parameter NamedTuple for the homogeneous model by adding constant
scalar rates `λ_const`, `k_const`, `m_const`. If not supplied, they default
to the mean of the corresponding seasonal function (i.e. exp(a0) when the
Fourier series has zero higher-order coefficients).
"""
function make_homogeneous_params(p; λ_const=nothing, k_const=nothing, m_const=nothing)
    λ_c = isnothing(λ_const) ? exp(p.λ_coeffs.a0) : λ_const
    k_c = isnothing(k_const) ? exp(p.k_coeffs.a0) : k_const
    m_c = isnothing(m_const) ? exp(p.m_coeffs.a0) : m_const
    return merge(p, (λ_const = λ_c, k_const = k_c, m_const = m_c))
end
