"""
    02_first_order_conditions.jl

Seasonal first-order conditions for the aquaculture bioeconomic model.
Implements the harvest FOC and stocking FOC from README §§ 7, 10–11 for
the full time-dependent (seasonal) case.

Given an approximate continuation value V(t) represented as Fourier
coefficients, this module:

1. Solves the **harvest FOC** at Fourier nodes to obtain the optimal
   cycle duration τ*(t₀) = T*(t₀) − t₀ as a periodic function of
   stocking date.
2. Computes the **cycle value** Ṽ(t₀) at each node.
3. Evaluates the **stocking FOC** residual Ṽ'(t₀) − δ·Ṽ(t₀) at nodes.
   The stocking time is typically a corner solution (immediate restocking,
   fallow = 0) when the residual is negative.
4. Fits Fourier series (with N harmonics) to the nodal solutions.

All periodic unknowns are fit as truncated Fourier series with period
365 days. The number of harmonics N controls accuracy; with 2N+1 equally
spaced nodes the Fourier interpolation is exact at the nodes.
"""

using Roots
using QuadGK
using OrdinaryDiffEq
using LinearAlgebra

include("00_model_functions.jl")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Fourier infrastructure
# ──────────────────────────────────────────────────────────────────────────────

const PERIOD = 365.0

"""
    fourier_nodes(N)

Return `2N+1` equally spaced points in [0, 365).
"""
function fourier_nodes(N)
    n_pts = 2N + 1
    return [i * PERIOD / n_pts for i in 0:(n_pts - 1)]
end

"""
    fourier_basis_matrix(t_nodes, N)

Build the (2N+1) × (2N+1) Fourier basis matrix for fitting.
Columns: [1, sin(ωt), cos(ωt), sin(2ωt), cos(2ωt), ...].
"""
function fourier_basis_matrix(t_nodes, N)
    n_pts = length(t_nodes)
    ω = 2π / PERIOD
    A = zeros(n_pts, 2N + 1)
    for (i, t) in enumerate(t_nodes)
        A[i, 1] = 1.0
        for k in 1:N
            A[i, 2k]     = sin(k * ω * t)
            A[i, 2k + 1] = cos(k * ω * t)
        end
    end
    return A
end

"""
    fit_fourier(t_nodes, f_values, N)

Fit a truncated Fourier series with `N` harmonics to values at `2N+1` nodes.
Returns `(a0, a, b)` NamedTuple compatible with `fourier_eval`/`fourier_derivative`.
"""
function fit_fourier(t_nodes, f_values, N)
    A = fourier_basis_matrix(t_nodes, N)
    coeffs = A \ f_values
    a0 = coeffs[1]
    a = [coeffs[2k] for k in 1:N]
    b = [coeffs[2k + 1] for k in 1:N]
    return (a0 = a0, a = a, b = b)
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cycle evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    prepare_cycle(t₀, T_max, p)

Pre-solve growth, mortality, and indemnity ODEs from `t₀` to `T_max`.
Returns `(L_sol, n_sol, I_sol)` that can be evaluated at any `t ∈ [t₀, T_max]`.
"""
function prepare_cycle(t₀, T_max, p)
    L_sol = solve_length(t₀, T_max, p.L₀, p)
    n_sol = solve_numbers(t₀, T_max, p.n₀, p)
    I_sol = solve_indemnity(t₀, T_max, L_sol, n_sol, p)
    return (L_sol = L_sol, n_sol = n_sol, I_sol = I_sol)
end

"""
    v_seasonal(t, L_sol, n_sol, p)

Stock value at calendar time `t` using precomputed ODE solutions.
"""
function v_seasonal(t, L_sol, n_sol, p)
    return n_sol(t) * f_value(L_sol(t), p)
end

"""
    Y_H_seasonal(T, t₀, L_sol, n_sol, I_sol, p)

Harvest income at planned harvest date `T` using precomputed ODE solutions.
  Y_H = v(T) − c_s·exp(δ_b·(T−t₀)) − Φ(T,t₀) − Π(T,t₀) − c_h
"""
function Y_H_seasonal(T, t₀, L_sol, n_sol, I_sol, p)
    v_T = v_seasonal(T, L_sol, n_sol, p)
    stocking = p.c_s * exp(p.δ_b * (T - t₀))
    Φ_val = Φ_accumulated(T, t₀, L_sol, n_sol, p)
    Π_val = Π_accumulated(T, t₀, I_sol, p)
    return v_T - stocking - Φ_val - Π_val - p.c_h
end

"""
    Y_L_seasonal(τ, t₀, L_sol, n_sol, I_sol, p)

Loss income at catastrophic event time `τ` using precomputed ODE solutions.
  Y_L = I(τ) − c_s·exp(δ_b·(τ−t₀)) − Φ(τ,t₀) − Π(τ,t₀) − c₂
"""
function Y_L_seasonal(τ, t₀, L_sol, n_sol, I_sol, p)
    stocking = p.c_s * exp(p.δ_b * (τ - t₀))
    Φ_val = Φ_accumulated(τ, t₀, L_sol, n_sol, p)
    Π_val = Π_accumulated(τ, t₀, I_sol, p)
    return I_sol(τ) - stocking - Φ_val - Π_val - p.c₂
end

"""
    Y_H_prime_seasonal(T, t₀, L_sol, n_sol, I_sol, p)

Derivative of harvest income with respect to T:
  dY_H/dT = dv/dT − c_s·δ_b·exp(δ_b·(T−t₀)) − dΦ/dT − dΠ/dT

where dv/dT = n(T)·[−m(T)·f(L) + f'(L)·k(T)·(L∞−L)]
and dΦ/dT = φ(T) + δ_b·Φ(T), dΠ/dT = π(T) + δ_b·Π(T).
"""
function Y_H_prime_seasonal(T, t₀, L_sol, n_sol, I_sol, p)
    L_T = L_sol(T)
    n_T = n_sol(T)
    W = W_weight(L_T, p)

    # df/dL via chain rule: f = W·σ(W), df/dW = σ + W·σ(1−σ)/s
    dWdL = p.ω * p.β * L_T^(p.β - 1)
    σ = 1.0 / (1.0 + exp(-(W - p.W₅₀) / p.s))
    dfdW = σ + W * σ * (1 - σ) / p.s
    dfdL = dfdW * dWdL

    f_L = f_value(L_T, p)
    k_T = k_growth(T, p)
    m_T = m_rate(T, p)

    # dv/dT via product rule on v = n·f(L)
    dv_dT = n_T * (-m_T * f_L + dfdL * k_T * (p.L∞ - L_T))

    # Cost derivatives via Leibniz rule
    stocking_prime = p.c_s * p.δ_b * exp(p.δ_b * (T - t₀))
    φ_T = p.η * v_seasonal(T, L_sol, n_sol, p)
    Φ_T = Φ_accumulated(T, t₀, L_sol, n_sol, p)
    dΦ_dT = φ_T + p.δ_b * Φ_T

    π_T = π_premium(T, I_sol, p)
    Π_T = Π_accumulated(T, t₀, I_sol, p)
    dΠ_dT = π_T + p.δ_b * Π_T

    return dv_dT - stocking_prime - dΦ_dT - dΠ_dT
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. Harvest FOC
# ──────────────────────────────────────────────────────────────────────────────

"""
    harvest_foc_residual(T, t₀, V_coeffs, L_sol, n_sol, I_sol, p)

Residual of the harvest FOC (README § 7):

  LHS: (∂Y_H/∂T) · u'(Y_H)
  RHS: δ·(V(T) + u(Y_H)) + λ(T)·(u(Y_H) − u(Y_L(T))) − V'(T)

Returns +Inf when Y_H ≤ 0 (not yet profitable).
"""
function harvest_foc_residual(T, t₀, V_coeffs, L_sol, n_sol, I_sol, p)
    yh = Y_H_seasonal(T, t₀, L_sol, n_sol, I_sol, p)
    yh ≤ 0 && return Inf

    yl = Y_L_seasonal(T, t₀, L_sol, n_sol, I_sol, p)
    yh_prime = Y_H_prime_seasonal(T, t₀, L_sol, n_sol, I_sol, p)

    V_T = fourier_eval(T, V_coeffs)
    V_prime_T = fourier_derivative(T, V_coeffs)
    λ_T = λ(T, p)

    u_yh = u(yh, p)
    u_yl = u(max(yl, 1e-10), p)  # guard against negative Y_L

    lhs = yh_prime * u_prime(yh, p)
    rhs = p.δ * (V_T + u_yh) + λ_T * (u_yh - u_yl) - V_prime_T

    return lhs - rhs
end

"""
    find_harvest_bracket(t₀, V_coeffs, L_sol, n_sol, I_sol, p;
                         τ_min=10.0, τ_max=1500.0, n_pts=2000)

Find a bracket around a sign change of the harvest FOC residual, scanning
over cycle duration τ = T − t₀. Only considers finite-valued (Y_H > 0)
transitions. Prefers the last finite positive→negative crossing. If no
exact crossing is found, returns a narrow bracket around the finite point
closest to zero (approximate root).
"""
function find_harvest_bracket(t₀, V_coeffs, L_sol, n_sol, I_sol, p;
                              τ_min=10.0, τ_max=1500.0, n_pts=2000)
    τs = range(τ_min, τ_max, length=n_pts)
    last_bracket = nothing
    best_idx = 1
    best_abs = Inf

    vals = Float64[]
    for i in 1:length(τs)
        T_i = t₀ + τs[i]
        v = harvest_foc_residual(T_i, t₀, V_coeffs, L_sol, n_sol, I_sol, p)
        push!(vals, v)

        if isfinite(v) && abs(v) < best_abs
            best_abs = abs(v)
            best_idx = i
        end
    end

    # Look for the last finite positive→negative crossing
    for i in 2:length(τs)
        if isfinite(vals[i-1]) && isfinite(vals[i]) && vals[i-1] > 0 && vals[i] < 0
            last_bracket = (t₀ + τs[i-1], t₀ + τs[i])
        end
    end

    if !isnothing(last_bracket)
        return last_bracket
    end

    # Fallback: narrow bracket around the minimum-|residual| point
    step = (τ_max - τ_min) / n_pts
    τ_lo = max(τ_min, τs[best_idx] - step)
    τ_hi = min(τ_max, τs[best_idx] + step)
    return (t₀ + τ_lo, t₀ + τ_hi)
end

"""
    solve_harvest_foc(t₀, V_coeffs, p; τ_max=1500.0)

Solve the harvest FOC for the optimal harvest date T* given stocking date t₀
and continuation value V(t) specified by Fourier coefficients.

Returns `T*` (a calendar date, not a duration).
"""
function solve_harvest_foc(t₀, V_coeffs, p; τ_max=1500.0)
    T_upper = t₀ + τ_max
    cycle = prepare_cycle(t₀, T_upper, p)

    bracket = find_harvest_bracket(t₀, V_coeffs, cycle.L_sol, cycle.n_sol, cycle.I_sol, p; τ_max=τ_max)
    f(T) = harvest_foc_residual(T, t₀, V_coeffs, cycle.L_sol, cycle.n_sol, cycle.I_sol, p)

    # If bracket values have the same sign (near-zero fallback), return the
    # point with smaller absolute residual
    f_lo, f_hi = f(bracket[1]), f(bracket[2])
    if !isfinite(f_lo) || !isfinite(f_hi) || f_lo * f_hi > 0
        return abs(f_lo) < abs(f_hi) ? bracket[1] : bracket[2]
    end

    return find_zero(f, bracket, Bisection())
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cycle value function Ṽ(t₀)
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_Vtilde(t₀, T_star, V_coeffs, p)

Compute the expected present utility of a cycle stocked at `t₀` with
planned harvest at `T_star`:

  Ṽ(t₀) = S(T*,t₀)·e^{−δ(T*−t₀)}·(u(Y_H(T*)) + V(T*))
         + ∫_{t₀}^{T*} S(s,t₀)·λ(s)·e^{−δ(s−t₀)}·(u(Y_L(s)) + V(s)) ds

The ODE solutions are computed internally.
"""
function compute_Vtilde(t₀, T_star, V_coeffs, p)
    cycle = prepare_cycle(t₀, T_star, p)

    # Harvest branch
    yh = Y_H_seasonal(T_star, t₀, cycle.L_sol, cycle.n_sol, cycle.I_sol, p)
    V_T = fourier_eval(T_star, V_coeffs)
    surv_T = S(t₀, T_star, p)
    disc_T = exp(-p.δ * (T_star - t₀))
    harvest_term = surv_T * disc_T * (u(max(yh, 1e-10), p) + V_T)

    # Loss branch integral
    function integrand(s)
        surv_s = S(t₀, s, p)
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        yl = Y_L_seasonal(s, t₀, cycle.L_sol, cycle.n_sol, cycle.I_sol, p)
        V_s = fourier_eval(s, V_coeffs)
        return surv_s * λ_s * disc_s * (u(max(yl, 1e-10), p) + V_s)
    end

    loss_integral, _ = quadgk(integrand, t₀ + 1e-6, T_star; rtol=1e-6)
    return harvest_term + loss_integral
end

"""
    compute_Vtilde_decomposed(t₀, T_star, p)

Decompose the cycle value into a utility-only component `f` and a
discount factor `g`, such that Ṽ(t₀) = f + g·V̄ when V(t) ≈ V̄ (constant):

  f = S(T*,t₀)·e^{−δτ}·u(Y_H) + ∫ S·λ·e^{−δ·…}·u(Y_L) ds
  g = S(T*,t₀)·e^{−δτ}        + ∫ S·λ·e^{−δ·…} ds

This decomposition enables a direct solve for V at each node:
  V(t) = e^{−δd*}·f / (1 − e^{−δd*}·g)
which converges in one step for the homogeneous case and dramatically
accelerates convergence for the seasonal case.

# Returns
`(f, g)` — the utility-only part and the expected discount factor.
"""
function compute_Vtilde_decomposed(t₀, T_star, p)
    cycle = prepare_cycle(t₀, T_star, p)

    # Harvest branch
    yh = Y_H_seasonal(T_star, t₀, cycle.L_sol, cycle.n_sol, cycle.I_sol, p)
    surv_T = S(t₀, T_star, p)
    disc_T = exp(-p.δ * (T_star - t₀))

    f_harvest = surv_T * disc_T * u(max(yh, 1e-10), p)
    g_harvest = surv_T * disc_T

    # Loss branch: separate utility and discount factor integrals
    function f_integrand(s)
        surv_s = S(t₀, s, p)
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        yl = Y_L_seasonal(s, t₀, cycle.L_sol, cycle.n_sol, cycle.I_sol, p)
        return surv_s * λ_s * disc_s * u(max(yl, 1e-10), p)
    end

    function g_integrand(s)
        surv_s = S(t₀, s, p)
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        return surv_s * λ_s * disc_s
    end

    f_loss, _ = quadgk(f_integrand, t₀ + 1e-6, T_star; rtol=1e-6)
    g_loss, _ = quadgk(g_integrand, t₀ + 1e-6, T_star; rtol=1e-6)

    f = f_harvest + f_loss
    g = g_harvest + g_loss

    return (f = f, g = g)
end

"""
    compute_Vtilde_deriv(t₀, T_star_coeffs, V_coeffs, p; h=0.1)

Numerical derivative dṼ/dt₀ via central finite differences.
T*(t₀) is evaluated from the Fourier series `T_star_coeffs` (which gives
the cycle duration τ*(t₀), so T* = t₀ + τ*(t₀)).
"""
function compute_Vtilde_deriv(t₀, T_star_coeffs, V_coeffs, p; h=0.1)
    τ_plus  = fourier_eval(t₀ + h, T_star_coeffs)
    τ_minus = fourier_eval(t₀ - h, T_star_coeffs)
    V_plus  = compute_Vtilde(t₀ + h, t₀ + h + τ_plus, V_coeffs, p)
    V_minus = compute_Vtilde(t₀ - h, t₀ - h + τ_minus, V_coeffs, p)
    return (V_plus - V_minus) / (2h)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. Stocking FOC
# ──────────────────────────────────────────────────────────────────────────────

"""
    stocking_foc_residual(t₀, T_star_coeffs, V_coeffs, p)

Residual of the stocking FOC (README § 10):
  Ṽ'(t₀) − δ · Ṽ(t₀)

A negative residual indicates that immediate restocking is optimal
(corner solution with fallow = 0).
"""
function stocking_foc_residual(t₀, T_star_coeffs, V_coeffs, p)
    τ_star = fourier_eval(t₀, T_star_coeffs)
    T_star = t₀ + τ_star
    Vtilde = compute_Vtilde(t₀, T_star, V_coeffs, p)
    Vtilde_prime = compute_Vtilde_deriv(t₀, T_star_coeffs, V_coeffs, p)
    return Vtilde_prime - p.δ * Vtilde
end

"""
    solve_stocking_foc(T_harvest, τ_star_coeffs, V_coeffs, p; d_max=180.0, n_scan=500)

Given harvest at calendar date `T_harvest`, find the optimal fallow duration
d* ≥ 0 before restocking. The optimal stocking date is t₀* = T_harvest + d*.

If the stocking FOC residual Ṽ'(t₀) − δ·Ṽ(t₀) is negative at t₀ = T_harvest,
then d* = 0 (corner solution: restock immediately).

If the residual is positive at t₀ = T_harvest, scans forward to find where
it crosses zero (interior solution). Returns the fallow duration d*.
"""
function solve_stocking_foc(T_harvest, τ_star_coeffs, V_coeffs, p;
                             d_max=180.0, n_scan=500)
    # Wrap harvest date into [0, 365)
    T_mod = mod(T_harvest, PERIOD)

    # Check residual at d=0 (immediate restocking)
    resid_0 = stocking_foc_residual(T_mod, τ_star_coeffs, V_coeffs, p)
    if resid_0 ≤ 0
        return 0.0  # corner solution
    end

    # Scan forward to find the positive→negative crossing
    ds = range(0.0, d_max, length=n_scan)
    prev_val = resid_0
    bracket = nothing

    for i in 2:length(ds)
        t₀_try = mod(T_mod + ds[i], PERIOD)
        val = stocking_foc_residual(t₀_try, τ_star_coeffs, V_coeffs, p)

        if isfinite(prev_val) && isfinite(val) && prev_val > 0 && val ≤ 0
            bracket = (ds[i-1], ds[i])
            break
        end
        prev_val = val
    end

    # If no crossing found, return d_max (residual stays positive)
    if isnothing(bracket)
        return d_max
    end

    # Bisect to refine
    f(d) = stocking_foc_residual(mod(T_mod + d, PERIOD), τ_star_coeffs, V_coeffs, p)
    return find_zero(f, bracket, Bisection())
end

"""
    solve_stocking_at_nodes(τ_star_coeffs, V_coeffs, p; N=10, d_max=180.0)

Solve the stocking FOC at `2N+1` Fourier nodes to obtain the optimal fallow
duration d*(T) as a periodic function of harvest date T, then fit a Fourier
series.

Returns `(d_star_coeffs, d_values, nodes, Vtilde_values, residuals)`.
"""
function solve_stocking_at_nodes(τ_star_coeffs, V_coeffs, p; N=10, d_max=180.0)
    nodes = fourier_nodes(N)
    d_values = Float64[]
    Vtilde_values = Float64[]
    residuals = Float64[]

    for T_harvest in nodes
        d_star = solve_stocking_foc(T_harvest, τ_star_coeffs, V_coeffs, p; d_max=d_max)
        push!(d_values, d_star)

        # Also record Ṽ and residual at the optimal stocking date
        t₀_star = mod(T_harvest + d_star, PERIOD)
        τ_star = fourier_eval(t₀_star, τ_star_coeffs)
        T_star = t₀_star + τ_star
        Vtilde = compute_Vtilde(t₀_star, T_star, V_coeffs, p)
        push!(Vtilde_values, Vtilde)
        push!(residuals, stocking_foc_residual(t₀_star, τ_star_coeffs, V_coeffs, p))
    end

    d_star_coeffs = fit_fourier(nodes, d_values, N)
    return (d_star_coeffs = d_star_coeffs, d_values = d_values, nodes = nodes,
            Vtilde_values = Vtilde_values, residuals = residuals)
end

"""
    solve_stocking_on_grid(τ_star_coeffs, V_coeffs, p; n_grid=100, d_max=180.0)

Solve the stocking FOC on a uniform grid for comparison with the Fourier
approximation. Returns `(t₀_grid, d_grid, Vtilde_values, residuals)`.
"""
function solve_stocking_on_grid(τ_star_coeffs, V_coeffs, p; n_grid=100, d_max=180.0)
    t₀_grid = range(0.0, PERIOD * (1 - 1/n_grid), length=n_grid)
    d_grid = Float64[]
    Vtilde_values = Float64[]
    residuals = Float64[]

    for T_harvest in t₀_grid
        d_star = solve_stocking_foc(T_harvest, τ_star_coeffs, V_coeffs, p; d_max=d_max)
        push!(d_grid, d_star)

        t₀_star = mod(T_harvest + d_star, PERIOD)
        τ_star = fourier_eval(t₀_star, τ_star_coeffs)
        T_star = t₀_star + τ_star
        Vtilde = compute_Vtilde(t₀_star, T_star, V_coeffs, p)
        push!(Vtilde_values, Vtilde)
        push!(residuals, stocking_foc_residual(t₀_star, τ_star_coeffs, V_coeffs, p))
    end

    return (t₀_grid = collect(t₀_grid), d_grid = d_grid,
            Vtilde_values = Vtilde_values, residuals = residuals)
end


# ──────────────────────────────────────────────────────────────────────────────
# 6. Node-level solvers
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_harvest_at_nodes(V_coeffs, p; N=10)

Solve the harvest FOC at `2N+1` Fourier nodes to obtain the optimal cycle
duration τ*(t₀) = T*(t₀) − t₀ at each node, then fit a Fourier series.

Returns `(τ_star_coeffs, τ_values, nodes)`.
"""
function solve_harvest_at_nodes(V_coeffs, p; N=10)
    nodes = fourier_nodes(N)
    τ_values = Float64[]

    for t₀ in nodes
        T_star = solve_harvest_foc(t₀, V_coeffs, p)
        push!(τ_values, T_star - t₀)
    end

    τ_star_coeffs = fit_fourier(nodes, τ_values, N)
    return (τ_star_coeffs = τ_star_coeffs, τ_values = τ_values, nodes = nodes)
end

"""
    solve_harvest_on_grid(V_coeffs, p; n_grid=100)

Solve the harvest FOC on a uniform grid of `n_grid` points in [0, 365)
for comparison with the Fourier approximation. Returns `(t₀_grid, τ_grid)`.
"""
function solve_harvest_on_grid(V_coeffs, p; n_grid=100)
    t₀_grid = range(0.0, PERIOD * (1 - 1/n_grid), length=n_grid)
    τ_grid = Float64[]

    for t₀ in t₀_grid
        T_star = solve_harvest_foc(t₀, V_coeffs, p)
        push!(τ_grid, T_star - t₀)
    end

    return (t₀_grid = collect(t₀_grid), τ_grid = τ_grid)
end

"""
    evaluate_stocking_foc_at_nodes(τ_star_coeffs, V_coeffs, p; N=10)

Evaluate the stocking FOC residual Ṽ'(t₀) − δ·Ṽ(t₀) at `2N+1` Fourier
nodes. A negative residual at a node indicates that immediate restocking
is optimal (corner solution).

Returns `(residuals, Vtilde_values, nodes)`.
"""
function evaluate_stocking_foc_at_nodes(τ_star_coeffs, V_coeffs, p; N=10)
    nodes = fourier_nodes(N)
    residuals = Float64[]
    Vtilde_values = Float64[]

    for t₀ in nodes
        τ_star = fourier_eval(t₀, τ_star_coeffs)
        T_star = t₀ + τ_star
        Vtilde = compute_Vtilde(t₀, T_star, V_coeffs, p)
        push!(Vtilde_values, Vtilde)

        Vtilde_prime = compute_Vtilde_deriv(t₀, τ_star_coeffs, V_coeffs, p)
        push!(residuals, Vtilde_prime - p.δ * Vtilde)
    end

    return (residuals = residuals, Vtilde_values = Vtilde_values, nodes = nodes)
end

"""
    evaluate_stocking_foc_on_grid(τ_star_coeffs, V_coeffs, p; n_grid=100)

Evaluate the stocking FOC residual on a uniform grid for comparison.
Returns `(t₀_grid, residuals, Vtilde_values)`.
"""
function evaluate_stocking_foc_on_grid(τ_star_coeffs, V_coeffs, p; n_grid=100)
    t₀_grid = range(0.0, PERIOD * (1 - 1/n_grid), length=n_grid)
    residuals = Float64[]
    Vtilde_values = Float64[]

    for t₀ in t₀_grid
        τ_star = fourier_eval(t₀, τ_star_coeffs)
        T_star = t₀ + τ_star
        Vtilde = compute_Vtilde(t₀, T_star, V_coeffs, p)
        push!(Vtilde_values, Vtilde)

        Vtilde_prime = compute_Vtilde_deriv(t₀, τ_star_coeffs, V_coeffs, p)
        push!(residuals, Vtilde_prime - p.δ * Vtilde)
    end

    return (t₀_grid = collect(t₀_grid), residuals = residuals, Vtilde_values = Vtilde_values)
end
