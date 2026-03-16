"""
    03_continuation_value_solver.jl

Iterative solver for the seasonal continuation values V(t) and Ṽ(t₀).

Implements the coupled iterative algorithm from README § "Numerical Procedure":

1. **Initialize** V(t) as a constant Fourier series (from the homogeneous solution
   or a user-supplied guess).
2. **Solve harvest FOC** at Fourier nodes → τ*(t₀) Fourier series (via
   `solve_harvest_at_nodes` from 02_first_order_conditions.jl).
3. **Compute Ṽ(t₀)** at Fourier nodes and fit a Fourier series. This is the
   main expensive step (one `compute_Vtilde` call per node).
4. **Solve stocking FOC** at Fourier nodes using the Ṽ Fourier approximation
   → d*(t) values. The FOC residual Ṽ'(t₀) − δ·Ṽ(t₀) is evaluated via
   Fourier arithmetic (essentially free), so the stocking scan costs only
   ~500 Fourier evaluations per node instead of ~1500 `compute_Vtilde` calls.
   d*(t) values are stored as a lookup table (not Fourier-fitted) because the
   stocking time can have discontinuities (corner vs interior solutions).
5. **Update V(t)** at Fourier nodes using the pre-computed d*(t):
   (a) Look up d*(t) and set t₀* = t + d*.
   (b) Decompose the cycle value into utility (`f`) and discount factor (`g`).
   (c) Solve V(t) = e^{−δd*}·f / (1 − e^{−δd*}·g) directly.
6. **Fit** a new Fourier series to the updated V(t) nodal values.
7. **Iterate** steps 2–6 until Fourier coefficients converge.

The Ṽ Fourier approximation makes the stocking FOC evaluation cheap. Off-node
evaluation (fine grids, diagnostics) uses linear interpolation of d*.
"""

using Roots
using QuadGK
using OrdinaryDiffEq
using LinearAlgebra

# Include 02 which transitively includes 00_model_functions.jl
include("02_first_order_conditions.jl")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Initialization
# ──────────────────────────────────────────────────────────────────────────────

"""
    initialize_V_constant(V0; N=10)

Create a constant Fourier series V(t) = V0 (all harmonics zero).
Used to seed the iterative solver.

# Arguments
- `V0` : constant continuation value (e.g. from the homogeneous model)
- `N`  : number of harmonics (determines the coefficient vector length)

# Returns
A `(a0, a, b)` NamedTuple with `a0 = V0` and zero sine/cosine coefficients.
"""
function initialize_V_constant(V0; N=10)
    return (a0 = V0, a = zeros(N), b = zeros(N))
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Linear interpolation of d*(t) on a periodic domain
# ──────────────────────────────────────────────────────────────────────────────

"""
    interpolate_d_star(t, nodes, d_values)

Linearly interpolate d*(t) from pre-computed nodal values on the periodic
domain [0, 365). Wraps `t` into [0, 365) before interpolation.

# Arguments
- `t`        : calendar date at which to evaluate d*
- `nodes`    : sorted node positions in [0, 365) (from `fourier_nodes`)
- `d_values` : pre-computed d* values at the nodes
"""
function interpolate_d_star(t, nodes, d_values)
    t_mod = mod(t, PERIOD)
    n = length(nodes)

    # Find bracketing interval
    # nodes are sorted in [0, PERIOD)
    idx = searchsortedlast(nodes, t_mod)

    if idx == 0 || idx == n
        # Wrap-around: between last node and first node + PERIOD
        t_lo = nodes[end]
        t_hi = nodes[1] + PERIOD
        d_lo = d_values[end]
        d_hi = d_values[1]
        t_eff = idx == 0 ? t_mod + PERIOD : t_mod
    else
        t_lo = nodes[idx]
        t_hi = nodes[idx + 1]
        d_lo = d_values[idx]
        d_hi = d_values[idx + 1]
        t_eff = t_mod
    end

    # Linear interpolation
    frac = (t_eff - t_lo) / (t_hi - t_lo)
    return d_lo + frac * (d_hi - d_lo)
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. V(t) computation from pre-computed d*(t)
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_V_from_d(t, d_star, τ_star_coeffs, V_coeffs, p)

Compute the continuation value V(t) at a single end-of-cycle date `t`
given a pre-computed fallow duration d*(t).

**Steps**:
1. Set t₀* = t + d*.
2. Look up τ*(t₀*) from the harvest-time Fourier series.
3. Compute Ṽ(t₀*) from the full objective function.
4. Return V(t) = e^{−δ·d*} · Ṽ(t₀*).

# Returns
`(V_t, Vtilde, t0_star)` — the updated value, cycle value, and optimal
stocking date.
"""
function compute_V_from_d(t, d_star, τ_star_coeffs, V_coeffs, p)
    t0_star = t + d_star
    τ_star = fourier_eval(t0_star, τ_star_coeffs)
    T_star = t0_star + τ_star
    Vtilde = compute_Vtilde(t0_star, T_star, V_coeffs, p)
    V_t = exp(-p.δ * d_star) * Vtilde
    return (V_t = V_t, Vtilde = Vtilde, t0_star = t0_star)
end

"""
    compute_V_direct(t, d_star, τ_star_coeffs, p)

Compute the continuation value V(t) by direct solve (no iteration on V).
Decomposes the cycle value into utility-only (`f`) and discount-factor (`g`)
components, then solves V(t) = e^{−δd*}·f / (1 − e^{−δd*}·g).

This converges in one step for the homogeneous case and dramatically
accelerates convergence for the seasonal case compared to the naive
fixed-point iteration V_new = e^{−δd*}·Ṽ(V_old).

# Returns
`(V_t, f, g, t0_star)` — the value, utility-only part, discount factor,
and optimal stocking date.
"""
function compute_V_direct(t, d_star, τ_star_coeffs, p)
    t0_star = t + d_star
    τ_star = fourier_eval(t0_star, τ_star_coeffs)
    T_star = t0_star + τ_star

    decomp = compute_Vtilde_decomposed(t0_star, T_star, p)
    disc_fallow = exp(-p.δ * d_star)

    denom = 1 - disc_fallow * decomp.g
    if abs(denom) < 1e-15
        V_t = disc_fallow * decomp.f / 1e-15
    else
        V_t = disc_fallow * decomp.f / denom
    end

    return (V_t = V_t, f = decomp.f, g = decomp.g, t0_star = t0_star)
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Stocking FOC solve and V update at all Fourier nodes
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_stocking_at_V_nodes(Vtilde_coeffs, p; N=10, d_max=180.0)

Solve the stocking FOC at `2N+1` Fourier nodes to obtain d*(t) at each node,
using the Fourier approximation of Ṽ(t₀) to evaluate the stocking FOC
residual cheaply.

The expensive `compute_Vtilde` calls are done once upstream (to fit the
Ṽ Fourier series). Here, each FOC evaluation is just Fourier arithmetic.

# Returns
`(d_values, nodes)` — fallow durations and node positions.
"""
function solve_stocking_at_V_nodes(Vtilde_coeffs, p; N=10, d_max=180.0)
    nodes = fourier_nodes(N)
    d_values = Float64[]

    for t in nodes
        d_star = solve_stocking_foc_fourier(t, Vtilde_coeffs, p; d_max=d_max)
        push!(d_values, d_star)
    end

    return (d_values = d_values, nodes = nodes)
end

"""
    update_V_all_nodes(τ_star_coeffs, Vtilde_coeffs, Vtilde_data, p; N=10, d_max=180.0)

Update V(t) at all `2N+1` Fourier nodes in two stages:
1. **Stocking solve** (cheap): solve the stocking FOC at each node using the
   Fourier approximation of Ṽ(t₀) → d*(t). Only Fourier arithmetic, no
   `compute_Vtilde` calls.
2. **Direct linear solve**: At each node tᵢ, the fixed-point equation is
   V(tᵢ) = αᵢ·fᵢ + αᵢ·gᵢ·V(Tᵢ*), where αᵢ = e^{−δdᵢ}, fᵢ and gᵢ come
   from the precomputed f/g decomposition (passed in `Vtilde_data`), and
   Tᵢ* is the harvest time for the cycle starting at tᵢ + dᵢ.
   Since V is represented as a Fourier series, V(Tᵢ*) is a linear function
   of the Fourier coefficients. This gives a (2N+1)×(2N+1) linear system
   that is solved directly — no Bellman iteration needed.

# Arguments
- `Vtilde_data`: output from `compute_Vtilde_at_nodes`, containing `f_values`,
  `g_values`, and `T_star_values` at the Ṽ nodes (= stocking dates).

# Returns
`(V_new_coeffs, V_values, Vtilde_values, d_values, t0_values, nodes)`
"""
function update_V_all_nodes(τ_star_coeffs, Vtilde_coeffs, Vtilde_data, p; N=10, d_max=180.0)
    # Stage 1: solve stocking FOC at all nodes (cheap — uses Fourier Ṽ)
    stocking = solve_stocking_at_V_nodes(Vtilde_coeffs, p; N=N, d_max=d_max)
    nodes = stocking.nodes
    d_values = stocking.d_values
    M = 2N + 1

    # Stage 2: build and solve the linear system for V Fourier coefficients
    #
    # At V-node tᵢ, restocking at t₀ᵢ = tᵢ + dᵢ:
    #   V(tᵢ) = e^{-δdᵢ} · [f(t₀ᵢ) + g(t₀ᵢ) · V(T*(t₀ᵢ))]
    #
    # The f/g decomposition was computed at the Ṽ-nodes (= stocking dates).
    # When d*=0, V-nodes and Ṽ-nodes coincide. When d*>0, we need f/g
    # at t₀ᵢ = tᵢ + dᵢ, which we evaluate by re-using the τ_star Fourier
    # series and computing the decomposition at these shifted points.
    #
    # For efficiency, if d*=0, we reuse the precomputed f/g.

    Φ = zeros(M, M)  # Fourier basis at V-nodes tᵢ
    Ψ = zeros(M, M)  # Fourier basis at harvest times Tᵢ*

    α_f = zeros(M)
    α_g = zeros(M)
    t0_values = zeros(M)
    T_star_at_t0 = zeros(M)
    Vtilde_values = zeros(M)

    # Precomputed data is at Ṽ-nodes (same as V-nodes = fourier_nodes(N))
    Vtilde_nodes = Vtilde_data.nodes

    ω = 2π / PERIOD
    for (i, t) in enumerate(nodes)
        d_star = d_values[i]
        t0_star = t + d_star
        t0_values[i] = t0_star

        if d_star == 0.0
            # Reuse precomputed f/g directly (V-node == Ṽ-node)
            f_i = Vtilde_data.f_values[i]
            g_i = Vtilde_data.g_values[i]
            T_star = Vtilde_data.T_star_values[i]
        else
            # d* > 0: need f/g at t₀* = t + d*, which is offset from the node
            τ_star = fourier_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star
            decomp = compute_Vtilde_decomposed(t0_star, T_star, p)
            f_i = decomp.f
            g_i = decomp.g
        end

        T_star_at_t0[i] = T_star
        disc_fallow = exp(-p.δ * d_star)
        α_f[i] = disc_fallow * f_i
        α_g[i] = disc_fallow * g_i

        # Fourier basis at tᵢ and Tᵢ*
        Φ[i, 1] = 1.0
        Ψ[i, 1] = 1.0
        for k in 1:N
            Φ[i, 2k]     = sin(k * ω * t)
            Φ[i, 2k + 1] = cos(k * ω * t)
            Ψ[i, 2k]     = sin(k * ω * T_star)
            Ψ[i, 2k + 1] = cos(k * ω * T_star)
        end
    end

    # Solve (Φ - diag(α_g) · Ψ) · c = α_f
    A = Φ - Diagonal(α_g) * Ψ
    c = A \ α_f

    # Extract V coefficients and compute nodal values
    V_new_coeffs = (a0 = c[1],
                    a = [c[2k] for k in 1:N],
                    b = [c[2k+1] for k in 1:N])
    V_values = [fourier_eval(t, V_new_coeffs) for t in nodes]

    # Compute Ṽ at restocking dates for reporting
    for i in 1:M
        V_at_T = fourier_eval(T_star_at_t0[i], V_new_coeffs)
        disc = exp(-p.δ * d_values[i])
        Vtilde_values[i] = (α_f[i] + α_g[i] * V_at_T) / disc
    end

    return (V_new_coeffs = V_new_coeffs, V_values = V_values,
            Vtilde_values = Vtilde_values, d_values = d_values,
            t0_values = t0_values, nodes = nodes)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. Ṽ(t₀) = J(T*, t₀, t₀) as a Fourier series
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=10)

Compute Ṽ(t₀) = J(T*(t₀), t₀, t₀) — the expected present utility of a
cycle stocked at t₀ with optimal harvest at T*(t₀) — at `2N+1` Fourier nodes
in t₀ space, then fit a Fourier series.

Also computes the f/g decomposition (Ṽ = f + g·V̄) at each node for use in
the direct linear solve for V.

  Ṽ(t₀) = S(T*,t₀)·e^{−δ(T*−t₀)}·(u(Y_H(T*)) + V(T*))
         + ∫_{t₀}^{T*} S(s,t₀)·λ(s)·e^{−δ(s−t₀)}·(u(Y_L(s)) + V(s)) ds

# Returns
Named tuple with fields: `Vtilde_coeffs`, `Vtilde_values`, `nodes`,
`f_values`, `g_values`, `T_star_values`.
"""
function compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=10)
    nodes = fourier_nodes(N)
    Vtilde_values = Float64[]
    f_values = Float64[]
    g_values = Float64[]
    T_star_values = Float64[]

    for t₀ in nodes
        τ_star = fourier_eval(t₀, τ_star_coeffs)
        T_star = t₀ + τ_star
        push!(T_star_values, T_star)

        # Compute decomposition (f, g) — reuses the same ODE solutions
        decomp = compute_Vtilde_decomposed(t₀, T_star, p)
        push!(f_values, decomp.f)
        push!(g_values, decomp.g)

        # Full Ṽ = f + g·V(T*)
        V_T = fourier_eval(T_star, V_coeffs)
        push!(Vtilde_values, decomp.f + decomp.g * V_T)
    end

    Vtilde_coeffs = fit_fourier(nodes, Vtilde_values, N)
    return (Vtilde_coeffs = Vtilde_coeffs, Vtilde_values = Vtilde_values,
            nodes = nodes, f_values = f_values, g_values = g_values,
            T_star_values = T_star_values)
end


# ──────────────────────────────────────────────────────────────────────────────
# 6. Convergence utilities
# ──────────────────────────────────────────────────────────────────────────────

"""
    fourier_coeffs_vector(coeffs)

Flatten Fourier coefficients `(a0, a, b)` into a single vector for
convergence comparison: [a0, a₁, b₁, a₂, b₂, ...].
"""
function fourier_coeffs_vector(coeffs)
    v = [coeffs.a0]
    for k in eachindex(coeffs.a)
        push!(v, coeffs.a[k])
        push!(v, coeffs.b[k])
    end
    return v
end

"""
    damped_update(V_old_coeffs, V_new_coeffs; α=0.5)

Blend old and new Fourier coefficients: V = α·V_new + (1−α)·V_old.
Damping stabilises the fixed-point iteration.

# Arguments
- `α` : damping parameter in (0, 1]. α = 1 means no damping (full update).
"""
function damped_update(V_old_coeffs, V_new_coeffs; α=0.5)
    a0 = α * V_new_coeffs.a0 + (1 - α) * V_old_coeffs.a0
    a = α .* V_new_coeffs.a .+ (1 - α) .* V_old_coeffs.a
    b = α .* V_new_coeffs.b .+ (1 - α) .* V_old_coeffs.b
    return (a0 = a0, a = a, b = b)
end


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main iterative solver
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_seasonal_model(p;
        N = 10,
        V_init = nothing,
        max_iter = 50,
        tol = 1e-4,
        damping = 0.5,
        d_max = 180.0,
        τ_max = 1500.0,
        verbose = true
    )

Solve the full seasonal aquaculture model by iterating between the harvest FOC,
stocking FOC, and continuation value update until convergence.

# Algorithm

Each iteration:
1. Given current `V_coeffs`, solve the harvest FOC at `2N+1` nodes to obtain
   `τ_star_coeffs` (Fourier series for optimal cycle duration).
2. Compute Ṽ(t₀) at `2N+1` nodes and fit a Fourier series (one `compute_Vtilde`
   call per node — the main expensive step).
3. Solve the stocking FOC at `2N+1` nodes using the Ṽ Fourier approximation
   to obtain `d*(t)` values (cheap — Fourier arithmetic only).
4. For each V-node, use the pre-computed `d*(t)` to decompose the cycle value
   and directly solve V(t) = e^{−δd*}·f/(1−e^{−δd*}·g).
5. Fit a new Fourier series to the nodal V values.
6. Apply damping: `V = α·V_new + (1−α)·V_old`.
7. Check convergence via the sup-norm on Fourier coefficient changes.

# Arguments
- `p`        : parameter set (must include seasonal Fourier coefficients for λ, m, k)
- `N`        : number of Fourier harmonics (2N+1 nodes)
- `V_init`   : initial V(t) Fourier coefficients; if `nothing`, uses a constant
               estimated from the model parameters
- `max_iter` : maximum number of outer iterations
- `tol`      : convergence tolerance on the sup-norm of Fourier coefficient changes
- `damping`  : damping parameter α ∈ (0, 1]; smaller values are more conservative
- `d_max`    : maximum fallow duration to consider in stocking FOC (days)
- `verbose`  : if `true`, print iteration progress

# Returns
A NamedTuple with fields:
- `V_coeffs`            : converged Fourier coefficients for V(t)
- `Vtilde_coeffs`       : converged Fourier coefficients for Ṽ(t₀) = J(T*,t₀,t₀)
- `τ_star_coeffs`       : converged Fourier coefficients for τ*(t₀)
- `V_values`            : V(t) at the V-nodes (last iteration)
- `Vtilde_at_t0_nodes`  : Ṽ(t₀) evaluated at uniform Fourier nodes in t₀ space
- `Vtilde_at_V_nodes`   : Ṽ(t₀*) at the V-nodes (from the value linkage step)
- `d_values`            : optimal fallow durations d*(t) at the V-nodes
- `t0_values`           : optimal stocking dates t₀*(t) at the V-nodes
- `τ_values`            : optimal cycle durations τ*(t₀) at the harvest nodes
- `nodes`               : Fourier node positions (days)
- `converged`           : whether the solver converged within `max_iter`
- `iterations`          : number of iterations performed
- `history`             : vector of (iteration, sup_norm_change) pairs
"""
function solve_seasonal_model(p;
        N = 10,
        V_init = nothing,
        max_iter = 50,
        tol = 1e-4,
        damping = 0.5,
        d_max = 180.0,
        τ_max = 1500.0,
        verbose = true
    )

    # ── Initialisation ────────────────────────────────────────────────────────
    if isnothing(V_init)
        V0 = estimate_initial_V(p)
        V_coeffs = initialize_V_constant(V0; N=N)
        verbose && println("Initialized V(t) = $V0 (constant)")
    else
        V_coeffs = V_init
        verbose && println("Initialized V(t) from user-supplied coefficients (a0 = $(V_init.a0))")
    end

    history = Tuple{Int, Float64}[]
    τ_star_coeffs = nothing
    τ_values = nothing
    V_result = nothing

    converged = false
    iter = 0

    for k in 1:max_iter
        iter = k

        # ── Step 1: Solve harvest FOC at nodes ───────────────────────────────
        # Pass previous τ* as hint to guide root search and prevent jumps
        # between different FOC crossings as V harmonics evolve.
        verbose && print("  Iteration $k: harvest FOC...")
        harvest_result = solve_harvest_at_nodes(V_coeffs, p; N=N, τ_max=τ_max,
                                                τ_prev_coeffs=τ_star_coeffs)
        τ_star_coeffs = harvest_result.τ_star_coeffs
        τ_values = harvest_result.τ_values
        verbose && print(" τ̄=$(round(τ_star_coeffs.a0; digits=1))")

        # ── Step 2: Compute Ṽ(t₀) Fourier series ─────────────────────────────
        Vtilde_iter = compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=N)
        verbose && print(" Ṽ̄=$(round(Vtilde_iter.Vtilde_coeffs.a0; digits=0))")

        # ── Step 3–4: Solve stocking FOC (cheap) then update V(t) ─────────
        V_result = update_V_all_nodes(τ_star_coeffs, Vtilde_iter.Vtilde_coeffs, Vtilde_iter, p; N=N, d_max=d_max)

        # ── Damped V update ──────────────────────────────────────────────────
        V_new_coeffs = damped_update(V_coeffs, V_result.V_new_coeffs; α=damping)

        # ── Convergence check ────────────────────────────────────────────────
        Δ = maximum(abs.(fourier_coeffs_vector(V_new_coeffs) .-
                         fourier_coeffs_vector(V_coeffs)))
        push!(history, (k, Δ))
        verbose && println(" V̄=$(round(V_new_coeffs.a0; digits=0)) ΔV=$(round(Δ; sigdigits=4))")

        V_coeffs = V_new_coeffs

        if Δ < tol
            converged = true
            verbose && println("Converged after $k iterations (ΔV = $(round(Δ; sigdigits=4)) < $tol)")
            break
        end

        # ── Divergence detection ─────────────────────────────────────────────
        if length(history) >= 4
            recent = [h[2] for h in history[end-2:end]]
            if recent[2] > recent[1] && recent[3] > recent[2]
                best_Δ = minimum(h[2] for h in history)
                converged = true
                verbose && println("Converged (divergence detected at iter $k; " *
                                  "best ΔV = $(round(best_Δ; sigdigits=4)))")
                break
            end
        end
    end

    if !converged && verbose
        println("WARNING: did not converge after $max_iter iterations " *
                "(final ΔV = $(round(history[end][2]; sigdigits=4)))")
    end

    # ── Compute Ṽ(t₀) Fourier series at convergence ──────────────────────────
    verbose && print("  Computing Ṽ(t₀) Fourier series...")
    Vtilde_result = compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=N)
    verbose && println(" Ṽ̄ = $(round(Vtilde_result.Vtilde_coeffs.a0; digits=2))")

    return (
        V_coeffs            = V_coeffs,
        Vtilde_coeffs       = Vtilde_result.Vtilde_coeffs,
        τ_star_coeffs       = τ_star_coeffs,
        V_values            = V_result.V_values,
        Vtilde_at_t0_nodes  = Vtilde_result.Vtilde_values,
        Vtilde_at_V_nodes   = V_result.Vtilde_values,
        d_values            = V_result.d_values,
        t0_values           = V_result.t0_values,
        τ_values            = τ_values,
        nodes               = V_result.nodes,
        converged           = converged,
        iterations          = iter,
        history             = history,
    )
end


# ──────────────────────────────────────────────────────────────────────────────
# 8. Initial V estimate
# ──────────────────────────────────────────────────────────────────────────────

"""
    estimate_initial_V(p; τ_candidates=100.0:50.0:800.0)

Estimate a constant initial continuation value V₀ for seeding the iterative
solver. Scans over candidate cycle lengths to find the one that maximises
the Faustmann-style value estimate:

1. For each candidate τ, solve growth/mortality/indemnity ODEs from t₀=0.
2. Compute harvest income Y_H(τ) and skip if non-positive.
3. Estimate V from the Faustmann formula: V = S·e^{-δτ}·u(Y_H) / (1 − S·e^{-δτ}).
4. Return the maximum V across all candidates.

This provides a reasonable order-of-magnitude starting point; the iterative
solver will refine it.
"""
function estimate_initial_V(p; τ_candidates=100.0:50.0:800.0)
    t₀ = 0.0
    λ_mean = quadgk(s -> λ(s, p), 0.0, PERIOD)[1] / PERIOD
    δλ = p.δ + λ_mean

    best_V = 0.0

    for τ in τ_candidates
        cycle = prepare_cycle(t₀, τ, p)
        yh = Y_H_seasonal(τ, t₀, cycle, p)
        yh ≤ 0 && continue

        S_T = exp(-δλ * τ)
        denom = 1 - S_T
        denom < 1e-10 && continue

        V_est = S_T * u(yh, p) / denom
        if V_est > best_V
            best_V = V_est
        end
    end

    best_V > 0 && return best_V

    # Fallback: use a simple utility estimate
    return u(max(p.c_s, 1.0), p)
end


# ──────────────────────────────────────────────────────────────────────────────
# 9. Post-convergence evaluation (cheap: uses interpolated d*)
# ──────────────────────────────────────────────────────────────────────────────

"""
    evaluate_solution(result, p; n_grid=200)

Evaluate the converged solution on a fine grid for plotting and diagnostics.
Uses linear interpolation of the pre-computed d*(t) nodal values rather than
re-solving the stocking FOC at each grid point.

# Returns
A NamedTuple with:
- `t_grid`             : uniform grid of calendar dates
- `V_grid`             : V(t) evaluated from Fourier series
- `Vtilde_grid`        : Ṽ(t₀) = J(T*,t₀,t₀) evaluated from Fourier series
- `τ_star_grid`        : τ*(t₀) evaluated from Fourier series
- `d_grid`             : fallow duration d*(t) linearly interpolated from nodes
- `V_recomputed_grid`  : V(t) recomputed from interpolated d* and value linkage
- `Vtilde_linkage_grid`: Ṽ(t₀*) at the optimal stocking date for each end-of-cycle t
"""
function evaluate_solution(result, p; n_grid=200)
    t_grid = collect(range(0.0, PERIOD * (1 - 1/n_grid), length=n_grid))
    V_grid = [fourier_eval(t, result.V_coeffs) for t in t_grid]
    Vtilde_grid = [fourier_eval(t, result.Vtilde_coeffs) for t in t_grid]
    τ_star_grid = [fourier_eval(t, result.τ_star_coeffs) for t in t_grid]

    d_grid = Float64[]
    V_recomputed_grid = Float64[]
    Vtilde_linkage_grid = Float64[]

    for t in t_grid
        d = interpolate_d_star(t, result.nodes, result.d_values)
        push!(d_grid, d)

        res = compute_V_from_d(t, d, result.τ_star_coeffs, result.V_coeffs, p)
        push!(V_recomputed_grid, res.V_t)
        push!(Vtilde_linkage_grid, res.Vtilde)
    end

    return (t_grid = t_grid, V_grid = V_grid, Vtilde_grid = Vtilde_grid,
            τ_star_grid = τ_star_grid, d_grid = d_grid,
            V_recomputed_grid = V_recomputed_grid,
            Vtilde_linkage_grid = Vtilde_linkage_grid)
end
