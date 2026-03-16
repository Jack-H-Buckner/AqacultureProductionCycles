"""
    03_continuation_value_solver.jl

Iterative solver for the seasonal continuation values V(t) and Ṽ(t₀).

Implements the coupled iterative algorithm from README § "Numerical Procedure":

1. **Initialize** V(t) as a constant periodic linear spline (from the homogeneous
   solution or a user-supplied guess).
2. **Solve harvest FOC** at equally spaced nodes → τ*(t₀) spline (via
   `solve_harvest_at_nodes` from 02_first_order_conditions.jl).
3. **Compute Ṽ(t₀)** at nodes and fit a linear spline. This is the main
   expensive step (one `compute_Vtilde` call per node).
4. **Solve stocking FOC** at nodes using the Ṽ spline approximation → d*(t)
   values. The FOC residual Ṽ'(t₀) − δ·Ṽ(t₀) is evaluated via spline
   interpolation (essentially free), so the stocking scan costs only ~500
   spline evaluations per node instead of ~1500 `compute_Vtilde` calls.
   d*(t) values are stored as a lookup table because the stocking time can
   have discontinuities (corner vs interior solutions).
5. **Update V(t)** at nodes using the pre-computed d*(t):
   (a) Look up d*(t) and set t₀* = t + d*.
   (b) Decompose the cycle value into utility (`f`) and discount factor (`g`).
   (c) Solve V(t) via a linear system using spline interpolation weights.
6. **Iterate** steps 2–5 until nodal values converge.

Linear splines can represent discontinuities and sharp features that arise
in continuation values without Gibbs ringing artifacts.
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

Create a constant periodic linear spline V(t) = V0 at 2N+1 equally spaced nodes.
Used to seed the iterative solver.

# Arguments
- `V0` : constant continuation value (e.g. from the homogeneous model)
- `N`  : determines number of nodes (2N+1)

# Returns
A `(nodes, values)` NamedTuple with constant values.
"""
function initialize_V_constant(V0; N=10)
    nodes = fourier_nodes(N)
    return make_spline(nodes, fill(V0, 2N + 1))
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
- `nodes`    : sorted node positions in [0, 365)
- `d_values` : pre-computed d* values at the nodes
"""
function interpolate_d_star(t, nodes, d_values)
    return spline_eval(t, make_spline(nodes, d_values))
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
2. Look up τ*(t₀*) from the harvest-time spline.
3. Compute Ṽ(t₀*) from the full objective function.
4. Return V(t) = e^{−δ·d*} · Ṽ(t₀*).

# Returns
`(V_t, Vtilde, t0_star)` — the updated value, cycle value, and optimal
stocking date.
"""
function compute_V_from_d(t, d_star, τ_star_coeffs, V_coeffs, p)
    t0_star = t + d_star
    τ_star = spline_eval(t0_star, τ_star_coeffs)
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
    τ_star = spline_eval(t0_star, τ_star_coeffs)
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
# 4. Stocking FOC solve and V update at all nodes
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_stocking_at_V_nodes(Vtilde_coeffs, p; N=10, d_max=180.0)

Solve the stocking FOC at `2N+1` nodes to obtain d*(t) at each node,
using the spline approximation of Ṽ(t₀) to evaluate the stocking FOC
residual cheaply.

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

Update V(t) at all `2N+1` nodes in two stages:
1. **Stocking solve** (cheap): solve the stocking FOC at each node using the
   spline approximation of Ṽ(t₀) → d*(t). Only spline arithmetic, no
   `compute_Vtilde` calls.
2. **Direct linear solve**: At each node tᵢ, the fixed-point equation is
   V(tᵢ) = αᵢ·fᵢ + αᵢ·gᵢ·V(Tᵢ*), where αᵢ = e^{−δdᵢ}, fᵢ and gᵢ come
   from the precomputed f/g decomposition (passed in `Vtilde_data`), and
   Tᵢ* is the harvest time for the cycle starting at tᵢ + dᵢ.
   Since V is represented as a linear spline, V(Tᵢ*) is a weighted average
   of two bracketing nodal values. This gives a (2N+1)×(2N+1) sparse linear
   system that is solved directly — no Bellman iteration needed.

# Arguments
- `Vtilde_data`: output from `compute_Vtilde_at_nodes`, containing `f_values`,
  `g_values`, and `T_star_values` at the Ṽ nodes (= stocking dates).

# Returns
`(V_new_coeffs, V_values, Vtilde_values, d_values, t0_values, nodes)`
"""
function update_V_all_nodes(τ_star_coeffs, Vtilde_coeffs, Vtilde_data, p; N=10, d_max=180.0)
    # Stage 1: solve stocking FOC at all nodes (cheap — uses spline Ṽ)
    stocking = solve_stocking_at_V_nodes(Vtilde_coeffs, p; N=N, d_max=d_max)
    nodes = stocking.nodes
    d_values = stocking.d_values
    M = 2N + 1

    # Stage 2: build and solve the linear system for V nodal values
    #
    # At node tᵢ, restocking at t₀ᵢ = tᵢ + dᵢ:
    #   V(tᵢ) = e^{-δdᵢ} · [f(t₀ᵢ) + g(t₀ᵢ) · V(T*(t₀ᵢ))]
    #
    # Since V is a linear spline, V(Tᵢ*) is a weighted average of two
    # bracketing nodal values: V(Tᵢ*) = (1-w)·V[j] + w·V[j+1].
    # This gives a linear system (I - diag(α_g) · W) · v = α_f
    # where W is the spline interpolation weight matrix.

    W = zeros(M, M)  # Spline interpolation weights: V(Tᵢ*) = Σⱼ W[i,j] · vⱼ

    α_f = zeros(M)
    α_g = zeros(M)
    t0_values = zeros(M)
    T_star_at_t0 = zeros(M)
    Vtilde_values = zeros(M)

    for (i, t) in enumerate(nodes)
        d_star = d_values[i]
        t0_star = t + d_star
        t0_values[i] = t0_star

        if d_star == 0.0
            # Reuse precomputed f/g directly (node == Ṽ-node)
            f_i = Vtilde_data.f_values[i]
            g_i = Vtilde_data.g_values[i]
            T_star = Vtilde_data.T_star_values[i]
        else
            # d* > 0: need f/g at t₀* = t + d*, which is offset from the node
            τ_star = spline_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star
            decomp = compute_Vtilde_decomposed(t0_star, T_star, p)
            f_i = decomp.f
            g_i = decomp.g
        end

        T_star_at_t0[i] = T_star
        disc_fallow = exp(-p.δ * d_star)
        α_f[i] = disc_fallow * f_i
        α_g[i] = disc_fallow * g_i

        # Spline interpolation weights for V(Tᵢ*)
        iw = spline_interp_weights(T_star, nodes)
        W[i, iw.idx_lo] += 1 - iw.weight
        W[i, iw.idx_hi] += iw.weight
    end

    # Solve (I - diag(α_g) · W) · v = α_f
    A = Matrix{Float64}(I, M, M) - Diagonal(α_g) * W
    v = A \ α_f

    # The nodal values ARE the spline coefficients
    V_new_coeffs = make_spline(nodes, v)
    V_values = copy(v)

    # Compute Ṽ at restocking dates for reporting
    for i in 1:M
        V_at_T = spline_eval(T_star_at_t0[i], V_new_coeffs)
        disc = exp(-p.δ * d_values[i])
        Vtilde_values[i] = (α_f[i] + α_g[i] * V_at_T) / disc
    end

    return (V_new_coeffs = V_new_coeffs, V_values = V_values,
            Vtilde_values = Vtilde_values, d_values = d_values,
            t0_values = t0_values, nodes = nodes)
end


"""
    update_V_all_nodes_no_fallow(τ_star_coeffs, Vtilde_data, p; N=10)

Update V(t) at all `2N+1` nodes with forced d*=0 (immediate restocking).
When d*=0, V(t) = Ṽ(t) = f(t) + g(t)·V(T*(t)), giving the same linear system
structure as `update_V_all_nodes` but without the stocking FOC solve.

# Returns
Same NamedTuple format as `update_V_all_nodes`.
"""
function update_V_all_nodes_no_fallow(τ_star_coeffs, Vtilde_data, p; N=10)
    nodes = fourier_nodes(N)
    M = 2N + 1

    W = zeros(M, M)
    α_f = zeros(M)
    α_g = zeros(M)
    t0_values = copy(nodes)
    T_star_at_t0 = zeros(M)
    d_values = zeros(M)

    for (i, t) in enumerate(nodes)
        # d* = 0: restocking immediately at t₀ = t
        f_i = Vtilde_data.f_values[i]
        g_i = Vtilde_data.g_values[i]
        T_star = Vtilde_data.T_star_values[i]

        T_star_at_t0[i] = T_star
        α_f[i] = f_i       # disc_fallow = exp(0) = 1
        α_g[i] = g_i

        iw = spline_interp_weights(T_star, nodes)
        W[i, iw.idx_lo] += 1 - iw.weight
        W[i, iw.idx_hi] += iw.weight
    end

    A = Matrix{Float64}(I, M, M) - Diagonal(α_g) * W
    v = A \ α_f

    V_new_coeffs = make_spline(nodes, v)
    V_values = copy(v)

    # V = Ṽ when d* = 0
    Vtilde_values = copy(v)

    return (V_new_coeffs = V_new_coeffs, V_values = V_values,
            Vtilde_values = Vtilde_values, d_values = d_values,
            t0_values = t0_values, nodes = nodes)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. Ṽ(t₀) = J(T*, t₀, t₀) as a periodic linear spline
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=10)

Compute Ṽ(t₀) = J(T*(t₀), t₀, t₀) — the expected present utility of a
cycle stocked at t₀ with optimal harvest at T*(t₀) — at `2N+1` nodes
in t₀ space, then fit a periodic linear spline.

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
        τ_star = spline_eval(t₀, τ_star_coeffs)
        T_star = t₀ + τ_star
        push!(T_star_values, T_star)

        # Compute decomposition (f, g) — reuses the same ODE solutions
        decomp = compute_Vtilde_decomposed(t₀, T_star, p)
        push!(f_values, decomp.f)
        push!(g_values, decomp.g)

        # Full Ṽ = f + g·V(T*)
        V_T = spline_eval(T_star, V_coeffs)
        push!(Vtilde_values, decomp.f + decomp.g * V_T)
    end

    Vtilde_coeffs = make_spline(nodes, Vtilde_values)
    return (Vtilde_coeffs = Vtilde_coeffs, Vtilde_values = Vtilde_values,
            nodes = nodes, f_values = f_values, g_values = g_values,
            T_star_values = T_star_values)
end


# ──────────────────────────────────────────────────────────────────────────────
# 6. Convergence utilities
# ──────────────────────────────────────────────────────────────────────────────

"""
    damped_update(V_old_coeffs, V_new_coeffs; α=0.5)

Blend old and new spline nodal values: V = α·V_new + (1−α)·V_old.
Damping stabilises the fixed-point iteration.

# Arguments
- `α` : damping parameter in (0, 1]. α = 1 means no damping (full update).
"""
function damped_update(V_old_coeffs, V_new_coeffs; α=0.5)
    new_values = α .* V_new_coeffs.values .+ (1 - α) .* V_old_coeffs.values
    return make_spline(V_old_coeffs.nodes, new_values)
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

Solve the full seasonal aquaculture model in two phases.

# Algorithm

**Phase 1: Direct solve (f/g decomposition)**

Each iteration:
1. Given current `V_coeffs`, solve the harvest FOC at `2N+1` nodes to obtain
   `τ_star_coeffs` (spline for optimal cycle duration).
2. Compute Ṽ(t₀) at `2N+1` nodes and fit a spline (one `compute_Vtilde`
   call per node — the main expensive step).
3. Solve the stocking FOC at `2N+1` nodes using the Ṽ spline approximation
   to obtain `d*(t)` values (cheap — spline arithmetic only).
4. For each node, use the pre-computed `d*(t)` to decompose the cycle value
   and solve a linear system for V nodal values.
5. Apply damping: `V = α·V_new + (1−α)·V_old`.
6. Check convergence via the sup-norm on nodal value changes.

**Phase 2: Bellman fixed-point refinement**

Starting from the Phase 1 solution, iterate V via the full Bellman equation —
evaluating V(s) at every quadrature point in the loss integral rather than
using the f/g decomposition. Each Bellman iteration re-solves the harvest FOC
(τ*) and stocking FOC (d*) given the current V, then updates V at each node
via V(t) = e^{-δd*}·Ṽ(t₀*) where Ṽ is computed from the full objective.
This eliminates the approximation error from the direct solve and captures
within-cycle seasonality of V that the decomposition misses.

# Arguments
- `p`        : parameter set (must include seasonal Fourier coefficients for λ, m, k)
- `N`        : determines number of nodes (2N+1)
- `V_init`   : initial V(t) spline coefficients; if `nothing`, uses a constant
               estimated from the model parameters
- `max_iter` : maximum number of outer iterations
- `tol`      : convergence tolerance on the sup-norm of nodal value changes
- `damping`  : damping parameter α ∈ (0, 1]; smaller values are more conservative
- `d_max`          : maximum fallow duration to consider in stocking FOC (days)
- `force_no_fallow` : if `true`, force d*=0 at all nodes (immediate restocking)
- `verbose`        : if `true`, print iteration progress

# Returns
A NamedTuple with fields:
- `V_coeffs`            : converged spline for V(t) (nodes, values)
- `Vtilde_coeffs`       : converged spline for Ṽ(t₀) = J(T*,t₀,t₀)
- `τ_star_coeffs`       : converged spline for τ*(t₀)
- `V_values`            : V(t) at the nodes (last iteration)
- `Vtilde_at_t0_nodes`  : Ṽ(t₀) evaluated at uniform nodes in t₀ space
- `Vtilde_at_V_nodes`   : Ṽ(t₀*) at the nodes (from the value linkage step)
- `d_values`            : optimal fallow durations d*(t) at the nodes
- `t0_values`           : optimal stocking dates t₀*(t) at the nodes
- `τ_values`            : optimal cycle durations τ*(t₀) at the harvest nodes
- `nodes`               : node positions (days)
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
        force_no_fallow = false,
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
        V_mean = sum(V_init.values) / length(V_init.values)
        verbose && println("Initialized V(t) from user-supplied spline (mean = $(round(V_mean; digits=2)))")
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
        # between different FOC crossings as V evolves.
        verbose && print("  Iteration $k: harvest FOC...")
        harvest_result = solve_harvest_at_nodes(V_coeffs, p; N=N, τ_max=τ_max,
                                                τ_prev_coeffs=τ_star_coeffs)
        τ_star_coeffs = harvest_result.τ_star_coeffs
        τ_values = harvest_result.τ_values
        τ_mean = sum(τ_star_coeffs.values) / length(τ_star_coeffs.values)
        verbose && print(" τ̄=$(round(τ_mean; digits=1))")

        # ── Step 2: Compute Ṽ(t₀) spline ─────────────────────────────────────
        Vtilde_iter = compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=N)
        Vt_mean = sum(Vtilde_iter.Vtilde_coeffs.values) / length(Vtilde_iter.Vtilde_coeffs.values)
        verbose && print(" Ṽ̄=$(round(Vt_mean; digits=0))")

        # ── Step 3–4: Solve stocking FOC (cheap) then update V(t) ─────────
        if force_no_fallow
            V_result = update_V_all_nodes_no_fallow(τ_star_coeffs, Vtilde_iter, p; N=N)
        else
            V_result = update_V_all_nodes(τ_star_coeffs, Vtilde_iter.Vtilde_coeffs, Vtilde_iter, p; N=N, d_max=d_max)
        end

        # ── Damped V update ──────────────────────────────────────────────────
        V_new_coeffs = damped_update(V_coeffs, V_result.V_new_coeffs; α=damping)

        # ── Convergence check ────────────────────────────────────────────────
        Δ = maximum(abs.(V_new_coeffs.values .- V_coeffs.values))
        push!(history, (k, Δ))
        V_bar = sum(V_new_coeffs.values) / length(V_new_coeffs.values)
        verbose && println(" V̄=$(round(V_bar; digits=0)) ΔV=$(round(Δ; sigdigits=4))")

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

    direct_iter = iter
    direct_converged = converged

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Full Bellman fixed-point iterations
    # ══════════════════════════════════════════════════════════════════════════
    # The direct solve (f/g decomposition) approximates V(T*) as constant
    # within each cycle evaluation. The full Bellman evaluates V(s) at every
    # quadrature point in the loss integral, capturing seasonality that the
    # decomposition misses. We iterate with τ* and d* fixed from Phase 1.

    verbose && println("\n  Phase 2: Bellman fixed-point refinement...")

    bellman_history = Tuple{Int, Float64}[]
    bellman_converged = false
    bellman_iter = 0

    nodes = V_result.nodes
    d_values = V_result.d_values
    t0_values = V_result.t0_values
    Vtilde_bellman = zeros(length(nodes))
    M = length(nodes)

    for k in 1:max_iter
        bellman_iter = k

        # ── Re-solve harvest FOC (τ*) given current V ─────────────────────
        verbose && print("    Bellman $k: harvest FOC...")
        harvest_result = solve_harvest_at_nodes(V_coeffs, p; N=N, τ_max=τ_max,
                                                τ_prev_coeffs=τ_star_coeffs)
        τ_star_coeffs = harvest_result.τ_star_coeffs
        τ_values = harvest_result.τ_values
        τ_mean = sum(τ_star_coeffs.values) / length(τ_star_coeffs.values)
        verbose && print(" τ̄=$(round(τ_mean; digits=1))")

        # ── Re-solve stocking FOC (d*) ────────────────────────────────────
        if force_no_fallow
            d_values = zeros(M)
        else
            # Compute Ṽ spline for the stocking FOC evaluation
            Vtilde_iter = compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=N)
            stocking = solve_stocking_at_V_nodes(Vtilde_iter.Vtilde_coeffs, p; N=N, d_max=d_max)
            d_values = stocking.d_values
        end

        # ── Bellman V update: V(tᵢ) = e^{-δdᵢ}·Ṽ(t₀ᵢ) ──────────────────
        V_new_values = zeros(M)
        t0_values = zeros(M)

        for (i, t) in enumerate(nodes)
            d_star = d_values[i]
            t0_star = t + d_star
            t0_values[i] = t0_star
            τ_star = spline_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star

            Vt = compute_Vtilde(t0_star, T_star, V_coeffs, p)
            Vtilde_bellman[i] = Vt
            V_new_values[i] = exp(-p.δ * d_star) * Vt
        end

        V_new_coeffs = make_spline(nodes, V_new_values)
        V_new_coeffs = damped_update(V_coeffs, V_new_coeffs; α=damping)

        Δ = maximum(abs.(V_new_coeffs.values .- V_coeffs.values))
        push!(bellman_history, (k, Δ))
        V_bar = sum(V_new_coeffs.values) / length(V_new_coeffs.values)
        verbose && println(" V̄=$(round(V_bar; digits=0)) ΔV=$(round(Δ; sigdigits=4))")

        V_coeffs = V_new_coeffs

        if Δ < tol
            bellman_converged = true
            verbose && println("  Bellman converged after $k iterations (ΔV = $(round(Δ; sigdigits=4)) < $tol)")
            break
        end

        # Divergence detection
        if length(bellman_history) >= 4
            recent = [h[2] for h in bellman_history[end-2:end]]
            if recent[2] > recent[1] && recent[3] > recent[2]
                best_Δ = minimum(h[2] for h in bellman_history)
                bellman_converged = true
                verbose && println("  Bellman converged (divergence at iter $k; " *
                                  "best ΔV = $(round(best_Δ; sigdigits=4)))")
                break
            end
        end
    end

    if !bellman_converged && verbose
        println("  WARNING: Bellman did not converge after $max_iter iterations " *
                "(final ΔV = $(round(bellman_history[end][2]; sigdigits=4)))")
    end

    total_iter = direct_iter + bellman_iter
    converged = direct_converged && bellman_converged

    # ── Compute final Ṽ(t₀) spline ───────────────────────────────────────────
    verbose && print("  Computing Ṽ(t₀) spline...")
    Vtilde_result = compute_Vtilde_at_nodes(τ_star_coeffs, V_coeffs, p; N=N)
    Vt_mean = sum(Vtilde_result.Vtilde_coeffs.values) / length(Vtilde_result.Vtilde_coeffs.values)
    verbose && println(" Ṽ̄ = $(round(Vt_mean; digits=2))")

    # Combine histories
    full_history = vcat(history, [(direct_iter + k, Δ) for (k, Δ) in bellman_history])

    return (
        V_coeffs            = V_coeffs,
        Vtilde_coeffs       = Vtilde_result.Vtilde_coeffs,
        τ_star_coeffs       = τ_star_coeffs,
        V_values            = V_coeffs.values,
        Vtilde_at_t0_nodes  = Vtilde_result.Vtilde_values,
        Vtilde_at_V_nodes   = Vtilde_bellman,
        d_values            = d_values,
        t0_values           = t0_values,
        τ_values            = τ_values,
        nodes               = nodes,
        converged           = converged,
        iterations          = total_iter,
        history             = full_history,
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
- `V_grid`             : V(t) evaluated from spline
- `Vtilde_grid`        : Ṽ(t₀) = J(T*,t₀,t₀) evaluated from spline
- `τ_star_grid`        : τ*(t₀) evaluated from spline
- `d_grid`             : fallow duration d*(t) linearly interpolated from nodes
- `V_recomputed_grid`  : V(t) recomputed from interpolated d* and value linkage
- `Vtilde_linkage_grid`: Ṽ(t₀*) at the optimal stocking date for each end-of-cycle t
"""
function evaluate_solution(result, p; n_grid=200)
    t_grid = collect(range(0.0, PERIOD * (1 - 1/n_grid), length=n_grid))
    V_grid = [spline_eval(t, result.V_coeffs) for t in t_grid]
    Vtilde_grid = [spline_eval(t, result.Vtilde_coeffs) for t in t_grid]
    τ_star_grid = [spline_eval(t, result.τ_star_coeffs) for t in t_grid]

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
