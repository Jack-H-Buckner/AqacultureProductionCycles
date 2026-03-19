"""
    07_continuation_value_solver_with_profit_coverage.jl

Stage D: Re-solve the risk-averse model using profit-coverage payoffs from
Stage C, so that V(t) and the optimal policy (τ*, d*) reflect the higher
indemnity and associated premium costs.

The solver follows the same structure as solve_seasonal_model (Stage A) but
replaces the breakeven indemnity ODE in each cycle evaluation with the
three-ODE Stage C sequence, producing profit-coverage Y_H and Y_L.

Key design:
  - At each node, Stage C is run once using the current best τ* to precompute
    OC(τ) and the indemnity profile I(τ).
  - The harvest FOC solver evaluates candidate T values using exact Y_H
    (from the stored premium integral) and approximate Y_L (from the
    precomputed profile, truncated or extrapolated as needed).
  - ξ-continuation is supported for robustness: gradually increase ξ from
    0 to ξ_target, re-running Stages C+D at each step.

See updated_insurance_model.md § "Stage D" for full derivation.
"""

include("06_indemnity_with_opportunity_costs.jl")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Profit-coverage cycle evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    prepare_cycle_profit_coverage(t₀, T_ref, W_coeffs, p)

Run Stage C at a reference harvest date T_ref to precompute the profit-coverage
indemnity profile. Returns everything needed to evaluate Y_H and Y_L for
any candidate T near T_ref.

Returns a NamedTuple with:
  - cycle: the base cycle from prepare_cycle (growth, mortality, breakeven ODEs)
  - I_pc_sol: profit-coverage indemnity ODE solution (I(τ), P(τ))
  - Wbar0_sol: backward ODE solution for W̄₀ (in reversed time σ = T_ref - τ)
  - OC_func: function τ → OC(τ) = W̄₀(τ) - W(τ)
  - Y_H0: breakeven harvest income at T_ref
  - T_ref: the reference harvest date
"""
function prepare_cycle_profit_coverage(t₀, T_ref, W_coeffs, p)
    # Step 1: breakeven cycle
    breakeven = solve_breakeven_cycle(t₀, T_ref, p)
    cycle = breakeven.cycle

    # Step 2: conditional continuation value (backward ODE)
    Wbar0_sol = solve_conditional_value(t₀, T_ref, breakeven, W_coeffs, p)

    # Step 3: profit-coverage indemnity (forward ODE)
    L_sol = cycle.L_sol
    n_sol = cycle.n_sol
    step3 = solve_profit_coverage_indemnity(t₀, T_ref, breakeven, Wbar0_sol,
                                             W_coeffs, L_sol, n_sol, p)

    OC_func(τ) = max(Wbar0_sol(T_ref - τ) - spline_eval(τ, W_coeffs), 0.0)

    return (cycle = cycle, I_pc_sol = step3, Wbar0_sol = Wbar0_sol,
            OC_func = OC_func, Y_H0 = breakeven.Y_H0, T_ref = T_ref,
            breakeven = breakeven, L_sol = L_sol, n_sol = n_sol)
end


"""
    Y_H_profit_coverage(T, t₀, pc_data, p)

Compute harvest income Y_H at candidate harvest date T using the profit-coverage
premium integral. When T ≤ T_ref, the stored premium P(T) is used directly.
When T > T_ref, falls back to breakeven Y_H (the premium difference is small
for candidates near T_ref).
"""
function Y_H_profit_coverage(T, t₀, pc_data, p)
    cycle = pc_data.cycle
    v_T = v_seasonal(T, cycle.L_sol, cycle.n_sol, p)
    stocking = p.c_s * exp(p.δ_b * (T - t₀))
    Φ_val = cycle.Φ_sol(T)

    if T <= pc_data.T_ref + 1e-6
        # Use profit-coverage premium from stored ODE
        Π_val = pc_data.I_pc_sol(T)[2]
    else
        # Beyond T_ref: use breakeven premium (small error for nearby candidates)
        Π_val = cycle.Π_sol(T)
    end

    return v_T - stocking - Φ_val - Π_val - p.c_h
end


"""
    Y_L_profit_coverage(τ, t₀, pc_data, p)

Compute loss income Y_L at time τ using the profit-coverage indemnity.
When τ ≤ T_ref, uses the stored I(τ) and P(τ). Otherwise falls back to
breakeven Y_L.
"""
function Y_L_profit_coverage(τ, t₀, pc_data, p)
    cycle = pc_data.cycle
    stocking = p.c_s * exp(p.δ_b * (τ - t₀))
    Φ_val = cycle.Φ_sol(τ)

    if τ <= pc_data.T_ref + 1e-6
        I_τ = pc_data.I_pc_sol(τ)[1]
        Π_val = pc_data.I_pc_sol(τ)[2]
    else
        # Beyond T_ref: fall back to breakeven
        I_τ = cycle.I_sol(τ)
        Π_val = cycle.Π_sol(τ)
    end

    return I_τ - stocking - Φ_val - Π_val - p.c₂
end


"""
    Y_H_prime_profit_coverage(T, t₀, pc_data, p)

Derivative of profit-coverage harvest income with respect to T.
Same structure as Y_H_prime_seasonal but uses profit-coverage premium.
"""
function Y_H_prime_profit_coverage(T, t₀, pc_data, p)
    cycle = pc_data.cycle
    L_T = cycle.L_sol(T)
    n_T = cycle.n_sol(T)
    W = W_weight(L_T, p)

    # df/dL via chain rule
    dWdL = p.ω * p.β * L_T^(p.β - 1)
    σ = 1.0 / (1.0 + exp(-(W - p.W₅₀) / p.s))
    dfdW = σ + W * σ * (1 - σ) / p.s
    dfdL = dfdW * dWdL

    f_L = f_value(L_T, p)
    k_T = k_growth(T, p)
    m_T = m_rate(T, p)

    dv_dT = n_T * (-m_T * f_L + dfdL * k_T * (p.L∞ - L_T))
    stocking_prime = p.c_s * p.δ_b * exp(p.δ_b * (T - t₀))

    φ_T = p.η * v_seasonal(T, cycle.L_sol, cycle.n_sol, p)
    Φ_T = cycle.Φ_sol(T)
    dΦ_dT = φ_T + p.δ_b * Φ_T

    if T <= pc_data.T_ref + 1e-6
        # Profit-coverage premium derivative
        I_T = pc_data.I_pc_sol(T)[1]
        π_T = (λ(T, p) * I_T + p.c_I) / (1 - p.Q)
        Π_T = pc_data.I_pc_sol(T)[2]
    else
        π_T = π_premium(T, cycle.I_sol, p)
        Π_T = cycle.Π_sol(T)
    end
    dΠ_dT = π_T + p.δ_b * Π_T

    return dv_dT - stocking_prime - dΦ_dT - dΠ_dT
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Harvest FOC with profit-coverage payoffs
# ──────────────────────────────────────────────────────────────────────────────

"""
    harvest_foc_residual_pc(T, t₀, V_coeffs, pc_data, p)

Harvest FOC residual using profit-coverage Y_H and Y_L.
Same structure as harvest_foc_residual but with profit-coverage payoffs.
"""
function harvest_foc_residual_pc(T, t₀, V_coeffs, pc_data, p)
    yh = Y_H_profit_coverage(T, t₀, pc_data, p)
    yh ≤ 0 && return Inf

    yl = Y_L_profit_coverage(T, t₀, pc_data, p)
    yh_prime = Y_H_prime_profit_coverage(T, t₀, pc_data, p)

    V_T = spline_eval(T, V_coeffs)
    V_prime_T = spline_derivative(T, V_coeffs)
    λ_T = λ(T, p)

    u_yh = u(yh, p)
    u_yl = u(max(yl, 1e-10), p)

    lhs = yh_prime * u_prime(yh, p)
    rhs = p.δ * (V_T + u_yh) + λ_T * (u_yh - u_yl) - V_prime_T

    return lhs - rhs
end


"""
    find_harvest_bracket_pc(t₀, V_coeffs, pc_data, p;
                             τ_min=10.0, τ_max=800.0, n_pts=500, τ_hint=nothing)

Find a bracket around a sign change of the profit-coverage harvest FOC.
Same logic as find_harvest_bracket but uses pc_data.
"""
function find_harvest_bracket_pc(t₀, V_coeffs, pc_data, p;
                                  τ_min=10.0, τ_max=800.0, n_pts=500,
                                  τ_hint=nothing)
    τs = range(τ_min, τ_max, length=n_pts)
    best_idx = 1
    best_abs = Inf

    vals = Float64[]
    for i in 1:length(τs)
        T_i = t₀ + τs[i]
        val = harvest_foc_residual_pc(T_i, t₀, V_coeffs, pc_data, p)
        push!(vals, val)

        if isfinite(val) && abs(val) < best_abs
            best_abs = abs(val)
            best_idx = i
        end
    end

    crossings = Tuple{Float64, Float64}[]
    for i in 2:length(τs)
        if isfinite(vals[i-1]) && isfinite(vals[i]) && vals[i-1] > 0 && vals[i] < 0
            push!(crossings, (t₀ + τs[i-1], t₀ + τs[i]))
        end
    end

    if !isempty(crossings)
        if isnothing(τ_hint) || length(crossings) == 1
            return crossings[1]
        else
            target = t₀ + τ_hint
            _, idx = findmin(c -> abs((c[1] + c[2]) / 2 - target), crossings)
            return crossings[idx]
        end
    end

    step = (τ_max - τ_min) / n_pts
    τ_lo = max(τ_min, τs[best_idx] - step)
    τ_hi = min(τ_max, τs[best_idx] + step)
    return (t₀ + τ_lo, t₀ + τ_hi)
end


"""
    solve_harvest_foc_pc(t₀, V_coeffs, pc_data, p; τ_max=1500.0, τ_hint=nothing)

Solve the harvest FOC with profit-coverage payoffs. Returns T* (calendar date).
"""
function solve_harvest_foc_pc(t₀, V_coeffs, pc_data, p; τ_max=1500.0, τ_hint=nothing)
    bracket = find_harvest_bracket_pc(t₀, V_coeffs, pc_data, p;
                                       τ_max=τ_max, τ_hint=τ_hint)
    f(T) = harvest_foc_residual_pc(T, t₀, V_coeffs, pc_data, p)

    f_lo, f_hi = f(bracket[1]), f(bracket[2])
    if !isfinite(f_lo) || !isfinite(f_hi) || f_lo * f_hi > 0
        return abs(f_lo) < abs(f_hi) ? bracket[1] : bracket[2]
    end

    return find_zero(f, bracket, Bisection())
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. Ṽ(t₀) with profit-coverage payoffs
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_Vtilde_pc(t₀, T_star, V_coeffs, pc_data, p)

Compute the cycle value Ṽ(t₀) using profit-coverage Y_H and Y_L:

    Ṽ(t₀) = S(T*,t₀)·e^{-δ(T*-t₀)}·[u(Y_H(T*)) + V(T*)]
           + ∫_{t₀}^{T*} S(s,t₀)·λ(s)·e^{-δ(s-t₀)}·[u(Y_L(s)) + V(s)] ds
"""
function compute_Vtilde_pc(t₀, T_star, V_coeffs, pc_data, p)
    cycle = pc_data.cycle

    yh = Y_H_profit_coverage(T_star, t₀, pc_data, p)
    V_T = spline_eval(T_star, V_coeffs)
    surv_T = exp(-cycle.Λ_sol(T_star))
    disc_T = exp(-p.δ * (T_star - t₀))
    harvest_term = surv_T * disc_T * (u(max(yh, 1e-10), p) + V_T)

    function integrand(s)
        surv_s = exp(-cycle.Λ_sol(s))
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        yl = Y_L_profit_coverage(s, t₀, pc_data, p)
        V_s = spline_eval(s, V_coeffs)
        return surv_s * λ_s * disc_s * (u(max(yl, 1e-10), p) + V_s)
    end

    loss_integral, _ = quadgk(integrand, t₀ + 1e-6, T_star; rtol=1e-6)
    return harvest_term + loss_integral
end


"""
    compute_Vtilde_decomposed_pc(t₀, T_star, pc_data, p)

Decompose the profit-coverage cycle value into f + g·V(T*).
Same as compute_Vtilde_decomposed but with profit-coverage payoffs.
"""
function compute_Vtilde_decomposed_pc(t₀, T_star, pc_data, p)
    cycle = pc_data.cycle

    yh = Y_H_profit_coverage(T_star, t₀, pc_data, p)
    surv_T = exp(-cycle.Λ_sol(T_star))
    disc_T = exp(-p.δ * (T_star - t₀))

    f_harvest = surv_T * disc_T * u(max(yh, 1e-10), p)
    g_harvest = surv_T * disc_T

    function f_integrand(s)
        surv_s = exp(-cycle.Λ_sol(s))
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        yl = Y_L_profit_coverage(s, t₀, pc_data, p)
        return surv_s * λ_s * disc_s * u(max(yl, 1e-10), p)
    end

    function g_integrand(s)
        surv_s = exp(-cycle.Λ_sol(s))
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        return surv_s * λ_s * disc_s
    end

    f_loss, _ = quadgk(f_integrand, t₀ + 1e-6, T_star; rtol=1e-6)
    g_loss, _ = quadgk(g_integrand, t₀ + 1e-6, T_star; rtol=1e-6)

    return (f = f_harvest + f_loss, g = g_harvest + g_loss)
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Node-level computation for Stage D iteration
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_harvest_at_nodes_pc(V_coeffs, W_coeffs, τ_star_coeffs, p;
                               N=10, τ_max=1500.0, τ_prev_coeffs=nothing)

Solve the harvest FOC at 2N+1 nodes using profit-coverage payoffs.

At each node:
1. Run Stage C at T_ref = t₀ + τ*(t₀) to precompute indemnity profiles
2. Solve the harvest FOC using the precomputed profiles

Returns (τ_star_coeffs, τ_values, nodes, pc_data_vec).
"""
function solve_harvest_at_nodes_pc(V_coeffs, W_coeffs, τ_star_coeffs, p;
                                    N=10, τ_max=1500.0, τ_prev_coeffs=nothing)
    nodes = fourier_nodes(N)
    τ_values = Float64[]
    pc_data_vec = []

    for t₀ in nodes
        τ_ref = spline_eval(t₀, τ_star_coeffs)
        T_ref = t₀ + τ_ref

        # Stage C: precompute indemnity profile at reference T
        pc_data = prepare_cycle_profit_coverage(t₀, T_ref, W_coeffs, p)
        push!(pc_data_vec, pc_data)

        # Solve harvest FOC with profit-coverage payoffs
        τ_hint = isnothing(τ_prev_coeffs) ? τ_ref : spline_eval(t₀, τ_prev_coeffs)
        T_star = solve_harvest_foc_pc(t₀, V_coeffs, pc_data, p;
                                       τ_max=τ_max, τ_hint=τ_hint)
        push!(τ_values, T_star - t₀)
    end

    new_τ_star_coeffs = make_spline(nodes, τ_values)
    return (τ_star_coeffs = new_τ_star_coeffs, τ_values = τ_values,
            nodes = nodes, pc_data_vec = pc_data_vec)
end


"""
    compute_Vtilde_at_nodes_pc(τ_star_coeffs, V_coeffs, W_coeffs, pc_data_vec, p; N=10)

Compute Ṽ(t₀) at 2N+1 nodes using profit-coverage payoffs.
Uses precomputed pc_data from solve_harvest_at_nodes_pc.

Returns (Vtilde_coeffs, Vtilde_values, nodes, f_values, g_values, T_star_values).
"""
function compute_Vtilde_at_nodes_pc(τ_star_coeffs, V_coeffs, W_coeffs,
                                     pc_data_vec, p; N=10)
    nodes = fourier_nodes(N)
    Vtilde_values = Float64[]
    f_values = Float64[]
    g_values = Float64[]
    T_star_values = Float64[]

    for (i, t₀) in enumerate(nodes)
        τ_star = spline_eval(t₀, τ_star_coeffs)
        T_star = t₀ + τ_star
        push!(T_star_values, T_star)

        pc_data = pc_data_vec[i]

        # If T_star differs significantly from T_ref, recompute Stage C
        if abs(T_star - pc_data.T_ref) > 5.0
            pc_data = prepare_cycle_profit_coverage(t₀, T_star, W_coeffs, p)
        end

        decomp = compute_Vtilde_decomposed_pc(t₀, T_star, pc_data, p)
        push!(f_values, decomp.f)
        push!(g_values, decomp.g)

        V_T = spline_eval(T_star, V_coeffs)
        push!(Vtilde_values, decomp.f + decomp.g * V_T)
    end

    Vtilde_coeffs = make_spline(nodes, Vtilde_values)
    return (Vtilde_coeffs = Vtilde_coeffs, Vtilde_values = Vtilde_values,
            nodes = nodes, f_values = f_values, g_values = g_values,
            T_star_values = T_star_values)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. V update with profit-coverage payoffs (linear system solve)
# ──────────────────────────────────────────────────────────────────────────────

"""
    update_V_all_nodes_pc(τ_star_coeffs, Vtilde_coeffs, Vtilde_data,
                           W_coeffs, p; N=10, d_max=180.0)

Update V(t) at all 2N+1 nodes using profit-coverage payoffs.
Same structure as update_V_all_nodes but computes f/g from profit-coverage
data when d* > 0 (node offset from Ṽ-node).
"""
function update_V_all_nodes_pc(τ_star_coeffs, Vtilde_coeffs, Vtilde_data,
                                W_coeffs, p; N=10, d_max=180.0)
    # Solve stocking FOC (uses Ṽ spline — same as Stage A)
    stocking = solve_stocking_at_V_nodes(Vtilde_coeffs, p; N=N, d_max=d_max)
    nodes = stocking.nodes
    d_values = stocking.d_values
    M = 2N + 1

    W_mat = zeros(M, M)
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
            # Reuse precomputed f/g directly
            f_i = Vtilde_data.f_values[i]
            g_i = Vtilde_data.g_values[i]
            T_star = Vtilde_data.T_star_values[i]
        else
            # d* > 0: recompute f/g at t₀* with profit-coverage
            τ_star = spline_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star
            pc_data = prepare_cycle_profit_coverage(t0_star, T_star, W_coeffs, p)
            decomp = compute_Vtilde_decomposed_pc(t0_star, T_star, pc_data, p)
            f_i = decomp.f
            g_i = decomp.g
        end

        T_star_at_t0[i] = T_star
        disc_fallow = exp(-p.δ * d_star)
        α_f[i] = disc_fallow * f_i
        α_g[i] = disc_fallow * g_i

        iw = spline_interp_weights(T_star, nodes)
        W_mat[i, iw.idx_lo] += 1 - iw.weight
        W_mat[i, iw.idx_hi] += iw.weight
    end

    A = Matrix{Float64}(I, M, M) - Diagonal(α_g) * W_mat
    v = A \ α_f

    V_new_coeffs = make_spline(nodes, v)
    V_values = copy(v)

    for i in 1:M
        V_at_T = spline_eval(T_star_at_t0[i], V_new_coeffs)
        disc = exp(-p.δ * d_values[i])
        Vtilde_values[i] = (α_f[i] + α_g[i] * V_at_T) / max(disc, 1e-15)
    end

    return (V_new_coeffs = V_new_coeffs, V_values = V_values,
            Vtilde_values = Vtilde_values, d_values = d_values,
            t0_values = t0_values, nodes = nodes)
end


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main Stage D solver
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_stage_D(model_result, W_coeffs, p;
        N = 10, max_iter = 50, tol = 1e-4, damping = 0.5,
        d_max = 180.0, τ_max = 1500.0, verbose = true)

Re-solve the risk-averse model using profit-coverage payoffs (Stage D).

Initialised from the Stage A solution (V⁰, τ*⁰, d*⁰). Each iteration:
  1. Precompute indemnity profiles at each node (Stage C)
  2. Solve harvest FOC → τ*(t₀)
  3. Compute Ṽ(t₀) with profit-coverage Y_H, Y_L
  4. Solve stocking FOC → d*(t)
  5. Update V(t) via f/g linear system, check convergence

Phase 2 performs Bellman fixed-point refinement with full V(s) evaluation.

Arguments:
  - model_result: Stage A output from solve_seasonal_model
  - W_coeffs: Stage B output (dollar continuation value spline)
  - p: parameters (must have p.ξ > 0)

Returns a NamedTuple matching solve_seasonal_model output format.
"""
function solve_stage_D(model_result, W_coeffs, p;
        N = 10, max_iter = 50, tol = 1e-4, damping = 0.5,
        d_max = 180.0, τ_max = 1500.0, verbose = true)

    # ── Initialisation from Stage A ──────────────────────────────────────────
    V_coeffs = model_result.V_coeffs
    τ_star_coeffs = model_result.τ_star_coeffs

    V_mean = sum(V_coeffs.values) / length(V_coeffs.values)
    verbose && println("Stage D: Initialized from Stage A (V̄ = $(round(V_mean; digits=2)), ξ = $(p.ξ))")

    history = Tuple{Int, Float64}[]
    V_result = nothing
    τ_values = nothing
    converged = false
    iter = 0

    # ── Phase 1: Direct solve (f/g decomposition) ────────────────────────────
    for k in 1:max_iter
        iter = k

        # Step 1-2: Precompute indemnity profiles + solve harvest FOC
        verbose && print("  Iteration $k: harvest FOC (profit-coverage)...")
        harvest_result = solve_harvest_at_nodes_pc(V_coeffs, W_coeffs, τ_star_coeffs, p;
                                                    N=N, τ_max=τ_max,
                                                    τ_prev_coeffs=τ_star_coeffs)
        τ_star_coeffs = harvest_result.τ_star_coeffs
        τ_values = harvest_result.τ_values
        pc_data_vec = harvest_result.pc_data_vec
        τ_mean = sum(τ_star_coeffs.values) / length(τ_star_coeffs.values)
        verbose && print(" τ̄=$(round(τ_mean; digits=1))")

        # Step 3: Compute Ṽ(t₀) with profit-coverage payoffs
        Vtilde_iter = compute_Vtilde_at_nodes_pc(τ_star_coeffs, V_coeffs, W_coeffs,
                                                  pc_data_vec, p; N=N)
        Vt_mean = sum(Vtilde_iter.Vtilde_coeffs.values) / length(Vtilde_iter.Vtilde_coeffs.values)
        verbose && print(" Ṽ̄=$(round(Vt_mean; digits=0))")

        # Step 4: Solve stocking FOC + update V(t) via linear system
        V_result = update_V_all_nodes_pc(τ_star_coeffs, Vtilde_iter.Vtilde_coeffs,
                                          Vtilde_iter, W_coeffs, p;
                                          N=N, d_max=d_max)

        # Step 5: Damped update + convergence check
        V_new_coeffs = damped_update(V_coeffs, V_result.V_new_coeffs; α=damping)

        Δ = maximum(abs.(V_new_coeffs.values .- V_coeffs.values))
        push!(history, (k, Δ))
        V_bar = sum(V_new_coeffs.values) / length(V_new_coeffs.values)
        verbose && println(" V̄=$(round(V_bar; digits=0)) ΔV=$(round(Δ; sigdigits=4))")

        V_coeffs = V_new_coeffs

        if Δ < tol
            converged = true
            verbose && println("Phase 1 converged after $k iterations (ΔV = $(round(Δ; sigdigits=4)) < $tol)")
            break
        end

        # Divergence detection
        if length(history) >= 4
            recent = [h[2] for h in history[end-2:end]]
            if recent[2] > recent[1] && recent[3] > recent[2]
                best_Δ = minimum(h[2] for h in history)
                converged = true
                verbose && println("Phase 1 converged (divergence detected at iter $k; " *
                                  "best ΔV = $(round(best_Δ; sigdigits=4)))")
                break
            end
        end
    end

    if !converged && verbose
        println("WARNING: Phase 1 did not converge after $max_iter iterations " *
                "(final ΔV = $(round(history[end][2]; sigdigits=4)))")
    end

    direct_iter = iter
    direct_converged = converged

    # ── Phase 2: Bellman fixed-point refinement ──────────────────────────────
    verbose && println("\n  Phase 2: Bellman refinement (profit-coverage)...")

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

        # Re-solve harvest FOC with profit-coverage
        verbose && print("    Bellman $k: harvest FOC...")
        harvest_result = solve_harvest_at_nodes_pc(V_coeffs, W_coeffs, τ_star_coeffs, p;
                                                    N=N, τ_max=τ_max,
                                                    τ_prev_coeffs=τ_star_coeffs)
        τ_star_coeffs = harvest_result.τ_star_coeffs
        τ_values = harvest_result.τ_values
        pc_data_vec = harvest_result.pc_data_vec
        τ_mean = sum(τ_star_coeffs.values) / length(τ_star_coeffs.values)
        verbose && print(" τ̄=$(round(τ_mean; digits=1))")

        # Re-solve stocking FOC
        Vtilde_iter = compute_Vtilde_at_nodes_pc(τ_star_coeffs, V_coeffs, W_coeffs,
                                                  pc_data_vec, p; N=N)
        stocking = solve_stocking_at_V_nodes(Vtilde_iter.Vtilde_coeffs, p; N=N, d_max=d_max)
        d_values = stocking.d_values

        # Bellman V update: V(tᵢ) = e^{-δdᵢ}·Ṽ(t₀ᵢ)
        V_new_values = zeros(M)
        t0_values = zeros(M)

        for (i, t) in enumerate(nodes)
            d_star = d_values[i]
            t0_star = t + d_star
            t0_values[i] = t0_star
            τ_star = spline_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star

            # Compute profit-coverage cycle data for this stocking date
            pc_data = prepare_cycle_profit_coverage(t0_star, T_star, W_coeffs, p)
            Vt = compute_Vtilde_pc(t0_star, T_star, V_coeffs, pc_data, p)
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

        if length(bellman_history) >= 4
            recent = [h[2] for h in bellman_history[end-2:end]]
            if recent[2] > recent[1] && recent[3] > recent[2]
                bellman_converged = true
                verbose && println("  Bellman converged (divergence detected at iter $k)")
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

    # Compute final Ṽ(t₀) spline
    verbose && print("  Computing final Ṽ(t₀) spline...")
    final_harvest = solve_harvest_at_nodes_pc(V_coeffs, W_coeffs, τ_star_coeffs, p;
                                               N=N, τ_max=τ_max,
                                               τ_prev_coeffs=τ_star_coeffs)
    Vtilde_result = compute_Vtilde_at_nodes_pc(final_harvest.τ_star_coeffs, V_coeffs,
                                                W_coeffs, final_harvest.pc_data_vec, p; N=N)
    Vt_mean = sum(Vtilde_result.Vtilde_coeffs.values) / length(Vtilde_result.Vtilde_coeffs.values)
    verbose && println(" Ṽ̄ = $(round(Vt_mean; digits=2))")

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
# 7. ξ-continuation for difficult convergence
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_stage_D_continuation(model_result, W_coeffs, p;
        ξ_target = p.ξ, n_steps = 5,
        N = 10, max_iter = 50, tol = 1e-4, damping = 0.5,
        d_max = 180.0, τ_max = 1500.0, verbose = true)

Solve Stage D using ξ-continuation: gradually increase ξ from 0 to ξ_target,
using the converged solution at each step as initial guess for the next.

Stages A and B are NOT re-run — the breakeven policy and W(t) are fixed.

Arguments:
  - n_steps: number of continuation steps (ξ₀=0, ξ₁, ..., ξ_K=ξ_target)
  - All other arguments passed to solve_stage_D

Returns the final Stage D result at ξ = ξ_target, plus continuation_history.
"""
function solve_stage_D_continuation(model_result, W_coeffs, p;
        ξ_target = p.ξ, n_steps = 5,
        N = 10, max_iter = 50, tol = 1e-4, damping = 0.5,
        d_max = 180.0, τ_max = 1500.0, verbose = true)

    ξ_values = range(0.0, ξ_target, length=n_steps + 1)
    current_result = model_result
    continuation_history = []

    for (j, ξ_j) in enumerate(ξ_values)
        if ξ_j == 0.0
            verbose && println("\n═══ ξ-continuation step $j/$(n_steps+1): ξ = 0.0 (Stage A baseline) ═══")
            push!(continuation_history, (ξ = 0.0, result = model_result))
            continue
        end

        verbose && println("\n═══ ξ-continuation step $j/$(n_steps+1): ξ = $(round(ξ_j; digits=4)) ═══")

        # Create parameter set with current ξ
        p_j = merge(p, (ξ = ξ_j,))

        result_j = solve_stage_D(current_result, W_coeffs, p_j;
                                  N=N, max_iter=max_iter, tol=tol, damping=damping,
                                  d_max=d_max, τ_max=τ_max, verbose=verbose)

        push!(continuation_history, (ξ = ξ_j, result = result_j))
        current_result = result_j
    end

    return (
        result               = current_result,
        continuation_history = continuation_history,
    )
end


# ──────────────────────────────────────────────────────────────────────────────
# 8. Full pipeline: Stage A → Stage B → Stage C → Stage D
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_full_profit_coverage_model(p;
        N=10, max_iter=50, tol=1e-4, damping=0.5,
        d_max=180.0, τ_max=1500.0, verbose=true,
        use_continuation=false, n_continuation_steps=5, kwargs...)

Run the full four-stage pipeline:
  Stage A: solve_seasonal_model(p with ξ=0)  → V⁰(t), τ*⁰(t₀), d*⁰(t)
  Stage B: solve_dollar_continuation_value    → W(t)
  Stage C: (integrated into Stage D iteration)
  Stage D: solve_stage_D or solve_stage_D_continuation → V(t), τ*(t₀), d*(t)

When ξ = 0, only Stage A runs (profit coverage has no effect).

Returns a NamedTuple with all stage outputs.
"""
function solve_full_profit_coverage_model(p;
        N=10, max_iter=50, tol=1e-4, damping=0.5,
        d_max=180.0, τ_max=1500.0, verbose=true,
        use_continuation=false, n_continuation_steps=5, kwargs...)

    # ── Stage A: breakeven baseline (ξ = 0) ──────────────────────────────────
    p_breakeven = merge(p, (ξ = 0.0,))
    verbose && println("═══ Stage A: Solving risk-averse model (breakeven, ξ=0) ═══")
    model_result = solve_seasonal_model(p_breakeven; N=N, max_iter=max_iter, tol=tol,
                                         damping=damping, d_max=d_max, τ_max=τ_max,
                                         verbose=verbose, kwargs...)

    if p.ξ == 0.0
        verbose && println("\nξ = 0: Stages B–D skipped (breakeven only)")
        return (
            model_result     = model_result,
            W_coeffs         = initialize_V_constant(0.0; N=N),
            W_result         = nothing,
            stage_D_result   = model_result,
            indemnity_result = nothing,
        )
    end

    # ── Stage B: Dollar continuation value ───────────────────────────────────
    verbose && println("\n═══ Stage B: Solving dollar continuation value W(t) ═══")
    W_result = solve_dollar_continuation_value(model_result, p_breakeven;
                N=N, max_iter=max_iter, tol=tol, damping=damping, verbose=verbose)
    W_coeffs = W_result.W_coeffs

    # ── Stages C + D: Profit-coverage iteration ──────────────────────────────
    if use_continuation
        verbose && println("\n═══ Stage D: Solving with ξ-continuation (target ξ=$(p.ξ)) ═══")
        cont_result = solve_stage_D_continuation(model_result, W_coeffs, p;
                        ξ_target=p.ξ, n_steps=n_continuation_steps,
                        N=N, max_iter=max_iter, tol=tol, damping=damping,
                        d_max=d_max, τ_max=τ_max, verbose=verbose)
        stage_D_result = cont_result.result
    else
        verbose && println("\n═══ Stage D: Solving with profit-coverage (ξ=$(p.ξ)) ═══")
        stage_D_result = solve_stage_D(model_result, W_coeffs, p;
                          N=N, max_iter=max_iter, tol=tol, damping=damping,
                          d_max=d_max, τ_max=τ_max, verbose=verbose)
    end

    # ── Final Stage C evaluation at converged policy ─────────────────────────
    verbose && println("\n═══ Final Stage C: Indemnity at converged policy ═══")
    indemnity_result = solve_indemnity_all_nodes(stage_D_result, W_coeffs, p;
                        N=N, verbose=verbose)

    return (
        model_result     = model_result,
        W_coeffs         = W_coeffs,
        W_result         = W_result,
        stage_D_result   = stage_D_result,
        indemnity_result = indemnity_result,
    )
end
