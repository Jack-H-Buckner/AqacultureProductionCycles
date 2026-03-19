"""
    05_dollar_continuation_value.jl

Compute the dollar continuation value W(t) — the expected NPV in dollars of the
production system at calendar time t, assuming the firm follows the risk-averse
policy from Stage A and holds breakeven insurance.

W(t) satisfies the same functional equation as V(t) but with CRRA utility u(·)
replaced by the identity. Cash flows Y_H⁰ and Y_L⁰ are evaluated under
breakeven insurance (the existing solve_indemnity ODE), making W independent
of the coverage fraction ξ.

    W̃(t₀) = S(T*,t₀)·e^{-δ(T*-t₀)}·[Y_H⁰(T*) + W(T*)]
           + ∫_{t₀}^{T*} S(s,t₀)·λ(s)·e^{-δ(s-t₀)}·[Y_L⁰(s) + W(s)] ds

    W(t) = e^{-δ·d*(t)}·W̃(t₀*(t))

The algorithm mirrors solve_seasonal_model with two modifications:
  (i)  u(Y_H) → Y_H and u(Y_L) → Y_L everywhere
  (ii) policy (τ*, d*) is fixed from Stage A — no FOC re-solving

The breakeven cash flows come from prepare_cycle(), which internally calls
solve_indemnity() — the breakeven indemnity ODE with I₀(t₀) = c_s + c₂.

See updated_insurance_model.md § "Stage B" for full derivation.
"""

include("03_continuation_value_solver.jl")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Initialization
# ──────────────────────────────────────────────────────────────────────────────

"""
    estimate_initial_W(p; τ_candidates=100.0:50.0:800.0)

Estimate a constant initial dollar continuation value W₀ by scanning over
candidate cycle lengths. Same as estimate_initial_V but using dollar income
(identity) instead of utility:

    W₀ ≈ max_τ { S·e^{-δτ}·Y_H(τ) / (1 - S·e^{-δτ}) }
"""
function estimate_initial_W(p; τ_candidates=100.0:50.0:800.0)
    t₀ = 0.0
    λ_mean = quadgk(s -> λ(s, p), 0.0, PERIOD)[1] / PERIOD
    δλ = p.δ + λ_mean

    best_W = 0.0

    for τ in τ_candidates
        cycle = prepare_cycle(t₀, τ, p)
        yh = Y_H_seasonal(τ, t₀, cycle, p)
        yh ≤ 0 && continue

        S_T = exp(-δλ * τ)
        denom = 1 - S_T
        denom < 1e-10 && continue

        W_est = S_T * yh / denom    # Y_H directly, not u(Y_H)
        if W_est > best_W
            best_W = W_est
        end
    end

    best_W > 0 && return best_W
    return max(p.c_s, 1.0)
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dollar-valued Ṽ decomposition (u → identity)
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_Wtilde_decomposed(t₀, T_star, p)

Decompose the dollar-valued cycle value into:
  f_W = S·e^{-δτ}·Y_H + ∫ S·λ·e^{-δ·…}·Y_L ds      (dollar-only component)
  g   = S·e^{-δτ}      + ∫ S·λ·e^{-δ·…} ds            (discount factor — unchanged)

so that W̃(t₀) = f_W + g·W(T*).

The g component is identical to the existing compute_Vtilde_decomposed since
it captures only the probability-weighted discount factor and does not involve
the utility function.
"""
function compute_Wtilde_decomposed(t₀, T_star, p)
    cycle = prepare_cycle(t₀, T_star, p)

    # Harvest branch
    yh = Y_H_seasonal(T_star, t₀, cycle, p)
    surv_T = exp(-cycle.Λ_sol(T_star))
    disc_T = exp(-p.δ * (T_star - t₀))

    f_harvest = surv_T * disc_T * max(yh, 1e-10)   # Y_H directly (not u(Y_H))
    g_harvest = surv_T * disc_T                      # unchanged

    # Loss branch: Y_L directly (not u(Y_L))
    function f_integrand(s)
        surv_s = exp(-cycle.Λ_sol(s))
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        yl = Y_L_seasonal(s, t₀, cycle, p)
        return surv_s * λ_s * disc_s * max(yl, 1e-10)   # Y_L, not u(Y_L)
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


"""
    compute_Wtilde(t₀, T_star, W_coeffs, p)

Compute the full dollar-valued cycle value W̃(t₀) using the current W spline:

    W̃(t₀) = S(T*,t₀)·e^{-δ(T*-t₀)}·[Y_H(T*) + W(T*)]
           + ∫_{t₀}^{T*} S(s,t₀)·λ(s)·e^{-δ(s-t₀)}·[Y_L(s) + W(s)] ds

Same as compute_Vtilde but with u(Y) → Y.
"""
function compute_Wtilde(t₀, T_star, W_coeffs, p)
    cycle = prepare_cycle(t₀, T_star, p)

    # Harvest branch
    yh = Y_H_seasonal(T_star, t₀, cycle, p)
    W_T = spline_eval(T_star, W_coeffs)
    surv_T = exp(-cycle.Λ_sol(T_star))
    disc_T = exp(-p.δ * (T_star - t₀))
    harvest_term = surv_T * disc_T * (max(yh, 1e-10) + W_T)

    # Loss branch integral
    function integrand(s)
        surv_s = exp(-cycle.Λ_sol(s))
        λ_s = λ(s, p)
        disc_s = exp(-p.δ * (s - t₀))
        yl = Y_L_seasonal(s, t₀, cycle, p)
        W_s = spline_eval(s, W_coeffs)
        return surv_s * λ_s * disc_s * (max(yl, 1e-10) + W_s)
    end

    loss_integral, _ = quadgk(integrand, t₀ + 1e-6, T_star; rtol=1e-6)
    return harvest_term + loss_integral
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. W̃ at all nodes
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_Wtilde_at_nodes(τ_star_coeffs, W_coeffs, p; N=10)

Compute W̃(t₀) at 2N+1 nodes using the f/g decomposition with u → identity.
Uses the fixed τ* from Stage A.

Returns: (Wtilde_coeffs, Wtilde_values, nodes, f_values, g_values, T_star_values)
"""
function compute_Wtilde_at_nodes(τ_star_coeffs, W_coeffs, p; N=10)
    nodes = fourier_nodes(N)
    Wtilde_values = Float64[]
    f_values = Float64[]
    g_values = Float64[]
    T_star_values = Float64[]

    for t₀ in nodes
        τ_star = spline_eval(t₀, τ_star_coeffs)
        T_star = t₀ + τ_star
        push!(T_star_values, T_star)

        decomp = compute_Wtilde_decomposed(t₀, T_star, p)
        push!(f_values, decomp.f)
        push!(g_values, decomp.g)

        W_T = spline_eval(T_star, W_coeffs)
        push!(Wtilde_values, decomp.f + decomp.g * W_T)
    end

    Wtilde_coeffs = make_spline(nodes, Wtilde_values)
    return (Wtilde_coeffs = Wtilde_coeffs, Wtilde_values = Wtilde_values,
            nodes = nodes, f_values = f_values, g_values = g_values,
            T_star_values = T_star_values)
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Linear system solve for W at all nodes (fixed policy)
# ──────────────────────────────────────────────────────────────────────────────

"""
    update_W_all_nodes(τ_star_coeffs, Wtilde_data, d_values, p; N=10)

Update W(t) at all 2N+1 nodes using the fixed policy (τ*, d*) from Stage A.

Unlike update_V_all_nodes, this does NOT re-solve the stocking FOC — d* values
are passed in directly from the Stage A solution.

Builds and solves: (I - diag(α·g)·𝒲)·w = α·f_W
"""
function update_W_all_nodes(τ_star_coeffs, Wtilde_data, d_values, p; N=10)
    nodes = fourier_nodes(N)
    M = 2N + 1

    W_mat = zeros(M, M)    # spline interpolation weight matrix
    α_f = zeros(M)
    α_g = zeros(M)
    t0_values = zeros(M)
    T_star_at_t0 = zeros(M)

    for (i, t) in enumerate(nodes)
        d_star = d_values[i]
        t0_star = t + d_star
        t0_values[i] = t0_star

        if d_star == 0.0
            # Reuse precomputed f/g directly (node == W̃-node)
            f_i = Wtilde_data.f_values[i]
            g_i = Wtilde_data.g_values[i]
            T_star = Wtilde_data.T_star_values[i]
        else
            # d* > 0: need f/g at t₀* = t + d*, offset from node
            τ_star = spline_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star
            decomp = compute_Wtilde_decomposed(t0_star, T_star, p)
            f_i = decomp.f
            g_i = decomp.g
        end

        T_star_at_t0[i] = T_star
        disc_fallow = exp(-p.δ * d_star)
        α_f[i] = disc_fallow * f_i
        α_g[i] = disc_fallow * g_i

        # Spline interpolation weights for W(Tᵢ*)
        iw = spline_interp_weights(T_star, nodes)
        W_mat[i, iw.idx_lo] += 1 - iw.weight
        W_mat[i, iw.idx_hi] += iw.weight
    end

    # Solve (I - diag(α_g) · W_mat) · w = α_f
    A = Matrix{Float64}(I, M, M) - Diagonal(α_g) * W_mat
    w = A \ α_f

    W_new_coeffs = make_spline(nodes, w)

    return (W_new_coeffs = W_new_coeffs, W_values = copy(w),
            t0_values = t0_values, nodes = nodes)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main solver: dollar continuation value W(t)
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_dollar_continuation_value(model_result, p;
        N = 10, max_iter = 50, tol = 1e-4, damping = 0.5, verbose = true)

Compute the dollar continuation value W(t) under the fixed policy from Stage A.

Stage A output (from solve_seasonal_model) provides:
  - τ_star_coeffs: optimal cycle duration spline
  - d_values: optimal fallow durations at nodes
  - nodes: node positions

The algorithm iterates the same linear-system approach as solve_seasonal_model
but with u(·) replaced by the identity and no FOC re-solving.

Phase 1: Direct solve via f_W/g decomposition (fast convergence).
Phase 2: Bellman refinement with full W(s) evaluation at quadrature points.

Returns a NamedTuple with:
  - W_coeffs: converged periodic spline for W(t)
  - converged: whether the solver converged
  - iterations: total iterations performed
  - history: convergence trace
"""
function solve_dollar_continuation_value(model_result, p;
        N = 10, max_iter = 50, tol = 1e-4, damping = 0.5, verbose = true)

    τ_star_coeffs = model_result.τ_star_coeffs
    d_values = model_result.d_values
    nodes = model_result.nodes

    # ── Initialisation ────────────────────────────────────────────────────────
    W0 = estimate_initial_W(p)
    W_coeffs = initialize_V_constant(W0; N=N)  # reuse the same spline initializer
    verbose && println("Stage B: Initialized W(t) = $W0 (constant)")

    history = Tuple{Int, Float64}[]
    converged = false
    iter = 0

    # ── Phase 1: Direct solve (f_W/g decomposition) ──────────────────────────
    for k in 1:max_iter
        iter = k

        verbose && print("  Iteration $k: computing W̃...")

        # Compute W̃ decomposition at nodes (using fixed τ*)
        Wtilde_data = compute_Wtilde_at_nodes(τ_star_coeffs, W_coeffs, p; N=N)
        Wt_mean = sum(Wtilde_data.Wtilde_values) / length(Wtilde_data.Wtilde_values)
        verbose && print(" W̃̄=$(round(Wt_mean; digits=0))")

        # Solve linear system for W using fixed d*
        W_result = update_W_all_nodes(τ_star_coeffs, Wtilde_data, d_values, p; N=N)

        # Damped update
        W_new_coeffs = damped_update(W_coeffs, W_result.W_new_coeffs; α=damping)

        # Convergence check
        Δ = maximum(abs.(W_new_coeffs.values .- W_coeffs.values))
        push!(history, (k, Δ))
        W_bar = sum(W_new_coeffs.values) / length(W_new_coeffs.values)
        verbose && println(" W̄=$(round(W_bar; digits=0)) ΔW=$(round(Δ; sigdigits=4))")

        W_coeffs = W_new_coeffs

        if Δ < tol
            converged = true
            verbose && println("Phase 1 converged after $k iterations (ΔW = $(round(Δ; sigdigits=4)) < $tol)")
            break
        end

        # Divergence detection
        if length(history) >= 4
            recent = [h[2] for h in history[end-2:end]]
            if recent[2] > recent[1] && recent[3] > recent[2]
                converged = true
                verbose && println("Phase 1 converged (divergence detected at iter $k)")
                break
            end
        end
    end

    if !converged && verbose
        println("WARNING: Phase 1 did not converge after $max_iter iterations")
    end

    direct_iter = iter

    # ── Phase 2: Bellman fixed-point refinement ──────────────────────────────
    verbose && println("\n  Phase 2: Bellman refinement for W...")

    bellman_history = Tuple{Int, Float64}[]
    bellman_converged = false
    bellman_iter = 0

    for k in 1:max_iter
        bellman_iter = k

        verbose && print("    Bellman $k:")

        W_new_values = zeros(length(nodes))

        for (i, t) in enumerate(nodes)
            d_star = d_values[i]
            t0_star = t + d_star
            τ_star = spline_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star

            Wt = compute_Wtilde(t0_star, T_star, W_coeffs, p)
            W_new_values[i] = exp(-p.δ * d_star) * Wt
        end

        W_new_coeffs = make_spline(nodes, W_new_values)
        W_new_coeffs = damped_update(W_coeffs, W_new_coeffs; α=damping)

        Δ = maximum(abs.(W_new_coeffs.values .- W_coeffs.values))
        push!(bellman_history, (k, Δ))
        W_bar = sum(W_new_coeffs.values) / length(W_new_coeffs.values)
        verbose && println(" W̄=$(round(W_bar; digits=0)) ΔW=$(round(Δ; sigdigits=4))")

        W_coeffs = W_new_coeffs

        if Δ < tol
            bellman_converged = true
            verbose && println("  Bellman converged after $k iterations (ΔW = $(round(Δ; sigdigits=4)) < $tol)")
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
        println("  WARNING: Bellman did not converge after $max_iter iterations")
    end

    full_history = vcat(history, [(direct_iter + k, Δ) for (k, Δ) in bellman_history])

    return (
        W_coeffs   = W_coeffs,
        converged  = converged && bellman_converged,
        iterations = direct_iter + bellman_iter,
        history    = full_history,
    )
end
