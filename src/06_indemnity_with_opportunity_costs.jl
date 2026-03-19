"""
    06_indemnity_with_opportunity_costs.jl

Compute the profit-coverage indemnity I(τ) via three sequential ODE solves.
The indemnity compensates accumulated costs plus a fraction ξ of the opportunity
cost of losing the cycle, where opportunity costs are evaluated under breakeven
insurance.

    I(τ) = C(τ) + ξ · OC(τ)

where OC(τ) = W̄₀(τ) - W(τ) is the difference between the conditional value
of completing the current cycle and the value of starting fresh.

When ξ = 0 this reduces to the baseline breakeven indemnity and delegates
to the existing solve_indemnity() — a single ODE, no extra computation.

When ξ > 0, Stage C runs three sequential ODEs (no iteration):

  Step 1 (forward):  Breakeven indemnity I₀(τ), accumulated premium P₀(τ)
                     → Y_H⁰, Y_L⁰ (breakeven cash flows)

  Step 2 (backward): Conditional continuation value W̄₀(τ)
                     → OC(τ) = W̄₀(τ) - W(τ)

  Step 3 (forward):  Profit-coverage indemnity I(τ) with OC as known forcing
                     → I(τ), accumulated premium P(τ), Y_H

See updated_insurance_model.md § "Stage C" for full derivation.
"""

include("05_dollar_continuation_value.jl")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Step 1: Breakeven indemnity (forward ODE)
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_breakeven_cycle(t₀, T, L_sol, n_sol, p)

Solve the breakeven indemnity and accumulated premium on [t₀, T].
This is the existing solve_indemnity plus accumulated premium as auxiliary state.

Returns (I₀_sol, Π₀_sol, Y_H0, cycle) where:
  - I₀_sol: breakeven indemnity ODE solution
  - Π₀_sol: accumulated breakeven premium ODE solution
  - Y_H0: breakeven harvest income at T
  - cycle: the full cycle tuple from prepare_cycle
"""
function solve_breakeven_cycle(t₀, T, p)
    cycle = prepare_cycle(t₀, T, p)
    Y_H0 = Y_H_seasonal(T, t₀, cycle, p)
    return (I₀_sol = cycle.I_sol, Π₀_sol = cycle.Π_sol, Y_H0 = Y_H0, cycle = cycle)
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Step 2: Conditional continuation value (backward ODE)
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_conditional_value(t₀, T, breakeven, W_coeffs, p)

Solve the backward ODE for W̄₀(τ) — the expected dollar value of completing
the current cycle from time τ onward, conditional on survival to τ, using
breakeven cash flows.

    W̄₀'(τ) = (δ + λ(τ))·W̄₀(τ) - λ(τ)·[Y_L⁰(τ) + W(τ)]
    W̄₀(T) = Y_H⁰(T) + W(T)

Integrated backward by substituting σ = T - τ:

    Ŵ'(σ) = -(δ + λ(T-σ))·Ŵ(σ) + λ(T-σ)·[Y_L⁰(T-σ) + W(T-σ)]
    Ŵ(0) = Y_H⁰ + W(T)

Returns an ODE solution in the σ variable. To evaluate W̄₀ at calendar time τ,
use W̄₀_sol(T - τ).
"""
function solve_conditional_value(t₀, T, breakeven, W_coeffs, p)
    cycle = breakeven.cycle
    Y_H0 = breakeven.Y_H0
    W_T = spline_eval(T, W_coeffs)

    # Terminal condition (σ=0 corresponds to τ=T)
    Wbar0_T = Y_H0 + W_T

    # ODE in reversed time σ = T - τ
    function dWbar_dsigma(Wbar, params, σ)
        τ = T - σ
        λ_τ = λ(τ, params)
        yl = Y_L_seasonal(τ, t₀, cycle, params)
        W_τ = spline_eval(τ, W_coeffs)
        return -(params.δ + λ_τ) * Wbar + λ_τ * (max(yl, 1e-10) + W_τ)
    end

    σ_max = T - t₀
    prob = ODEProblem(dWbar_dsigma, Wbar0_T, (0.0, σ_max), p)
    return solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. Step 3: Profit-coverage indemnity (forward ODE with clamped OC)
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_profit_coverage_indemnity(t₀, T, breakeven, Wbar0_sol, W_coeffs,
                                    L_sol, n_sol, p)

Solve the profit-coverage indemnity ODE on [t₀, T]:

    I'(τ) = (λ(τ)/(1-Q) + δ_b)·I(τ) + φ(τ) + c_I/(1-Q) - δ_b·c₂
           + ξ·[OC⁺'(τ) - δ_b·OC⁺(τ)]

    I(t₀) = c_s + c₂ + ξ·OC⁺(t₀)

where OC(τ) = W̄₀(τ) - W(τ) and OC⁺(τ) = max(OC(τ), 0). The max ensures
the indemnity only covers positive opportunity costs, so I(τ) ≥ I₀(τ).

The ODE remains self-referential through the premium: π(τ) = (λ·I + c_I)/(1-Q)
enters the accumulated cost C(τ), which is why this must be an ODE rather than
a pointwise formula.

Accumulates the premium integral as auxiliary state:
    P'(τ) = π(τ) + δ_b·P,  P(t₀) = 0

Returns the ODE solution with u[1] = I(τ), u[2] = P(τ).
"""
function solve_profit_coverage_indemnity(t₀, T, breakeven, Wbar0_sol, W_coeffs,
                                          L_sol, n_sol, p)
    cycle = breakeven.cycle
    ξ_val = p.ξ

    # OC(τ) = W̄₀(τ) - W(τ)
    function OC_raw(τ)
        return Wbar0_sol(T - τ) - spline_eval(τ, W_coeffs)
    end

    # OC'(τ) = W̄₀'(τ) - W'(τ)
    function OC_prime_raw(τ)
        Wbar0_τ = Wbar0_sol(T - τ)
        λ_τ = λ(τ, p)
        yl = Y_L_seasonal(τ, t₀, cycle, p)
        W_τ = spline_eval(τ, W_coeffs)
        W_prime_τ = spline_derivative(τ, W_coeffs)

        Wbar0_prime = (p.δ + λ_τ) * Wbar0_τ - λ_τ * (max(yl, 1e-10) + W_τ)
        return Wbar0_prime - W_prime_τ
    end

    # OC⁺(τ) = max(OC(τ), 0) and d/dτ OC⁺
    OC_plus(τ) = max(OC_raw(τ), 0.0)
    OC_prime_plus(τ) = OC_raw(τ) > 0.0 ? OC_prime_raw(τ) : 0.0

    # Initial conditions
    I₀ = p.c_s + p.c₂ + ξ_val * OC_plus(t₀)
    P₀ = 0.0

    function ode_rhs!(du, u_state, params, τ)
        I_τ = u_state[1]
        P_τ = u_state[2]

        λ_τ = λ(τ, params)
        φ_τ = φ(τ, t₀, L_sol, n_sol, params)
        π_τ = (λ_τ * I_τ + params.c_I) / (1 - params.Q)

        du[1] = (λ_τ / (1 - params.Q) + params.δ_b) * I_τ +
                φ_τ +
                params.c_I / (1 - params.Q) -
                params.δ_b * params.c₂ +
                ξ_val * (OC_prime_plus(τ) - params.δ_b * OC_plus(τ))

        du[2] = π_τ + params.δ_b * P_τ
    end

    prob = ODEProblem(ode_rhs!, [I₀, P₀], (t₀, T), p)
    return solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Combined solver: three sequential ODEs
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_indemnity_with_opportunity_costs(t₀, T, W_coeffs, L_sol, n_sol, p;
        verbose=false)

Solve the profit-coverage indemnity via three sequential ODE solves.

When ξ = 0, delegates to the existing solve_indemnity() (single ODE).

When ξ > 0, runs:
  Step 1: Breakeven indemnity forward → I₀, Y_H⁰, Y_L⁰
  Step 2: Conditional continuation value backward → W̄₀(τ), OC(τ)
  Step 3: Profit-coverage indemnity forward → I(τ), P(τ)

No iteration is required — each step feeds the next.

Returns:
  - I_sol: function τ → I(τ) on [t₀, T]
  - OC_sol: function τ → OC(τ) on [t₀, T] (opportunity cost)
  - Y_H_converged: harvest income under profit coverage
  - Y_H0: breakeven harvest income
  - Π_T: total accumulated premium at T
  - OC_t0: opportunity cost at stocking (should be ≈ 0)
"""
function solve_indemnity_with_opportunity_costs(t₀, T, W_coeffs, L_sol, n_sol, p;
        verbose=false)

    # ── ξ = 0 shortcut: baseline breakeven indemnity ─────────────────────────
    if p.ξ == 0.0
        I_sol = solve_indemnity(t₀, T, L_sol, n_sol, p)
        Π_sol = solve_accumulated_premium(t₀, T, I_sol, p)
        Π_T = Π_sol(T)

        v_T = v(T, t₀, L_sol, n_sol, p)
        Φ_T = Φ_accumulated(T, t₀, L_sol, n_sol, p)
        stocking = p.c_s * exp(p.δ_b * (T - t₀))
        Y_H_val = v_T - stocking - Φ_T - Π_T - p.c_h

        return (I_sol = I_sol, OC_sol = τ -> 0.0,
                Y_H_converged = Y_H_val, Y_H0 = Y_H_val,
                Π_T = Π_T, OC_t0 = 0.0)
    end

    # ── Step 1: Breakeven indemnity ──────────────────────────────────────────
    breakeven = solve_breakeven_cycle(t₀, T, p)
    verbose && println("    Step 1: Y_H⁰ = $(round(breakeven.Y_H0; digits=0))")

    # ── Step 2: Conditional continuation value (backward) ────────────────────
    Wbar0_sol = solve_conditional_value(t₀, T, breakeven, W_coeffs, p)

    OC_t0 = Wbar0_sol(T - t₀) - spline_eval(t₀, W_coeffs)
    verbose && println("    Step 2: OC(t₀) = $(round(OC_t0; digits=2)), OC(T*) = $(round(Wbar0_sol(0.0) - spline_eval(T, W_coeffs); digits=0))")

    # ── Step 3: Profit-coverage indemnity (pointwise + premium ODE) ─────────
    step3 = solve_profit_coverage_indemnity(t₀, T, breakeven, Wbar0_sol,
                                             W_coeffs, L_sol, n_sol, p)

    Π_T = step3(T)[2]
    v_T = v(T, t₀, L_sol, n_sol, p)
    Φ_T = Φ_accumulated(T, t₀, L_sol, n_sol, p)
    stocking = p.c_s * exp(p.δ_b * (T - t₀))
    Y_H_val = v_T - stocking - Φ_T - Π_T - p.c_h
    verbose && println("    Step 3: Y_H = $(round(Y_H_val; digits=0)) (Δ from breakeven: $(round(Y_H_val - breakeven.Y_H0; digits=0)))")

    OC_func(τ) = Wbar0_sol(T - τ) - spline_eval(τ, W_coeffs)

    return (I_sol = t -> step3(t)[1],
            OC_sol = OC_func,
            Y_H_converged = Y_H_val,
            Y_H0 = breakeven.Y_H0,
            Π_T = Π_T,
            OC_t0 = OC_t0)
end


# ──────────────────────────────────────────────────────────────────────────────
# 5. Convenience: solve indemnity at all nodes under the Stage A policy
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_indemnity_all_nodes(model_result, W_coeffs, p; N=10, verbose=false)

Solve the profit-coverage indemnity at each of the 2N+1 stocking-date nodes.

When ξ = 0, each node uses the baseline solve_indemnity (single ODE).
When ξ > 0, each node runs the three sequential ODE solves.

Returns a NamedTuple with:
  - I_solutions: vector of I(τ) solution functions
  - OC_solutions: vector of OC(τ) solution functions
  - Y_H_values: harvest income under profit coverage
  - Y_H0_values: breakeven harvest income
  - Π_T_values: total premium Π(T*)
  - OC_t0_values: opportunity cost at stocking (should be ≈ 0)
  - t0_values, T_star_values: stocking/harvest dates
  - all_converged: always true (no iteration)
"""
function solve_indemnity_all_nodes(model_result, W_coeffs, p; N=10, verbose=false)
    nodes = model_result.nodes
    τ_star_coeffs = model_result.τ_star_coeffs
    d_values = model_result.d_values

    I_solutions = []
    OC_solutions = []
    Y_H_values = Float64[]
    Y_H0_values = Float64[]
    Π_T_values = Float64[]
    OC_t0_values = Float64[]
    t0_values = Float64[]
    T_star_values = Float64[]

    for (i, t) in enumerate(nodes)
        d_star = d_values[i]
        t0_star = t + d_star
        τ_star = spline_eval(t0_star, τ_star_coeffs)
        T_star = t0_star + τ_star

        push!(t0_values, t0_star)
        push!(T_star_values, T_star)

        L_sol = solve_length(t0_star, T_star, p.L₀, p)
        n_sol = solve_numbers(t0_star, T_star, p.n₀, p)

        verbose && println("  Node $i (t=$t, t₀=$(round(t0_star; digits=1)), T*=$(round(T_star; digits=1))):")

        result = solve_indemnity_with_opportunity_costs(
            t0_star, T_star, W_coeffs, L_sol, n_sol, p; verbose=verbose)

        push!(I_solutions, result.I_sol)
        push!(OC_solutions, result.OC_sol)
        push!(Y_H_values, result.Y_H_converged)
        push!(Y_H0_values, result.Y_H0)
        push!(Π_T_values, result.Π_T)
        push!(OC_t0_values, result.OC_t0)
    end

    return (I_solutions = I_solutions, OC_solutions = OC_solutions,
            Y_H_values = Y_H_values, Y_H0_values = Y_H0_values,
            Π_T_values = Π_T_values, OC_t0_values = OC_t0_values,
            t0_values = t0_values, T_star_values = T_star_values,
            all_converged = true)
end


# ──────────────────────────────────────────────────────────────────────────────
# 6. Full pipeline: Stage A → Stage B → Stage C
# ──────────────────────────────────────────────────────────────────────────────

"""
    solve_model_with_opportunity_cost_indemnity(p;
        N=10, max_iter=50, tol=1e-4, damping=0.5, verbose=true, kwargs...)

Run the full three-stage pipeline:
  Stage A: solve_seasonal_model(p)     → V(t), τ*(t₀), d*(t)
  Stage B: solve_dollar_continuation_value(model, p) → W(t)
           (skipped when ξ = 0)
  Stage C: three sequential ODE solves → I(τ), OC(τ), Y_H

W(t) is independent of ξ — Stages A and B only need to run once.
Different ξ values only require re-running Stage C.

Returns a NamedTuple combining all stage outputs.
"""
function solve_model_with_opportunity_cost_indemnity(p;
        N=10, max_iter=50, tol=1e-4, damping=0.5, verbose=true, kwargs...)

    # ── Stage A ──────────────────────────────────────────────────────────────
    verbose && println("═══ Stage A: Solving risk-averse model ═══")
    model_result = solve_seasonal_model(p; N=N, max_iter=max_iter, tol=tol,
                                         damping=damping, verbose=verbose, kwargs...)

    # ── Stage B ──────────────────────────────────────────────────────────────
    if p.ξ > 0.0
        verbose && println("\n═══ Stage B: Solving dollar continuation value W(t) ═══")
        W_result = solve_dollar_continuation_value(model_result, p;
                    N=N, max_iter=max_iter, tol=tol, damping=damping, verbose=verbose)
        W_coeffs = W_result.W_coeffs
    else
        verbose && println("\n═══ Stage B: Skipped (ξ = 0) ═══")
        W_coeffs = initialize_V_constant(0.0; N=N)
        W_result = nothing
    end

    # ── Stage C ──────────────────────────────────────────────────────────────
    verbose && println("\n═══ Stage C: Solving indemnity with opportunity costs ═══")
    indemnity_result = solve_indemnity_all_nodes(model_result, W_coeffs, p;
                        N=N, verbose=verbose)

    return (
        model_result     = model_result,
        W_coeffs         = W_coeffs,
        W_result         = W_result,
        indemnity_result = indemnity_result,
    )
end
