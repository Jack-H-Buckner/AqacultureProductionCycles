"""
    simulation_validation_profit_coverage.jl

Validate the Stage D continuation value V(t) against Monte Carlo simulation
for the medium-risk seasonal case with profit-coverage insurance (ξ=0.1, ξ=0.25).

At each of 12 monthly starting dates, simulates 1500 paths of 100 cycles each
under the Stage D optimal policy, computing cycle incomes using the
profit-coverage indemnity (Stage C three-ODE sequence). Compares the simulated
expected present utility to the solver's V(t).

Outputs:
- `simulation_validation_profit_coverage.csv` — analytical V, simulated mean,
  SE at each starting date for both ξ values
"""

using CSV, DataFrames, Statistics, Random

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "07_continuation_value_solver_with_profit_coverage.jl"))
include(joinpath(@__DIR__, "..", "src", "04_simulate_production_cycles.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))


# ──────────────────────────────────────────────────────────────────────────────
# Custom simulation: cycle with profit-coverage payoffs
# ──────────────────────────────────────────────────────────────────────────────

"""
    simulate_cycle_pc(t_start, cycle_idx, model_result, W_coeffs, p, rng)

Simulate a single production cycle using profit-coverage indemnity.
Uses the Stage D policy (τ*, d*) from `model_result` and computes cycle
incomes via the Stage C three-ODE sequence.
"""
function simulate_cycle_pc(t_start, cycle_idx, model_result, W_coeffs, p, rng)
    # Optimal policy from Stage D
    d_star = interpolate_d_star(t_start, model_result.nodes, model_result.d_values)
    t0 = t_start + d_star
    tau_star = spline_eval(t0, model_result.τ_star_coeffs)
    T_planned = t0 + tau_star

    # Pre-solve base cycle ODEs (growth, mortality, cumulative hazard)
    cycle = prepare_cycle(t0, T_planned, p)

    # Sample loss event (hazard process is independent of indemnity type)
    τ_loss = sample_loss_time(t0, T_planned, cycle.Λ_sol, rng)
    loss = !isnothing(τ_loss)
    t_end = loss ? τ_loss : T_planned

    # Compute income using profit-coverage indemnity (Stage C)
    L_sol = cycle.L_sol
    n_sol = cycle.n_sol

    if p.ξ == 0.0
        # Breakeven: use standard cycle cash flows
        if loss
            income = Y_L_seasonal(t_end, t0, cycle, p)
        else
            income = Y_H_seasonal(t_end, t0, cycle, p)
        end
    else
        # Profit coverage: run Stage C three-ODE sequence
        result_pc = solve_indemnity_with_opportunity_costs(
            t0, T_planned, W_coeffs, L_sol, n_sol, p; verbose=false)

        if loss
            # Y_L at loss time from profit-coverage indemnity
            I_τ = result_pc.I_sol(t_end)
            stocking = p.c_s * exp(p.δ_b * (t_end - t0))
            Φ_val = cycle.Φ_sol(t_end)
            # Need accumulated premium from the profit-coverage ODE
            # The I_sol from solve_indemnity_with_opportunity_costs wraps step3
            # which has [I, P] as state. We need to reconstruct P.
            # Use the breakeven premium for the loss integral since solve_indemnity_with_opportunity_costs
            # returns I_sol as a scalar function. Recompute Π from the PC indemnity.
            I_pc_sol = result_pc.I_sol
            Π_pc = solve_accumulated_premium_from_func(t0, t_end, I_pc_sol, p)
            income = I_τ - stocking - Φ_val - Π_pc - p.c₂
        else
            income = result_pc.Y_H_converged
        end
    end

    return CycleOutcome(
        cycle_idx, t_start, d_star, t0, tau_star, T_planned, t_end,
        t_end - t0, loss, income, 0.0,
        cycle.L_sol(t_end), W_weight(cycle.L_sol(t_end), p),
        cycle.n_sol(t_end), cycle.n_sol(t_end) * W_weight(cycle.L_sol(t_end), p) / 1000.0,
        cycle.Φ_sol(t_end), 0.0, 0.0,
    )
end

"""
    solve_accumulated_premium_from_func(t₀, T, I_func, p)

Compute accumulated premium Π(T) given an indemnity function I(t).
Solves Π'(t) = π(t) + δ_b·Π, Π(t₀) = 0, where π(t) = (λ(t)·I(t) + c_I)/(1-Q).
"""
function solve_accumulated_premium_from_func(t₀, T, I_func, p)
    function dΠdt(Π, params, t)
        π_t = (λ(t, params) * I_func(t) + params.c_I) / (1 - params.Q)
        return π_t + params.δ_b * Π
    end
    prob = ODEProblem(dΠdt, 0.0, (t₀, T), p)
    sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)
    return sol(T)
end

"""
    simulate_production_cycles_pc(model_result, W_coeffs, p; kwargs...)

Monte Carlo simulation using profit-coverage payoffs.
"""
function simulate_production_cycles_pc(model_result, W_coeffs, p;
        n_cycles::Int = 100,
        n_sims::Int   = 1000,
        t_init::Float64 = 0.0,
        seed::Int = SEED,
    )
    rng = MersenneTwister(seed)
    all_paths = Vector{Vector{CycleOutcome}}(undef, n_sims)

    for sim in 1:n_sims
        path = CycleOutcome[]
        t_current = t_init

        for c in 1:n_cycles
            outcome = simulate_cycle_pc(t_current, c, model_result, W_coeffs, p, rng)
            push!(path, outcome)
            t_current = outcome.t_end
        end

        all_paths[sim] = path
    end

    return all_paths
end


# ══════════════════════════════════════════════════════════════════════════════
# 1. Solve Stage A: medium-risk seasonal model via homotopy
# ══════════════════════════════════════════════════════════════════════════════

p_seasonal = merge(default_params, (
    λ_coeffs = λ_medium_coeffs,
    γ        = 0.5,
    Y_MIN    = 0.0,
    ξ        = 0.0,
))

# Homogeneous warm start
p_hom = merge(homogeneous_params, (
    λ_const = exp(λ_medium_coeffs.a0),
    γ       = 0.5,
    Y_MIN   = 0.0,
))

println("Solving homogeneous warm start (medium risk)...")
T_hom = solve_insurance(p_hom)
I_hom = solve_indemnity_homogeneous(T_hom, p_hom)
V_hom = insurance_value(T_hom, I_hom, p_hom)
println("  T* = $(round(T_hom; digits=1)) days, V* = $(round(V_hom; digits=0))")

# Homotopy
function scale_seasonal_params(p, α)
    scale_coeffs(c) = (a0 = c.a0, a = α .* c.a, b = α .* c.b)
    return merge(p, (
        λ_coeffs = scale_coeffs(p.λ_coeffs),
        m_coeffs = scale_coeffs(p.m_coeffs),
        k_coeffs = scale_coeffs(p.k_coeffs),
    ))
end

N_solver = 10
homotopy_steps = [0.25, 0.5, 0.75, 1.0]
V_coeffs = initialize_V_constant(V_hom; N=N_solver)
model_result = nothing

println("\nStage A: Solving seasonal model (medium risk)...")
for (i, α) in enumerate(homotopy_steps)
    p_scaled = scale_seasonal_params(p_seasonal, α)
    println("  Homotopy $i/$(length(homotopy_steps)): α = $α")

    global model_result = solve_seasonal_model(p_scaled;
        N        = N_solver,
        V_init   = V_coeffs,
        max_iter = 20,
        tol      = 1e-3,
        damping  = 0.5,
        verbose  = true,
    )

    global V_coeffs = model_result.V_coeffs
end

println("Stage A solved (converged = $(model_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Solve Stage B: dollar continuation value W(t)
# ══════════════════════════════════════════════════════════════════════════════

println("\nStage B: Solving dollar continuation value W(t)...")
W_result = solve_dollar_continuation_value(model_result, p_seasonal;
    N       = N_solver,
    max_iter = 20,
    tol     = 1e-3,
    damping = 0.5,
    verbose = true,
)
W_coeffs = W_result.W_coeffs
println("Stage B solved (converged = $(W_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Solve Stage D at ξ=0.1 and ξ=0.25
# ══════════════════════════════════════════════════════════════════════════════

ξ_values = [0.1, 0.25]
stage_D_results = Dict{Float64, Any}()

for ξ_val in ξ_values
    p_xi = merge(p_seasonal, (ξ = ξ_val,))

    println("\nStage D: Solving with ξ=$ξ_val via continuation...")
    cont_result = solve_stage_D_continuation(model_result, W_coeffs, p_xi;
        ξ_target             = ξ_val,
        n_steps              = 3,
        N                    = N_solver,
        max_iter             = 25,
        tol                  = 1e-3,
        damping              = 0.5,
        verbose              = true,
    )
    stage_D_results[ξ_val] = cont_result.result
    println("Stage D (ξ=$ξ_val) solved (converged = $(cont_result.result.converged))")
end

# ══════════════════════════════════════════════════════════════════════════════
# 4. Simulate from multiple starting dates for each ξ
# ══════════════════════════════════════════════════════════════════════════════

N_CYCLES = 50
N_SIMS   = 500
t_inits  = collect(0.0:60.0:300.0)

# Output accumulator
out_t_init = Float64[]
out_xi = Float64[]
out_V_analytical = Float64[]
out_sim_mean = Float64[]
out_sim_se = Float64[]

for ξ_val in ξ_values
    p_xi = merge(p_seasonal, (ξ = ξ_val,))
    sd_result = stage_D_results[ξ_val]

    println("\nSimulating ξ=$ξ_val: $N_SIMS paths × $N_CYCLES cycles at $(length(t_inits)) starting dates...")

    for (j, t0_init) in enumerate(t_inits)
        println("  ξ=$ξ_val, t₀ = $(round(t0_init; digits=0))...")

        all_paths = simulate_production_cycles_pc(sd_result, W_coeffs, p_xi;
            n_cycles = N_CYCLES,
            n_sims   = N_SIMS,
            t_init   = t0_init,
            seed     = SEED + j,
        )

        # Compute path-level discounted utility + terminal V
        path_utilities = Float64[]
        for path in all_paths
            path_u = 0.0
            for outcome in path
                Y = max(outcome.income, 1e-10)
                path_u += exp(-p_xi.δ * (outcome.t_end - t0_init)) * u(Y, p_xi)
            end
            # Terminal continuation value from Stage D
            t_last = path[end].t_end
            V_terminal = spline_eval(t_last, sd_result.V_coeffs)
            path_u += exp(-p_xi.δ * (t_last - t0_init)) * V_terminal
            push!(path_utilities, path_u)
        end

        push!(out_t_init, t0_init)
        push!(out_xi, ξ_val)
        push!(out_V_analytical, spline_eval(t0_init, sd_result.V_coeffs))
        push!(out_sim_mean, mean(path_utilities))
        push!(out_sim_se, std(path_utilities) / sqrt(N_SIMS))
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 5. Export CSV
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

df = DataFrame(
    t_init        = out_t_init,
    xi            = out_xi,
    V_analytical  = out_V_analytical,
    sim_mean      = out_sim_mean,
    sim_se        = out_sim_se,
    sim_lower_2se = out_sim_mean .- 2 .* out_sim_se,
    sim_upper_2se = out_sim_mean .+ 2 .* out_sim_se,
)
CSV.write(joinpath(outdir, "simulation_validation_profit_coverage.csv"), df)
println("\nWrote simulation_validation_profit_coverage.csv ($(nrow(df)) rows)")

# ── Summary ───────────────────────────────────────────────────────────────────

for ξ_val in ξ_values
    mask = out_xi .== ξ_val
    V_an = out_V_analytical[mask]
    s_mean = out_sim_mean[mask]
    s_se = out_sim_se[mask]

    println("\n── Validation summary (medium risk, ξ=$ξ_val) ──────────")
    for (j, t0) in enumerate(t_inits)
        z = abs(s_mean[j] - V_an[j]) / s_se[j]
        flag = z > 3.0 ? " ← OUTSIDE 3SE" : ""
        println("  t₀=$(round(t0; digits=0)): " *
                "V=$(round(V_an[j]; digits=0)), " *
                "E[U]=$(round(s_mean[j]; digits=0)), " *
                "SE=$(round(s_se[j]; digits=0)), " *
                "|z|=$(round(z; digits=2))$flag")
    end
    rel_err = mean(abs.(s_mean .- V_an) ./ abs.(V_an))
    println("  Mean absolute relative error = $(round(rel_err * 100; digits=4))%")
end
