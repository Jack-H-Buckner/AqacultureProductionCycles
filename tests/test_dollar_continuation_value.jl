"""
    test_dollar_continuation_value.jl

Validates the dollar continuation value W(t) from 05_dollar_continuation_value.jl
against Monte Carlo simulation in the homogeneous (constant-rate) case.

W(t) is the expected NPV in dollars (not utility) of the production system. In
the homogeneous case, W is constant across all calendar dates and equals:

    W = S·e^{-δτ}·Y_H / (1 - S·e^{-δτ})

The simulated expected NPV — the discounted sum of per-cycle dollar incomes
plus a terminal W(t_last) — should converge to the solver's W as the number
of sample paths grows.

This test:
1. Solves the homogeneous model analytically → V*, T*.
2. Runs the seasonal solver (Stage A) with constant rates → policy splines.
3. Runs the dollar continuation value solver (Stage B) → W(t).
4. Simulates production cycles from multiple starting dates.
5. Computes path-level discounted dollar income (Σ e^{-δt}·Y + terminal W).
6. Compares the simulated E[dollar] to the solver's W at each starting date.
7. Exports CSV for validation plotting.
"""

using Test, CSV, DataFrames, Statistics, Random

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "04_simulate_production_cycles.jl"))
include(joinpath(@__DIR__, "..", "src", "05_dollar_continuation_value.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# Setup: solve the homogeneous model (Stage A)
# ══════════════════════════════════════════════════════════════════════════════

# Seasonal parameters with zero harmonics → constant rates
const hom_seasonal_params = merge(default_params, (
    λ_coeffs = (a0 = log(λ_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    m_coeffs = (a0 = log(m_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    k_coeffs = (a0 = log(k_const), a = [0.0, 0.0], b = [0.0, 0.0]),
))

println("Solving homogeneous case analytically...")
const T_star_hom = solve_insurance(homogeneous_params)
const I_sol_hom = solve_indemnity_homogeneous(T_star_hom, homogeneous_params)
const V_hom = insurance_value(T_star_hom, I_sol_hom, homogeneous_params)
println("  T* = $(round(T_star_hom; digits=2)) days")
println("  V* = $(round(V_hom; digits=2))")

# Run seasonal solver with constant rates (Stage A)
const N_SOLVER = 10
const V_init = initialize_V_constant(V_hom; N=N_SOLVER)

println("\nStage A: Running seasonal solver with constant rates...")
const model = solve_seasonal_model(hom_seasonal_params;
    N = N_SOLVER,
    V_init = V_init,
    max_iter = 200,
    tol = 1e-4,
    damping = 1.0,
    verbose = true,
)

# ══════════════════════════════════════════════════════════════════════════════
# Stage B: Solve dollar continuation value W(t)
# ══════════════════════════════════════════════════════════════════════════════

println("\nStage B: Solving dollar continuation value W(t)...")
const W_result = solve_dollar_continuation_value(model, hom_seasonal_params;
    N = N_SOLVER,
    max_iter = 200,
    tol = 1e-4,
    damping = 1.0,
    verbose = true,
)

# Analytical W for homogeneous case using f/g decomposition:
#   W̃ = f + g·W, and W = W̃ (since d*=0), so W = f/(1-g)
# where f = S·e^{-δT*}·Y_H + ∫ S·λ·e^{-δs}·Y_L ds  (dollar income, both branches)
#       g = S·e^{-δT*}      + ∫ S·λ·e^{-δs} ds       (expected discount factor)
const decomp_hom = compute_Wtilde_decomposed(0.0, T_star_hom, hom_seasonal_params)
const W_analytical = decomp_hom.f / (1 - decomp_hom.g)
println("  W_analytical = $(round(W_analytical; digits=2))")
println("  W_solver (mean) = $(round(mean(W_result.W_coeffs.values); digits=2))")

# ══════════════════════════════════════════════════════════════════════════════
# Simulate from multiple starting dates
# ══════════════════════════════════════════════════════════════════════════════

const N_CYCLES = 100
const N_SIMS   = 1000    # higher than V test: dollar income has more variance (no utility compression)
const t_inits  = collect(0.0:30.0:330.0)  # 12 monthly starting dates

println("\nSimulating $N_SIMS paths × $N_CYCLES cycles at $(length(t_inits)) starting dates...")

sim_means = Float64[]
sim_ses   = Float64[]
W_analytical_vec = Float64[]

for (j, t0_init) in enumerate(t_inits)
    println("  t₀ = $(round(t0_init; digits=0))...")

    all_paths = simulate_production_cycles(model, hom_seasonal_params;
        n_cycles = N_CYCLES,
        n_sims   = N_SIMS,
        t_init   = t0_init,
        seed     = SEED + j,
    )

    # Compute path-level discounted DOLLAR income sum relative to t_init.
    # Use Y directly (not u(Y)) — this is the key difference from V validation.
    # Add terminal W(t_end_last) to account for the infinite tail.
    path_dollars = Float64[]
    for path in all_paths
        path_d = 0.0
        for outcome in path
            Y = max(outcome.income, 1e-10)
            path_d += exp(-hom_seasonal_params.δ * (outcome.t_end - t0_init)) * Y
        end
        # Terminal value: W at the end of the last cycle, discounted to t_init
        t_last = path[end].t_end
        W_terminal = spline_eval(t_last, W_result.W_coeffs)
        path_d += exp(-hom_seasonal_params.δ * (t_last - t0_init)) * W_terminal
        push!(path_dollars, path_d)
    end

    push!(sim_means, mean(path_dollars))
    push!(sim_ses, std(path_dollars) / sqrt(N_SIMS))
    push!(W_analytical_vec, W_analytical)  # constant in homogeneous case
end

# ══════════════════════════════════════════════════════════════════════════════
# Export CSV for plotting
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

df = DataFrame(
    t_init            = t_inits,
    W_analytical      = W_analytical_vec,
    sim_mean          = sim_means,
    sim_se            = sim_ses,
    sim_lower_2se     = sim_means .- 2 .* sim_ses,
    sim_upper_2se     = sim_means .+ 2 .* sim_ses,
)
CSV.write(joinpath(outdir, "dollar_cv_validation.csv"), df)
println("\nWrote dollar_cv_validation.csv ($(nrow(df)) rows)")

# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

@testset "Dollar Continuation Value Validation (Homogeneous)" begin

    @testset "Stage A solver converged" begin
        @test model.converged
    end

    @testset "Stage B solver converged" begin
        @test W_result.converged
    end

    @testset "W(t) is approximately constant (homogeneous)" begin
        W_vals = W_result.W_coeffs.values
        W_range = maximum(W_vals) - minimum(W_vals)
        W_mean = mean(W_vals)
        rel_range = W_range / abs(W_mean)
        println("  W range / mean = $(round(rel_range * 100; digits=4))%")
        @test rel_range < 0.01  # within 1% relative range
    end

    @testset "W solver matches analytical W" begin
        W_solver_mean = mean(W_result.W_coeffs.values)
        rel_err = abs(W_solver_mean - W_analytical) / abs(W_analytical)
        println("  W_solver  = $(round(W_solver_mean; digits=2))")
        println("  W_analytical = $(round(W_analytical; digits=2))")
        println("  Relative error = $(round(rel_err * 100; digits=4))%")
        @test rel_err < 0.05  # within 5%
    end

    @testset "Simulated E[\$] consistent with W(t) at each starting date" begin
        W_solver_mean = mean(W_result.W_coeffs.values)
        for (j, t0) in enumerate(t_inits)
            z_score = abs(sim_means[j] - W_solver_mean) / sim_ses[j]
            within_3se = z_score < 3.0  # 3SE for multiple comparisons across 12 dates
            println("  t₀=$(round(t0; digits=0)): " *
                    "E[\$]=$(round(sim_means[j]; digits=2)), " *
                    "W=$(round(W_solver_mean; digits=2)), " *
                    "SE=$(round(sim_ses[j]; digits=2)), " *
                    "|z|=$(round(z_score; digits=2))" *
                    (within_3se ? "" : " ← OUTSIDE 3SE"))
            @test within_3se
        end
    end

    @testset "Overall mean close to W" begin
        overall_mean = mean(sim_means)
        W_solver_mean = mean(W_result.W_coeffs.values)
        rel_err = abs(overall_mean - W_solver_mean) / abs(W_solver_mean)
        println("  Grand mean E[\$] = $(round(overall_mean; digits=2))")
        println("  W (solver)       = $(round(W_solver_mean; digits=2))")
        println("  Relative error   = $(round(rel_err * 100; digits=4))%")
        @test rel_err < 0.05  # within 5%
    end
end

println("\n── Summary ─────────────────────────────────────")
println("  W* (analytical)     = $(round(W_analytical; digits=2))")
println("  W  (solver, mean)   = $(round(mean(W_result.W_coeffs.values); digits=2))")
println("  E[\$] (sim, mean)   = $(round(mean(sim_means); digits=2))")
println("  E[\$] (sim, SE)     = $(round(mean(sim_ses); digits=2))")
W_solver_final = mean(W_result.W_coeffs.values)
println("  Relative error      = $(round(abs(mean(sim_means) - W_solver_final) / abs(W_solver_final) * 100; digits=2))%")
