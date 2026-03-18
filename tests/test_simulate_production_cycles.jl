"""
    test_simulate_production_cycles.jl

Validates the Monte Carlo simulation (04_simulate_production_cycles.jl) against
the analytical continuation value from the homogeneous model.

In the homogeneous (constant-rate) case, the continuation value V is constant
across all calendar dates, and V = Ṽ (since d* = 0). The simulated expected
present utility — the discounted sum of per-cycle CRRA utilities — should
converge to V as the number of sample paths grows.

This test:
1. Solves the homogeneous model analytically → V*, T*.
2. Runs the seasonal solver with constant rates to get policy splines.
3. Simulates production cycles from multiple starting dates.
4. Compares the simulated E[U] to V* at each starting date.
5. Exports CSV for the validation plot (visualizations/plot_simulation_validation.R).
"""

using Test, CSV, DataFrames, Statistics, Random

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "04_simulate_production_cycles.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# Setup: solve the homogeneous model
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

# Run seasonal solver with constant rates
const N_SOLVER = 10
const V_init = initialize_V_constant(V_hom; N=N_SOLVER)

println("\nRunning seasonal solver with constant rates...")
const model = solve_seasonal_model(hom_seasonal_params;
    N = N_SOLVER,
    V_init = V_init,
    max_iter = 200,
    tol = 1e-4,
    damping = 1.0,
    verbose = true,
)

# ══════════════════════════════════════════════════════════════════════════════
# Simulate from multiple starting dates
# ══════════════════════════════════════════════════════════════════════════════

const N_CYCLES = 100    # enough cycles so truncation bias is negligible
const N_SIMS   = 500
const t_inits  = collect(0.0:30.0:330.0)  # 12 monthly starting dates

println("\nSimulating $N_SIMS paths × $N_CYCLES cycles at $(length(t_inits)) starting dates...")

# Store results: for each t_init, compute path-level discounted utility
sim_means = Float64[]
sim_ses   = Float64[]
Vtilde_analytical = Float64[]

for (j, t0_init) in enumerate(t_inits)
    println("  t₀ = $(round(t0_init; digits=0))...")

    all_paths = simulate_production_cycles(model, hom_seasonal_params;
        n_cycles = N_CYCLES,
        n_sims   = N_SIMS,
        t_init   = t0_init,
        seed     = SEED + j,  # different seed per starting date
    )

    # Compute path-level discounted utility sum, discounted relative to t_init.
    # Add terminal continuation value V(t_end_last) to account for the
    # infinite tail beyond the simulated horizon.
    path_utilities = Float64[]
    for path in all_paths
        path_u = 0.0
        for outcome in path
            Y = max(outcome.income, 1e-10)
            path_u += exp(-hom_seasonal_params.δ * (outcome.t_end - t0_init)) * u(Y, hom_seasonal_params)
        end
        # Terminal value: V at the end of the last cycle, discounted to t_init
        t_last = path[end].t_end
        V_terminal = spline_eval(t_last, model.V_coeffs)
        path_u += exp(-hom_seasonal_params.δ * (t_last - t0_init)) * V_terminal
        push!(path_utilities, path_u)
    end

    push!(sim_means, mean(path_utilities))
    push!(sim_ses, std(path_utilities) / sqrt(N_SIMS))
    push!(Vtilde_analytical, V_hom)  # constant in homogeneous case
end

# ══════════════════════════════════════════════════════════════════════════════
# Export CSV for plotting
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

df = DataFrame(
    t_init             = t_inits,
    Vtilde_analytical  = Vtilde_analytical,
    sim_mean           = sim_means,
    sim_se             = sim_ses,
    sim_lower_2se      = sim_means .- 2 .* sim_ses,
    sim_upper_2se      = sim_means .+ 2 .* sim_ses,
)
CSV.write(joinpath(outdir, "simulation_validation.csv"), df)
println("\nWrote simulation_validation.csv ($(nrow(df)) rows)")

# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

@testset "Simulation Validation (Homogeneous)" begin

    @testset "Solver converged" begin
        @test model.converged
    end

    @testset "Simulated E[U] consistent with V* at each starting date" begin
        for (j, t0) in enumerate(t_inits)
            z_score = abs(sim_means[j] - V_hom) / sim_ses[j]
            within_3se = z_score < 3.0  # 3SE for multiple comparisons across 12 dates
            println("  t₀=$(round(t0; digits=0)): " *
                    "E[U]=$(round(sim_means[j]; digits=2)), " *
                    "V*=$(round(V_hom; digits=2)), " *
                    "SE=$(round(sim_ses[j]; digits=2)), " *
                    "|z|=$(round(z_score; digits=2))" *
                    (within_3se ? "" : " ← OUTSIDE 3SE"))
            @test within_3se
        end
    end

    @testset "Overall mean close to V*" begin
        overall_mean = mean(sim_means)
        overall_se = mean(sim_ses) / sqrt(length(t_inits))  # SE of the grand mean
        rel_err = abs(overall_mean - V_hom) / abs(V_hom)
        println("  Grand mean E[U] = $(round(overall_mean; digits=2))")
        println("  V*               = $(round(V_hom; digits=2))")
        println("  Relative error   = $(round(rel_err * 100; digits=4))%")
        @test rel_err < 0.05  # within 5%
    end

    @testset "Loss rate approximately matches hazard" begin
        # In homogeneous case, P(loss in cycle) = 1 - S(0, T*) = 1 - exp(-λ·T*)
        expected_loss_rate = 1 - exp(-λ_const * T_star_hom)

        # Gather empirical loss rate from last simulation run
        all_paths = simulate_production_cycles(model, hom_seasonal_params;
            n_cycles = N_CYCLES, n_sims = N_SIMS, t_init = 0.0, seed = SEED,
        )
        n_losses = sum(o.loss for path in all_paths for o in path)
        n_total = N_SIMS * N_CYCLES
        empirical_loss_rate = n_losses / n_total

        println("  Expected loss rate = $(round(expected_loss_rate; digits=4))")
        println("  Empirical loss rate = $(round(empirical_loss_rate; digits=4))")
        @test abs(empirical_loss_rate - expected_loss_rate) < 0.03
    end
end

println("\n── Summary ─────────────────────────────────────")
println("  V* (analytical)     = $(round(V_hom; digits=2))")
println("  E[U] (sim, mean)    = $(round(mean(sim_means); digits=2))")
println("  E[U] (sim, SE)      = $(round(mean(sim_ses); digits=2))")
println("  Relative error      = $(round(abs(mean(sim_means) - V_hom) / abs(V_hom) * 100; digits=2))%")
