"""
    dollar_cv_validation_seasonal.jl

Validate the dollar continuation value W(t) from Stage B against Monte Carlo
simulation for the medium-risk seasonal (inhomogeneous) case.

Unlike the homogeneous validation where W is constant, the seasonal W(t) varies
with calendar date. At each of 12 monthly starting dates, we compare the
simulated expected discounted dollar income (Σ e^{-δt}·Y + terminal W) to the
solver's W(t).

This mirrors simulation_validation_seasonal.jl but replaces u(Y) with Y
(dollar income instead of utility) and V(t) with W(t).

Outputs:
- `dollar_cv_validation_seasonal.csv` — solver W, simulated mean, SE at each
  starting date
"""

using CSV, DataFrames, Statistics, Random

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "04_simulate_production_cycles.jl"))
include(joinpath(@__DIR__, "..", "src", "05_dollar_continuation_value.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# 1. Solve the medium-risk seasonal model via homotopy continuation (Stage A)
# ══════════════════════════════════════════════════════════════════════════════

p_seasonal = merge(default_params, (
    λ_coeffs = λ_medium_coeffs,
    γ        = 0.5,
    Y_MIN    = 0.0,
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

# Homotopy helper
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
result = nothing

println("\nStage A: Solving seasonal model (medium risk, N=$N_solver)...")
for (i, α) in enumerate(homotopy_steps)
    p_scaled = scale_seasonal_params(p_seasonal, α)
    println("  Homotopy $i/$(length(homotopy_steps)): α = $α")

    global result = solve_seasonal_model(p_scaled;
        N        = N_solver,
        V_init   = V_coeffs,
        max_iter = 200,
        tol      = 1e-4,
        damping  = 0.5,
        verbose  = true,
    )

    global V_coeffs = result.V_coeffs
end

println("\nStage A solved (converged = $(result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Solve dollar continuation value W(t) (Stage B)
# ══════════════════════════════════════════════════════════════════════════════

println("\nStage B: Solving dollar continuation value W(t)...")
W_result = solve_dollar_continuation_value(result, p_seasonal;
    N       = N_solver,
    max_iter = 200,
    tol     = 1e-4,
    damping = 0.5,
    verbose = true,
)
println("Stage B solved (converged = $(W_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Simulate from multiple starting dates
# ══════════════════════════════════════════════════════════════════════════════

N_CYCLES = 100
N_SIMS   = 1500
t_inits  = collect(0.0:30.0:330.0)

println("\nSimulating $N_SIMS paths × $N_CYCLES cycles at $(length(t_inits)) starting dates...")

sim_means = Float64[]
sim_ses   = Float64[]
W_solver_values = Float64[]

for (j, t0_init) in enumerate(t_inits)
    println("  t₀ = $(round(t0_init; digits=0))...")

    all_paths = simulate_production_cycles(result, p_seasonal;
        n_cycles = N_CYCLES,
        n_sims   = N_SIMS,
        t_init   = t0_init,
        seed     = SEED + j,
    )

    # Compute path-level discounted DOLLAR income sum relative to t_init.
    # Use Y directly (not u(Y)) — W is the dollar NPV, not utility NPV.
    # Add terminal W(t_end_last) to account for the infinite tail.
    path_dollars = Float64[]
    for path in all_paths
        path_d = 0.0
        for outcome in path
            Y = max(outcome.income, 1e-10)
            path_d += exp(-p_seasonal.δ * (outcome.t_end - t0_init)) * Y
        end
        # Terminal value: W at the end of the last cycle, discounted to t_init
        t_last = path[end].t_end
        W_terminal = spline_eval(t_last, W_result.W_coeffs)
        path_d += exp(-p_seasonal.δ * (t_last - t0_init)) * W_terminal
        push!(path_dollars, path_d)
    end

    push!(sim_means, mean(path_dollars))
    push!(sim_ses, std(path_dollars) / sqrt(N_SIMS))

    # Solver's W(t) at this starting date
    W_t0 = spline_eval(t0_init, W_result.W_coeffs)
    push!(W_solver_values, W_t0)
end

# ══════════════════════════════════════════════════════════════════════════════
# 4. Export CSV
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

df = DataFrame(
    t_init            = t_inits,
    W_solver          = W_solver_values,
    sim_mean          = sim_means,
    sim_se            = sim_ses,
    sim_lower_2se     = sim_means .- 2 .* sim_ses,
    sim_upper_2se     = sim_means .+ 2 .* sim_ses,
)
CSV.write(joinpath(outdir, "dollar_cv_validation_seasonal.csv"), df)
println("\nWrote dollar_cv_validation_seasonal.csv ($(nrow(df)) rows)")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n── Validation summary (medium risk, seasonal, dollar CV) ──────────")
for (j, t0) in enumerate(t_inits)
    z = abs(sim_means[j] - W_solver_values[j]) / sim_ses[j]
    flag = z > 3.0 ? " ← OUTSIDE 3SE" : ""
    println("  t₀=$(round(t0; digits=0)): " *
            "W=$(round(W_solver_values[j]; digits=0)), " *
            "E[Y]=$(round(sim_means[j]; digits=0)), " *
            "SE=$(round(sim_ses[j]; digits=0)), " *
            "|z|=$(round(z; digits=2))$flag")
end

overall_rel_err = mean(abs.(sim_means .- W_solver_values) ./ abs.(W_solver_values))
println("\n  Mean absolute relative error = $(round(overall_rel_err * 100; digits=4))%")
