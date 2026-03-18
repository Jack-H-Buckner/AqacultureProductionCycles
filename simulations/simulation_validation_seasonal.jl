"""
    simulation_validation_seasonal.jl

Validate the Monte Carlo simulation against the solver's continuation value
for the medium-risk seasonal (inhomogeneous) case.

Unlike the homogeneous validation where V is constant, the seasonal Ṽ(t₀)
varies with starting date. At each of 12 monthly starting dates, we compare
the simulated expected present utility (discounted sum of per-cycle utilities
+ terminal continuation value) to the solver's Ṽ(t₀).

Outputs:
- `simulation_validation_seasonal.csv` — analytical Ṽ, simulated mean, SE at
  each starting date
"""

using CSV, DataFrames, Statistics, Random

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "04_simulate_production_cycles.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# 1. Solve the medium-risk seasonal model via homotopy continuation
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

println("\nSolving seasonal model (medium risk, N=$N_solver)...")
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

println("\nModel solved (converged = $(result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Simulate from multiple starting dates
# ══════════════════════════════════════════════════════════════════════════════

N_CYCLES = 100
N_SIMS   = 1500
t_inits  = collect(0.0:30.0:330.0)

println("\nSimulating $N_SIMS paths × $N_CYCLES cycles at $(length(t_inits)) starting dates...")

sim_means = Float64[]
sim_ses   = Float64[]
Vtilde_analytical = Float64[]

for (j, t0_init) in enumerate(t_inits)
    println("  t₀ = $(round(t0_init; digits=0))...")

    all_paths = simulate_production_cycles(result, p_seasonal;
        n_cycles = N_CYCLES,
        n_sims   = N_SIMS,
        t_init   = t0_init,
        seed     = SEED + j,
    )

    # Compute path-level discounted utility sum relative to t_init,
    # plus terminal continuation value to remove truncation bias
    path_utilities = Float64[]
    for path in all_paths
        path_u = 0.0
        for outcome in path
            Y = max(outcome.income, 1e-10)
            path_u += exp(-p_seasonal.δ * (outcome.t_end - t0_init)) * u(Y, p_seasonal)
        end
        # Terminal value
        t_last = path[end].t_end
        V_terminal = spline_eval(t_last, result.V_coeffs)
        path_u += exp(-p_seasonal.δ * (t_last - t0_init)) * V_terminal
        push!(path_utilities, path_u)
    end

    push!(sim_means, mean(path_utilities))
    push!(sim_ses, std(path_utilities) / sqrt(N_SIMS))

    # Analytical Ṽ(t₀) — accounts for fallow: V(t) = e^{-δd*}·Ṽ(t₀*)
    # At the start of a path, we're at end-of-cycle time t, so the relevant
    # value is V(t) which includes any fallow discount.
    V_t0 = spline_eval(t0_init, result.V_coeffs)
    push!(Vtilde_analytical, V_t0)
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Export CSV
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

df = DataFrame(
    t_init            = t_inits,
    V_analytical      = Vtilde_analytical,
    sim_mean          = sim_means,
    sim_se            = sim_ses,
    sim_lower_2se     = sim_means .- 2 .* sim_ses,
    sim_upper_2se     = sim_means .+ 2 .* sim_ses,
)
CSV.write(joinpath(outdir, "simulation_validation_seasonal.csv"), df)
println("\nWrote simulation_validation_seasonal.csv ($(nrow(df)) rows)")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n── Validation summary (medium risk, seasonal) ──────────")
for (j, t0) in enumerate(t_inits)
    z = abs(sim_means[j] - Vtilde_analytical[j]) / sim_ses[j]
    flag = z > 3.0 ? " ← OUTSIDE 3SE" : ""
    println("  t₀=$(round(t0; digits=0)): " *
            "V=$(round(Vtilde_analytical[j]; digits=0)), " *
            "E[U]=$(round(sim_means[j]; digits=0)), " *
            "SE=$(round(sim_ses[j]; digits=0)), " *
            "|z|=$(round(z; digits=2))$flag")
end

overall_rel_err = mean(abs.(sim_means .- Vtilde_analytical) ./ abs.(Vtilde_analytical))
println("\n  Mean absolute relative error = $(round(overall_rel_err * 100; digits=4))%")
