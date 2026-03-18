"""
    production_cycle_timeseries.jl

Simulate a single sample path of 10 production cycles under the high-risk
seasonal optimal policy and export dense within-cycle time series for plotting.

Outputs:
- `production_cycle_timeseries.csv`  — dense within-cycle data (biomass, value,
  costs, hazard, premiums) at 50 points per cycle
- `production_cycle_endpoints.csv`   — per-cycle endpoint data (income, utility,
  loss indicator, fallow)
"""

using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "04_simulate_production_cycles.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── Solve the high-risk seasonal model ────────────────────────────────────────

println("Solving high-risk seasonal model...")

# Build high-risk parameter set — only λ differs from baseline
p_seasonal = merge(default_params, (
    λ_coeffs = λ_high_coeffs,
    γ        = 0.5,
    Y_MIN    = 0.0,
))

# Homogeneous warm start with high-risk λ
p_hom = merge(homogeneous_params, (
    λ_const = exp(λ_high_coeffs.a0),
    γ       = 0.5,
    Y_MIN   = 0.0,
))

T_star_hom = solve_insurance(p_hom)
I_sol_hom = solve_indemnity_homogeneous(T_star_hom, p_hom)
V_hom = insurance_value(T_star_hom, I_sol_hom, p_hom)
println("  T*_hom = $(round(T_star_hom; digits=1)) days, V*_hom = $(round(V_hom; digits=0))")

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
α_steps = [0.25, 0.5, 0.75, 1.0]
V_current = initialize_V_constant(V_hom; N=N_solver)
result = nothing

for (step, α) in enumerate(α_steps)
    p_scaled = scale_seasonal_params(p_seasonal, α)

    println("  Homotopy step $step/$(length(α_steps)): α = $α")
    global result = solve_seasonal_model(p_scaled;
        N = N_solver,
        V_init = V_current,
        max_iter = 200,
        tol = 1e-4,
        damping = 0.5,
        verbose = true,
    )
    global V_current = result.V_coeffs
end

println("\nModel solved (converged = $(result.converged))")

# ── Simulate one sample path of 10 cycles ────────────────────────────────────

println("\nSimulating 10 production cycles (high risk)...")
path_data = simulate_path_timeseries(result, p_seasonal;
    n_cycles = 10,
    t_init   = 0.0,
    seed     = SEED,
    n_pts    = 50,
)

# ── Export CSV ────────────────────────────────────────────────────────────────

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

ts = path_data.timeseries
ts_df = DataFrame(
    t            = ts.t,
    cycle        = ts.cycle,
    biomass_kg   = ts.biomass_kg,
    stock_value  = ts.stock_value,
    feed_cost    = ts.feed_cost,
    premium_cost = ts.premium_cost,
    premium_rate = ts.premium_rate,
    hazard_rate  = ts.hazard_rate,
    indemnity    = ts.indemnity,
    fallow_days  = ts.fallow_days,
    phase        = ts.phase,
)
CSV.write(joinpath(outdir, "production_cycle_timeseries.csv"), ts_df)
println("Wrote production_cycle_timeseries.csv ($(nrow(ts_df)) rows)")

ep = path_data.cycle_endpoints
ep_df = DataFrame(
    t_end   = ep.t_end,
    cycle   = ep.cycle,
    loss    = ep.loss,
    income  = ep.income,
    utility = ep.utility,
    d_star  = ep.d_star,
)
CSV.write(joinpath(outdir, "production_cycle_endpoints.csv"), ep_df)
println("Wrote production_cycle_endpoints.csv ($(nrow(ep_df)) rows)")

# Print summary
println("\n── Cycle summary (high risk) ────────────────────")
for i in 1:length(ep.cycle)
    event = ep.loss[i] ? "LOSS" : "harvest"
    println("  Cycle $(ep.cycle[i]): t_end = $(round(ep.t_end[i]; digits=1)), " *
            "$event, d* = $(round(ep.d_star[i]; digits=1)), " *
            "income = $(round(ep.income[i]; digits=0)), " *
            "u(Y) = $(round(ep.utility[i]; digits=0))")
end
