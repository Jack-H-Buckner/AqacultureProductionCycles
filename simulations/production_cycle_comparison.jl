"""
    production_cycle_comparison.jl

Simulate a single sample path of 10 production cycles under three risk
scenarios (baseline, medium, high) and export time series for a three-column
comparison plot.

Outputs:
- `production_cycle_comparison_ts.csv`  — dense within-cycle time series
- `production_cycle_comparison_ep.csv`  — per-cycle endpoint data
"""

using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "04_simulate_production_cycles.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── Scenario definitions ──────────────────────────────────────────────────────

scenarios = [
    (name = "Baseline",    λ_coeffs = λ_coeffs),
    (name = "Medium risk", λ_coeffs = λ_medium_coeffs),
    (name = "High risk",   λ_coeffs = λ_high_coeffs),
]

# ── Homotopy helper ───────────────────────────────────────────────────────────

function scale_seasonal_params(p, α)
    scale_coeffs(c) = (a0 = c.a0, a = α .* c.a, b = α .* c.b)
    return merge(p, (
        λ_coeffs = scale_coeffs(p.λ_coeffs),
        m_coeffs = scale_coeffs(p.m_coeffs),
        k_coeffs = scale_coeffs(p.k_coeffs),
    ))
end

# ── Solve and simulate each scenario ─────────────────────────────────────────

N_solver = 10
α_steps = [0.25, 0.5, 0.75, 1.0]
N_CYCLES = 10

all_ts_dfs = DataFrame[]
all_ep_dfs = DataFrame[]

for sc in scenarios
    println("\n══════════════════════════════════════════════════════")
    println("  Scenario: $(sc.name)")
    println("══════════════════════════════════════════════════════")

    # Build parameter set
    p_seasonal = merge(default_params, (
        λ_coeffs = sc.λ_coeffs,
        γ        = 0.5,
        Y_MIN    = 0.0,
    ))

    # Homogeneous warm start
    p_hom = merge(homogeneous_params, (
        λ_const = exp(sc.λ_coeffs.a0),
        γ       = 0.5,
        Y_MIN   = 0.0,
    ))

    T_hom = solve_insurance(p_hom)
    I_hom = solve_indemnity_homogeneous(T_hom, p_hom)
    V_hom = insurance_value(T_hom, I_hom, p_hom)
    println("  T*_hom = $(round(T_hom; digits=1)), V*_hom = $(round(V_hom; digits=0))")

    # Homotopy continuation
    V_coeffs = initialize_V_constant(V_hom; N=N_solver)
    result = nothing

    for (i, α) in enumerate(α_steps)
        p_scaled = scale_seasonal_params(p_seasonal, α)
        println("  Homotopy $i/$(length(α_steps)): α = $α")

        result = solve_seasonal_model(p_scaled;
            N        = N_solver,
            V_init   = V_coeffs,
            max_iter = 200,
            tol      = 1e-4,
            damping  = 0.5,
            verbose  = false,
        )
        V_coeffs = result.V_coeffs
    end

    println("  Solver converged = $(result.converged)")

    # Simulate one path
    path_data = simulate_path_timeseries(result, p_seasonal;
        n_cycles = N_CYCLES,
        t_init   = 0.0,
        seed     = SEED,
        n_pts    = 50,
    )

    # Build DataFrames with scenario label
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
        scenario     = fill(sc.name, length(ts.t)),
    )
    push!(all_ts_dfs, ts_df)

    ep = path_data.cycle_endpoints
    ep_df = DataFrame(
        t_end    = ep.t_end,
        cycle    = ep.cycle,
        loss     = ep.loss,
        income   = ep.income,
        utility  = ep.utility,
        d_star   = ep.d_star,
        scenario = fill(sc.name, length(ep.cycle)),
    )
    push!(all_ep_dfs, ep_df)

    # Print summary
    n_losses = count(ep.loss)
    println("  Cycles: $N_CYCLES, Losses: $n_losses")
    for i in 1:length(ep.cycle)
        event = ep.loss[i] ? "LOSS" : "harvest"
        println("    Cycle $(ep.cycle[i]): $event, d*=$(round(ep.d_star[i]; digits=1)), " *
                "income=$(round(ep.income[i]; digits=0))")
    end
end

# ── Export ────────────────────────────────────────────────────────────────────

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

ts_all = vcat(all_ts_dfs...)
CSV.write(joinpath(outdir, "production_cycle_comparison_ts.csv"), ts_all)
println("\nWrote production_cycle_comparison_ts.csv ($(nrow(ts_all)) rows)")

ep_all = vcat(all_ep_dfs...)
CSV.write(joinpath(outdir, "production_cycle_comparison_ep.csv"), ep_all)
println("Wrote production_cycle_comparison_ep.csv ($(nrow(ep_all)) rows)")
