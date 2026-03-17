"""
    high_risk_seasonal.jl

High-risk scenario. Increases the catastrophic hazard rate λ while keeping
mortality (m) and growth (k) at baseline values.

Runs both optimal fallow and forced no-fallow (d*=0) cases and exports:
- `high_risk_seasonal_grid.csv`    — fine-grid evaluation for both cases
- `high_risk_seasonal_nodes.csv`   — nodal values for both cases
- `high_risk_seasonal_params.csv`  — seasonal parameter functions on a daily grid
"""

using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── High-risk seasonal parameters ─────────────────────────────────────────────
# Only the catastrophic hazard rate differs from baseline.
# Mortality (m) and growth (k) use baseline values from parameters.jl.

high_risk_params = merge(default_params, (
    λ_coeffs = λ_high_coeffs,
    γ        = 0.5,
    Y_MIN    = 0.0,
))

# ── Export seasonal parameter curves ──────────────────────────────────────────

println("Exporting seasonal parameter curves...")
days = 0:364
param_df = DataFrame(
    t            = collect(days),
    k            = [k_growth(t, high_risk_params) for t in days],
    m            = [m_rate(t, high_risk_params) for t in days],
    lambda       = [λ(t, high_risk_params) for t in days],
    k_baseline   = [k_growth(t, default_params) for t in days],
    m_baseline   = [m_rate(t, default_params) for t in days],
    lambda_baseline = [λ(t, default_params) for t in days],
)

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)
CSV.write(joinpath(outdir, "high_risk_seasonal_params.csv"), param_df)
println("Wrote high_risk_seasonal_params.csv")

# ── Homotopy helper ──────────────────────────────────────────────────────────

function scale_seasonal_params(p, α)
    scale_coeffs(c) = (a0 = c.a0, a = α .* c.a, b = α .* c.b)
    return merge(p, (
        λ_coeffs = scale_coeffs(p.λ_coeffs),
        m_coeffs = scale_coeffs(p.m_coeffs),
        k_coeffs = scale_coeffs(p.k_coeffs),
    ))
end

# ── Warm start from homogeneous solution ──────────────────────────────────────

high_risk_hom = merge(homogeneous_params, (
    λ_const = exp(λ_high_coeffs.a0),
    γ       = 0.5,
    Y_MIN   = 0.0,
))

println("\nComputing homogeneous solution for warm start...")
T_hom = solve_insurance(high_risk_hom)
I_hom = solve_indemnity_homogeneous(T_hom, high_risk_hom)
V_hom = insurance_value(T_hom, I_hom, high_risk_hom)
println("  T* = $(round(T_hom; digits=1)) days, V* = $(round(V_hom; digits=0))")

N_solver = 40
all_grid_dfs = DataFrame[]
all_node_dfs = DataFrame[]

for (fallow_label, no_fallow) in [("optimal_fallow", false), ("no_fallow", true)]
    println("\n══════════════════════════════════════════════════════")
    println("  Case: $fallow_label")
    println("══════════════════════════════════════════════════════")

    # ── Homotopy continuation ─────────────────────────────────────────────
    homotopy_steps = [0.25, 0.5, 0.75, 1.0]
    V_coeffs = initialize_V_constant(V_hom; N=N_solver)
    result = nothing

    for (i, α) in enumerate(homotopy_steps)
        p_scaled = scale_seasonal_params(high_risk_params, α)
        println("\n  Homotopy step $i/$(length(homotopy_steps)): α = $α")

        result = solve_seasonal_model(p_scaled;
            N               = N_solver,
            V_init          = V_coeffs,
            max_iter        = 200,
            tol             = 1e-3,
            damping         = 0.5,
            force_no_fallow = no_fallow,
            τ_max           = 800.0,
            verbose         = true,
        )

        V_coeffs = result.V_coeffs
        V_bar = sum(V_coeffs.values) / length(V_coeffs.values)
        τ_bar = sum(result.τ_star_coeffs.values) / length(result.τ_star_coeffs.values)
        println("  → V̄ = $(round(V_bar; digits=2)), " *
                "τ̄ = $(round(τ_bar; digits=1)), " *
                "converged = $(result.converged) ($(result.iterations) iters)")
    end

    # ── Evaluate on fine grid ─────────────────────────────────────────────
    println("\nEvaluating solution on fine grid...")
    eval_result = evaluate_solution(result, high_risk_params; n_grid=200)

    # Compute stocking FOC components on the fine grid
    # The FOC residual is Ṽ'(t₀) - δ·Ṽ(t₀), evaluated at t₀ = t + d*(t)
    Vtilde_prime_grid = Float64[]
    delta_Vtilde_grid = Float64[]
    stocking_resid_grid = Float64[]
    for (i, t) in enumerate(eval_result.t_grid)
        t0 = t + eval_result.d_grid[i]
        Vt_prime = spline_derivative(t0, result.Vtilde_coeffs)
        Vt_val = spline_eval(t0, result.Vtilde_coeffs)
        push!(Vtilde_prime_grid, Vt_prime)
        push!(delta_Vtilde_grid, high_risk_params.δ * Vt_val)
        push!(stocking_resid_grid, Vt_prime - high_risk_params.δ * Vt_val)
    end

    grid_df = DataFrame(
        t              = eval_result.t_grid,
        fallow         = fill(fallow_label, 200),
        V              = eval_result.V_grid,
        Vtilde         = eval_result.Vtilde_grid,
        tau_star        = eval_result.τ_star_grid,
        d_star          = eval_result.d_grid,
        Vtilde_prime    = Vtilde_prime_grid,
        delta_Vtilde    = delta_Vtilde_grid,
        stocking_resid  = stocking_resid_grid,
    )
    push!(all_grid_dfs, grid_df)

    node_df = DataFrame(
        node           = result.nodes,
        fallow         = fill(fallow_label, length(result.nodes)),
        V_at_node      = result.V_values,
        tau_at_node    = result.τ_values,
        d_at_node      = result.d_values,
        t0_at_node     = result.t0_values,
    )
    push!(all_node_dfs, node_df)

    # ── Print summary ─────────────────────────────────────────────────────
    n_corner = count(d -> d == 0.0, result.d_values)
    println("\n── Summary ($fallow_label) ────────────────────")
    println("  Converged: $(result.converged) ($(result.iterations) iterations)")
    println("  τ̄*(t₀) = $(round(sum(result.τ_star_coeffs.values)/length(result.τ_star_coeffs.values); digits=1)) days")
    println("  V̄(t)   = $(round(sum(result.V_coeffs.values)/length(result.V_coeffs.values); digits=2))")
    println("  d̄*(t)   = $(round(mean(result.d_values); digits=2)) days")
    println("  Corner solutions (d*=0): $n_corner / $(length(result.d_values))")
    println("  τ* range: $(round(minimum(result.τ_values); digits=1)) – $(round(maximum(result.τ_values); digits=1)) days")
    println("  V  range: $(round(minimum(result.V_values); digits=0)) – $(round(maximum(result.V_values); digits=0))")
    println("  d* range: $(round(minimum(result.d_values); digits=1)) – $(round(maximum(result.d_values); digits=1)) days")
end

# ── Export ────────────────────────────────────────────────────────────────────

df_grid = vcat(all_grid_dfs...)
CSV.write(joinpath(outdir, "high_risk_seasonal_grid.csv"), df_grid)
println("\nWrote high_risk_seasonal_grid.csv ($(nrow(df_grid)) rows)")

df_nodes = vcat(all_node_dfs...)
CSV.write(joinpath(outdir, "high_risk_seasonal_nodes.csv"), df_nodes)
println("Wrote high_risk_seasonal_nodes.csv ($(nrow(df_nodes)) rows)")
