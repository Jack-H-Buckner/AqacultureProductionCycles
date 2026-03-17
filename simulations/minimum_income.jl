"""
    minimum_income.jl

Compare medium-risk scenario across insurance coverage levels:
Y_MIN = 0 (break-even), 5% of Y_H, and 10% of Y_H.
Each case runs with both optimal fallow and forced no-fallow (d*=0).

Outputs:
- `minimum_income_grid.csv`   — fine-grid evaluation for all cases
- `minimum_income_params.csv` — seasonal parameter functions
- `minimum_income_scalars.csv` — scalar summary (Y_H_hom, Y_MIN values)
"""

using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── Medium-risk scenario parameters (from parameters.jl) ─────────────────────
# Only λ differs from baseline; m and k use baseline values.

p_hom_base = merge(homogeneous_params, (
    λ_const = exp(λ_medium_coeffs.a0),
    γ       = 0.5,
    Y_MIN   = 0.0,
))

T_hom = solve_insurance(p_hom_base)
I_hom = solve_indemnity_homogeneous(T_hom, p_hom_base)
V_hom = insurance_value(T_hom, I_hom, p_hom_base)
Y_H_hom = Y_H_insurance(T_hom, I_hom, p_hom_base)

println("Homogeneous solution (medium risk, Y_MIN=0):")
println("  T*    = $(round(T_hom; digits=1)) days")
println("  V*    = $(round(V_hom; digits=0))")
println("  Y_H   = $(round(Y_H_hom; digits=0))")

Y_MIN_5pct  = 0.05 * Y_H_hom
Y_MIN_10pct = 0.10 * Y_H_hom
println("  Y_MIN (5%)  = $(round(Y_MIN_5pct; digits=0))")
println("  Y_MIN (10%) = $(round(Y_MIN_10pct; digits=0))")

# ── Export seasonal parameter curves ────────────────────────────────────────

println("\nExporting seasonal parameter curves...")
days = 0:364

p_seasonal_base = merge(default_params, (
    λ_coeffs = λ_medium_coeffs,
    γ        = 0.5,
    Y_MIN    = 0.0,
))

param_df = DataFrame(
    t      = collect(days),
    k      = [k_growth(t, p_seasonal_base) for t in days],
    m      = [m_rate(t, p_seasonal_base) for t in days],
    lambda  = [λ(t, p_seasonal_base) for t in days],
)

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)
CSV.write(joinpath(outdir, "minimum_income_params.csv"), param_df)
println("Wrote minimum_income_params.csv")

# Export scalar summary
scalars_df = DataFrame(
    T_hom   = [T_hom],
    V_hom   = [V_hom],
    Y_H_hom = [Y_H_hom],
    Y_MIN_5pct  = [Y_MIN_5pct],
    Y_MIN_10pct = [Y_MIN_10pct],
)
CSV.write(joinpath(outdir, "minimum_income_scalars.csv"), scalars_df)
println("Wrote minimum_income_scalars.csv")

# ── Homotopy helper ─────────────────────────────────────────────────────────

function scale_seasonal_params(p, α)
    scale_coeffs(c) = (a0 = c.a0, a = α .* c.a, b = α .* c.b)
    return merge(p, (
        λ_coeffs = scale_coeffs(p.λ_coeffs),
        m_coeffs = scale_coeffs(p.m_coeffs),
        k_coeffs = scale_coeffs(p.k_coeffs),
    ))
end

# ── Solver settings ─────────────────────────────────────────────────────────

N_solver = 40
homotopy_steps = [0.25, 0.5, 0.75, 1.0]
all_grid_dfs = DataFrame[]

scenarios = [
    (name = "ymin_zero",  Y_MIN_val = 0.0),
    (name = "ymin_5pct",  Y_MIN_val = Y_MIN_5pct),
    (name = "ymin_10pct", Y_MIN_val = Y_MIN_10pct),
]

for sc in scenarios
    # Build seasonal parameter set — only λ differs from baseline
    p_seasonal = merge(default_params, (
        λ_coeffs = λ_medium_coeffs,
        γ        = 0.5,
        Y_MIN    = sc.Y_MIN_val,
    ))

    # Homogeneous warm start with this Y_MIN
    p_hom = merge(homogeneous_params, (
        λ_const = exp(λ_medium_coeffs.a0),
        γ       = 0.5,
        Y_MIN   = sc.Y_MIN_val,
    ))

    T_hom_sc = solve_insurance(p_hom)
    I_hom_sc = solve_indemnity_homogeneous(T_hom_sc, p_hom)
    V_hom_sc = insurance_value(T_hom_sc, I_hom_sc, p_hom)

    for (fallow_label, no_fallow) in [("optimal_fallow", false), ("no_fallow", true)]
        println("\n══════════════════════════════════════════════════════")
        println("  $(sc.name), fallow: $fallow_label")
        println("  Y_MIN = $(round(sc.Y_MIN_val; digits=0))")
        println("  T*_hom = $(round(T_hom_sc; digits=1)), V*_hom = $(round(V_hom_sc; digits=0))")
        println("══════════════════════════════════════════════════════")

        # Homotopy continuation
        V_coeffs = initialize_V_constant(V_hom_sc; N=N_solver)
        result = nothing

        for (i, α) in enumerate(homotopy_steps)
            p_scaled = scale_seasonal_params(p_seasonal, α)
            println("  Homotopy $i/$(length(homotopy_steps)): α = $α")

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
        end

        # Evaluate on fine grid
        println("\nEvaluating on fine grid...")
        eval_result = evaluate_solution(result, p_seasonal; n_grid=200)

        # Stocking FOC components
        Vtilde_prime_grid = Float64[]
        delta_Vtilde_grid = Float64[]
        stocking_resid_grid = Float64[]
        for (i, t) in enumerate(eval_result.t_grid)
            t0 = t + eval_result.d_grid[i]
            Vt_prime = spline_derivative(t0, result.Vtilde_coeffs)
            Vt_val = spline_eval(t0, result.Vtilde_coeffs)
            push!(Vtilde_prime_grid, Vt_prime)
            push!(delta_Vtilde_grid, p_seasonal.δ * Vt_val)
            push!(stocking_resid_grid, Vt_prime - p_seasonal.δ * Vt_val)
        end

        grid_df = DataFrame(
            t              = eval_result.t_grid,
            ymin_case      = fill(sc.name, 200),
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

        # Summary
        n_corner = count(d -> d == 0.0, result.d_values)
        V_bar = sum(result.V_coeffs.values) / length(result.V_coeffs.values)
        τ_bar = sum(result.τ_star_coeffs.values) / length(result.τ_star_coeffs.values)
        println("\n── Summary ($(sc.name), $fallow_label) ──────")
        println("  τ̄* = $(round(τ_bar; digits=1)) days, V̄ = $(round(V_bar; digits=0))")
        println("  τ* range: $(round(minimum(result.τ_values); digits=1)) – $(round(maximum(result.τ_values); digits=1))")
        println("  d* range: $(round(minimum(result.d_values); digits=1)) – $(round(maximum(result.d_values); digits=1))")
        println("  Corner (d*=0): $n_corner / $(length(result.d_values))")
    end
end

# ── Export ──────────────────────────────────────────────────────────────────

df_all = vcat(all_grid_dfs...)
CSV.write(joinpath(outdir, "minimum_income_grid.csv"), df_all)
println("\nWrote minimum_income_grid.csv ($(nrow(df_all)) rows)")
