"""
    fixed_point_validation.jl

After converging the seasonal solver (which uses the f/g linear decomposition),
recompute Ṽ(t₀) and V(t) via a single fixed-point iteration using the full
Bellman equation:

    Ṽ(t₀) = S·e^{-δτ}·[u(Y_H) + V(T*)] + ∫ S·λ·e^{-δs}·[u(Y_L) + V(s)] ds

This evaluates V(s) at every quadrature point inside the loss integral, unlike
the f/g decomposition which collapses it to g·V(T*). Comparing the two reveals
the approximation error introduced by the decomposition.

Runs for N = 10, 20, 40, 60 with both optimal fallow and forced no-fallow cases.

Outputs:
- `fixed_point_validation_grid.csv` — Ṽ and V from both methods on a fine grid,
   with `N` and `fallow` columns identifying the configuration
"""

using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

seasonal_p = merge(default_params, (
    γ     = 0.1,
    Y_MIN = 1000.0,
))

N_values = [10, 20, 40, 60]
n_grid = 200
t_grid = collect(range(0.0, PERIOD * (1 - 1/n_grid), length=n_grid))

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

all_dfs = DataFrame[]

for N in N_values
    for (fallow_label, no_fallow) in [("optimal_fallow", false), ("no_fallow", true)]
        println("\n══════════════════════════════════════════════════════")
        println("  N = $N  ($(2N+1) nodes), fallow = $fallow_label")
        println("══════════════════════════════════════════════════════")

        # ── Solve the seasonal model ─────────────────────────────────────
        println("Solving seasonal model...")
        result = solve_seasonal_model(seasonal_p;
            N               = N,
            max_iter        = 200,
            tol             = 1e-4,
            damping         = 0.5,
            force_no_fallow = no_fallow,
            verbose         = true,
        )
        println("Converged: $(result.converged) in $(result.iterations) iterations")

        V_coeffs = result.V_coeffs
        τ_star_coeffs = result.τ_star_coeffs
        Vtilde_coeffs = result.Vtilde_coeffs

        # ── Compute on a fine grid ───────────────────────────────────────
        println("Computing full Bellman Ṽ and V on $(n_grid)-point grid...")

        V_linear = Float64[]
        Vtilde_linear = Float64[]
        V_bellman = Float64[]
        Vtilde_bellman = Float64[]
        d_vals = Float64[]
        tau_vals = Float64[]

        for t in t_grid
            d_star = interpolate_d_star(t, result.nodes, result.d_values)
            t0_star = t + d_star
            τ_star = spline_eval(t0_star, τ_star_coeffs)
            T_star = t0_star + τ_star

            push!(d_vals, d_star)
            push!(tau_vals, τ_star)
            push!(V_linear, spline_eval(t, V_coeffs))
            push!(Vtilde_linear, spline_eval(t0_star, Vtilde_coeffs))

            Vt_full = compute_Vtilde(t0_star, T_star, V_coeffs, seasonal_p)
            push!(Vtilde_bellman, Vt_full)
            push!(V_bellman, exp(-seasonal_p.δ * d_star) * Vt_full)
        end

        df = DataFrame(
            t               = t_grid,
            N               = fill(N, n_grid),
            nodes           = fill(2N + 1, n_grid),
            fallow          = fill(fallow_label, n_grid),
            d_star          = d_vals,
            tau_star        = tau_vals,
            V_linear        = V_linear,
            V_bellman       = V_bellman,
            Vtilde_linear   = Vtilde_linear,
            Vtilde_bellman  = Vtilde_bellman,
        )
        push!(all_dfs, df)

        V_rel_err = abs.(V_bellman .- V_linear) ./ abs.(V_linear) .* 100
        Vt_rel_err = abs.(Vtilde_bellman .- Vtilde_linear) ./ abs.(Vtilde_linear) .* 100

        println("  V  relative error: max = $(round(maximum(V_rel_err); digits=4))%, mean = $(round(mean(V_rel_err); digits=4))%")
        println("  Ṽ  relative error: max = $(round(maximum(Vt_rel_err); digits=4))%, mean = $(round(mean(Vt_rel_err); digits=4))%")
    end
end

# ── Export combined data ──────────────────────────────────────────────────────

df_all = vcat(all_dfs...)
CSV.write(joinpath(outdir, "fixed_point_validation_grid.csv"), df_all)
println("\nWrote fixed_point_validation_grid.csv ($(nrow(df_all)) rows)")
