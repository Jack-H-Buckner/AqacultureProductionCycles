"""
    seasonal_baseline.jl

Run the full seasonal solver with the default parameter set using homotopy
continuation: gradually increase seasonal amplitude from 0 (homogeneous) to 1
(full seasonal), using each converged solution as the warm start for the next.

Outputs:
- `seasonal_baseline_grid.csv` — V(t), Ṽ(t₀), τ*(t₀), d*(t) on a fine grid
- `seasonal_baseline_nodes.csv` — nodal values from the seasonal solver
- `seasonal_baseline_scalars.csv` — scalar summary (mean values, convergence)
"""

using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── Helper: scale seasonal amplitudes ──────────────────────────────────────────

"""
    scale_seasonal_params(p, α)

Return a copy of parameters `p` with seasonal harmonic amplitudes scaled by `α`.
At α=0, all rates are constant (homogeneous). At α=1, full seasonality.
"""
function scale_seasonal_params(p, α)
    scale_coeffs(c) = (a0 = c.a0, a = α .* c.a, b = α .* c.b)
    return merge(p, (
        λ_coeffs = scale_coeffs(p.λ_coeffs),
        m_coeffs = scale_coeffs(p.m_coeffs),
        k_coeffs = scale_coeffs(p.k_coeffs),
    ))
end

# ── Warm start from homogeneous solution ─────────────────────────────────────

println("Computing homogeneous solution for warm start...")
T_hom = solve_insurance(homogeneous_params)
I_hom = solve_indemnity_homogeneous(T_hom, homogeneous_params)
V_hom = insurance_value(T_hom, I_hom, homogeneous_params)
println("  T* = $(round(T_hom; digits=1)) days, V* = $(round(V_hom; digits=0))")

N_solver = 10

# ── Homotopy continuation ─────────────────────────────────────────────────────

homotopy_steps = [0.25, 0.5, 0.75, 1.0]
V_coeffs = initialize_V_constant(V_hom; N=N_solver)

result = nothing
for (i, α) in enumerate(homotopy_steps)
    global result, V_coeffs
    p_scaled = scale_seasonal_params(default_params, α)
    println("\n══ Homotopy step $i/$(length(homotopy_steps)): α = $α ══")

    result = solve_seasonal_model(p_scaled;
        N = N_solver,
        V_init = V_coeffs,
        max_iter = 50,
        tol = 1e-3,
        damping = 0.3,
        τ_max = 500.0,
        verbose = true,
    )

    V_coeffs = result.V_coeffs
    V_bar = sum(V_coeffs.values) / length(V_coeffs.values)
    τ_bar = sum(result.τ_star_coeffs.values) / length(result.τ_star_coeffs.values)
    println("  → V̄ = $(round(V_bar; digits=2)), " *
            "τ̄ = $(round(τ_bar; digits=1)), " *
            "converged = $(result.converged) ($(result.iterations) iters)")
end

# ── Evaluate final solution on fine grid ───────────────────────────────────────

println("\nEvaluating solution on fine grid...")
eval_result = evaluate_solution(result, default_params; n_grid=200)

# ── Export data ──────────────────────────────────────────────────────────────

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

# Fine-grid data
grid_df = DataFrame(
    t               = eval_result.t_grid,
    V               = eval_result.V_grid,
    Vtilde          = eval_result.Vtilde_grid,
    tau_star         = eval_result.τ_star_grid,
    d_star           = eval_result.d_grid,
    V_recomputed     = eval_result.V_recomputed_grid,
    Vtilde_linkage   = eval_result.Vtilde_linkage_grid,
)
CSV.write(joinpath(outdir, "seasonal_baseline_grid.csv"), grid_df)
println("Wrote seasonal_baseline_grid.csv ($(nrow(grid_df)) rows)")

# Nodal data
node_df = DataFrame(
    node            = result.nodes,
    V_at_node       = result.V_values,
    Vtilde_at_node  = result.Vtilde_at_V_nodes,
    tau_at_node     = result.τ_values,
    d_at_node       = result.d_values,
    t0_at_node      = result.t0_values,
)
CSV.write(joinpath(outdir, "seasonal_baseline_nodes.csv"), node_df)
println("Wrote seasonal_baseline_nodes.csv ($(nrow(node_df)) rows)")

# Scalar summary
scalar_df = DataFrame(
    quantity  = ["tau_mean", "V_mean", "Vtilde_mean", "d_mean",
                 "iterations", "converged"],
    value     = [sum(result.τ_star_coeffs.values)/length(result.τ_star_coeffs.values),
                 sum(result.V_coeffs.values)/length(result.V_coeffs.values),
                 sum(result.Vtilde_coeffs.values)/length(result.Vtilde_coeffs.values),
                 mean(result.d_values),
                 result.iterations, result.converged ? 1.0 : 0.0],
)
CSV.write(joinpath(outdir, "seasonal_baseline_scalars.csv"), scalar_df)
println("Wrote seasonal_baseline_scalars.csv")

# Print summary
n_corner = count(d -> d == 0.0, result.d_values)
println("\n── Summary ─────────────────────────────────────")
println("  Converged: $(result.converged) ($(result.iterations) iterations)")
println("  τ̄*(t₀) = $(round(sum(result.τ_star_coeffs.values)/length(result.τ_star_coeffs.values); digits=1)) days")
println("  V̄(t)   = $(round(sum(result.V_coeffs.values)/length(result.V_coeffs.values); digits=2))")
println("  Ṽ̄(t₀)  = $(round(sum(result.Vtilde_coeffs.values)/length(result.Vtilde_coeffs.values); digits=2))")
println("  d̄*(t)   = $(round(mean(result.d_values); digits=2)) days")
println("  Corner solutions (d*=0): $n_corner / $(length(result.d_values))")
println("  τ* range: $(round(minimum(result.τ_values); digits=1)) – $(round(maximum(result.τ_values); digits=1)) days")
println("  V  range: $(round(minimum(result.V_values); digits=0)) – $(round(maximum(result.V_values); digits=0))")
println("  d* range: $(round(minimum(result.d_values); digits=1)) – $(round(maximum(result.d_values); digits=1)) days")
