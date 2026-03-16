"""
    homogeneous_validation.jl

Run the full seasonal solver with constant (homogeneous) parameters and export
comparison data against the analytical homogeneous solution. Outputs CSV files
for plotting in R.

Outputs:
- `homogeneous_validation_grid.csv` — V(t), Ṽ(t₀), τ*(t₀), d*(t) on a fine grid
- `homogeneous_validation_nodes.csv` — nodal values from the seasonal solver
- `homogeneous_validation_scalars.csv` — scalar benchmarks (T*, V*, Ṽ*)
"""

using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── Homogeneous analytical solution ──────────────────────────────────────────

println("Solving homogeneous case analytically...")
T_star_hom = solve_insurance(homogeneous_params)
I_sol_hom = solve_indemnity_homogeneous(T_star_hom, homogeneous_params)
V_hom = insurance_value(T_star_hom, I_sol_hom, homogeneous_params)

# Ṽ = V when d* = 0 (no fallow discounting)
Vtilde_hom = V_hom

println("  T* = $(round(T_star_hom; digits=2)) days")
println("  V* = $(round(V_hom; digits=2))")

# ── Seasonal solver with constant rates ──────────────────────────────────────

# Zero out higher harmonics so seasonal functions reduce to constants
hom_seasonal_params = merge(default_params, (
    λ_coeffs = (a0 = log(λ_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    m_coeffs = (a0 = log(m_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    k_coeffs = (a0 = log(k_const), a = [0.0, 0.0], b = [0.0, 0.0]),
))

N_solver = 10

# Warm-start with homogeneous V
V_init = initialize_V_constant(V_hom; N=N_solver)

println("\nRunning seasonal solver (N=$N_solver)...")
result = solve_seasonal_model(hom_seasonal_params;
    N = N_solver,
    V_init = V_init,
    max_iter = 100,
    tol = 1e-4,
    damping = 0.5,
    verbose = true,
)

# ── Evaluate on fine grid ────────────────────────────────────────────────────

println("\nEvaluating solution on fine grid...")
eval_result = evaluate_solution(result, hom_seasonal_params; n_grid=200)

# ── Export data ──────────────────────────────────────────────────────────────

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

# Fine-grid data
grid_df = DataFrame(
    t              = eval_result.t_grid,
    V_seasonal     = eval_result.V_grid,
    Vtilde_seasonal = eval_result.Vtilde_grid,
    tau_seasonal    = eval_result.τ_star_grid,
    d_seasonal      = eval_result.d_grid,
    V_homogeneous   = fill(V_hom, length(eval_result.t_grid)),
    Vtilde_homogeneous = fill(Vtilde_hom, length(eval_result.t_grid)),
    tau_homogeneous = fill(T_star_hom, length(eval_result.t_grid)),
    d_homogeneous   = fill(0.0, length(eval_result.t_grid)),
)
CSV.write(joinpath(outdir, "homogeneous_validation_grid.csv"), grid_df)
println("Wrote homogeneous_validation_grid.csv ($(nrow(grid_df)) rows)")

# Nodal data
node_df = DataFrame(
    node            = result.nodes,
    V_at_node       = result.V_values,
    Vtilde_at_node  = result.Vtilde_at_V_nodes,
    tau_at_node     = result.τ_values,
    d_at_node       = result.d_values,
)
CSV.write(joinpath(outdir, "homogeneous_validation_nodes.csv"), node_df)
println("Wrote homogeneous_validation_nodes.csv ($(nrow(node_df)) rows)")

# Scalar benchmarks
scalar_df = DataFrame(
    quantity  = ["T_star", "V_star", "Vtilde_star", "iterations", "converged"],
    homogeneous = [T_star_hom, V_hom, Vtilde_hom, NaN, NaN],
    seasonal    = [sum(result.τ_star_coeffs.values)/length(result.τ_star_coeffs.values),
                   sum(result.V_coeffs.values)/length(result.V_coeffs.values),
                   sum(result.Vtilde_coeffs.values)/length(result.Vtilde_coeffs.values),
                   result.iterations,
                   result.converged ? 1.0 : 0.0],
)
CSV.write(joinpath(outdir, "homogeneous_validation_scalars.csv"), scalar_df)
println("Wrote homogeneous_validation_scalars.csv")

# Print summary
println("\n── Summary ─────────────────────────────────────")
τ_mean = sum(result.τ_star_coeffs.values) / length(result.τ_star_coeffs.values)
V_mean = sum(result.V_coeffs.values) / length(result.V_coeffs.values)
Vt_mean = sum(result.Vtilde_coeffs.values) / length(result.Vtilde_coeffs.values)
println("  τ*: homogeneous = $(round(T_star_hom; digits=2)), " *
        "seasonal = $(round(τ_mean; digits=2)) " *
        "(error = $(round(abs(τ_mean - T_star_hom) / T_star_hom * 100; digits=4))%)")
println("  V:  homogeneous = $(round(V_hom; digits=2)), " *
        "seasonal = $(round(V_mean; digits=2)) " *
        "(error = $(round(abs(V_mean - V_hom) / abs(V_hom) * 100; digits=4))%)")
println("  Ṽ:  homogeneous = $(round(Vtilde_hom; digits=2)), " *
        "seasonal = $(round(Vt_mean; digits=2)) " *
        "(error = $(round(abs(Vt_mean - Vtilde_hom) / abs(Vtilde_hom) * 100; digits=4))%)")
println("  d*: all corner = $(all(d -> d == 0.0, result.d_values))")
