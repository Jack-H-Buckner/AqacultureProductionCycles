"""
    profit_coverage_comparison.jl

Compare the continuation value V(t), optimal policy τ*(t₀), and fallow d*(t)
under breakeven insurance (ξ=0), small profit coverage (ξ=0.001), and moderate
profit coverage (ξ=0.25) for the baseline (low seasonal risk) scenario.

Runs the full four-stage pipeline:
  Stage A: risk-averse model with ξ=0 → V⁰(t), τ*⁰, d*⁰
  Stage B: dollar continuation value W(t)
  Stage C+D: re-solve with profit-coverage payoffs at ξ=0.001 and ξ=0.25

Exports fine-grid evaluations of all solutions for side-by-side plotting.

Outputs:
- `profit_coverage_comparison.csv` — fine-grid V, Ṽ, τ*, d* for all ξ values
- `profit_coverage_comparison_nodes.csv` — nodal values for all ξ values
"""

using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "07_continuation_value_solver_with_profit_coverage.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# 1. Stage A: breakeven baseline (ξ=0, low risk)
# ══════════════════════════════════════════════════════════════════════════════

p_seasonal = merge(default_params, (
    λ_coeffs = λ_coeffs,   # baseline low risk
    γ        = 0.5,
    Y_MIN    = 0.0,
    ξ        = 0.0,
))

# Homogeneous warm start
p_hom = merge(homogeneous_params, (
    λ_const = exp(λ_coeffs.a0),
    γ       = 0.5,
    Y_MIN   = 0.0,
))

println("Solving homogeneous warm start (baseline risk)...")
T_hom = solve_insurance(p_hom)
I_hom = solve_indemnity_homogeneous(T_hom, p_hom)
V_hom = insurance_value(T_hom, I_hom, p_hom)
println("  T* = $(round(T_hom; digits=1)) days, V* = $(round(V_hom; digits=0))")

# Homotopy
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
model_result = nothing

println("\nStage A: Solving seasonal model (baseline risk)...")
for (i, α) in enumerate(homotopy_steps)
    p_scaled = scale_seasonal_params(p_seasonal, α)
    println("  Homotopy $i/$(length(homotopy_steps)): α = $α")

    global model_result = solve_seasonal_model(p_scaled;
        N        = N_solver,
        V_init   = V_coeffs,
        max_iter = 25,
        tol      = 1e-3,
        damping  = 0.5,
        verbose  = true,
    )

    global V_coeffs = model_result.V_coeffs
end

println("Stage A solved (converged = $(model_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Stage B: dollar continuation value W(t)
# ══════════════════════════════════════════════════════════════════════════════

println("\nStage B: Solving dollar continuation value W(t)...")
W_result = solve_dollar_continuation_value(model_result, p_seasonal;
    N       = N_solver,
    max_iter = 50,
    tol     = 1e-3,
    damping = 0.5,
    verbose = true,
)
W_coeffs = W_result.W_coeffs
println("Stage B solved (converged = $(W_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Stage D: re-solve with profit coverage ξ=0.001
# ══════════════════════════════════════════════════════════════════════════════

ξ_val = 0.001
p_xi = merge(p_seasonal, (ξ = ξ_val,))

println("\nStage D: Solving with profit coverage (ξ=$ξ_val)...")
stage_D_result = solve_stage_D(model_result, W_coeffs, p_xi;
    N        = N_solver,
    max_iter = 50,
    tol      = 1e-3,
    damping  = 0.5,
    verbose  = true,
)
println("Stage D (ξ=0.001) solved (converged = $(stage_D_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 3b. Stage D: re-solve with profit coverage ξ=0.25 (using ξ-continuation)
# ══════════════════════════════════════════════════════════════════════════════

ξ_val2 = 0.25
p_xi2 = merge(p_seasonal, (ξ = ξ_val2,))

println("\nStage D: Solving with profit coverage (ξ=$ξ_val2) via continuation...")
cont_result = solve_stage_D_continuation(model_result, W_coeffs, p_xi2;
    ξ_target             = ξ_val2,
    n_steps              = 5,
    N                    = N_solver,
    max_iter             = 25,
    tol                  = 1e-3,
    damping              = 0.5,
    verbose              = true,
)
stage_D_result2 = cont_result.result
println("Stage D (ξ=$ξ_val2) solved (converged = $(stage_D_result2.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluate all solutions on a fine grid
# ══════════════════════════════════════════════════════════════════════════════

n_grid = 200

println("\nEvaluating solutions on fine grid ($n_grid points)...")

# Stage A evaluation
eval_A = evaluate_solution(model_result, p_seasonal; n_grid=n_grid)

# Stage D ξ=0.001 evaluation
eval_D = evaluate_solution(stage_D_result, p_xi; n_grid=n_grid)

# Stage D ξ=0.25 evaluation
eval_D2 = evaluate_solution(stage_D_result2, p_xi2; n_grid=n_grid)

# W(t) on fine grid
W_grid = [spline_eval(t, W_coeffs) for t in eval_A.t_grid]

# ══════════════════════════════════════════════════════════════════════════════
# 5. Export CSV
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

# Fine-grid comparison
df = DataFrame(
    t               = eval_A.t_grid,
    # Stage A (ξ=0)
    V_xi0           = eval_A.V_grid,
    Vtilde_xi0      = eval_A.Vtilde_grid,
    tau_star_xi0    = eval_A.τ_star_grid,
    d_star_xi0      = eval_A.d_grid,
    # Stage D (ξ=0.001)
    V_xi_small      = eval_D.V_grid,
    Vtilde_xi_small = eval_D.Vtilde_grid,
    tau_star_xi_small = eval_D.τ_star_grid,
    d_star_xi_small = eval_D.d_grid,
    # Stage D (ξ=0.25)
    V_xi_mid        = eval_D2.V_grid,
    Vtilde_xi_mid   = eval_D2.Vtilde_grid,
    tau_star_xi_mid = eval_D2.τ_star_grid,
    d_star_xi_mid   = eval_D2.d_grid,
    # Dollar continuation value
    W               = W_grid,
)
CSV.write(joinpath(outdir, "profit_coverage_comparison.csv"), df)
println("Wrote profit_coverage_comparison.csv ($(nrow(df)) rows)")

# Nodal comparison
nodes = model_result.nodes
node_df = DataFrame(
    node              = nodes,
    # Stage A
    V_xi0             = model_result.V_values,
    tau_xi0           = model_result.τ_values,
    d_xi0             = model_result.d_values,
    # Stage D ξ=0.001
    V_xi_small        = stage_D_result.V_values,
    tau_xi_small      = stage_D_result.τ_values,
    d_xi_small        = stage_D_result.d_values,
    # Stage D ξ=0.25
    V_xi_mid          = stage_D_result2.V_values,
    tau_xi_mid        = stage_D_result2.τ_values,
    d_xi_mid          = stage_D_result2.d_values,
    # W at nodes
    W_node            = [spline_eval(t, W_coeffs) for t in nodes],
)
CSV.write(joinpath(outdir, "profit_coverage_comparison_nodes.csv"), node_df)
println("Wrote profit_coverage_comparison_nodes.csv ($(nrow(node_df)) rows)")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n── Comparison summary (baseline risk) ──────────────────────")
println("  ξ = 0 (breakeven):   V̄ = $(round(mean(eval_A.V_grid); digits=0)), " *
        "τ̄ = $(round(mean(eval_A.τ_star_grid); digits=1)), " *
        "d̄ = $(round(mean(eval_A.d_grid); digits=1))")
println("  ξ = $ξ_val (small):  V̄ = $(round(mean(eval_D.V_grid); digits=0)), " *
        "τ̄ = $(round(mean(eval_D.τ_star_grid); digits=1)), " *
        "d̄ = $(round(mean(eval_D.d_grid); digits=1))")
println("  ξ = $ξ_val2 (mid):   V̄ = $(round(mean(eval_D2.V_grid); digits=0)), " *
        "τ̄ = $(round(mean(eval_D2.τ_star_grid); digits=1)), " *
        "d̄ = $(round(mean(eval_D2.d_grid); digits=1))")
println("  W̄ = $(round(mean(W_grid); digits=0))")

ΔV_small = maximum(abs.(eval_A.V_grid .- eval_D.V_grid))
Δτ_small = maximum(abs.(eval_A.τ_star_grid .- eval_D.τ_star_grid))
ΔV_mid = maximum(abs.(eval_A.V_grid .- eval_D2.V_grid))
Δτ_mid = maximum(abs.(eval_A.τ_star_grid .- eval_D2.τ_star_grid))
println("  ξ=0.001: max|ΔV| = $(round(ΔV_small; sigdigits=4)), max|Δτ| = $(round(Δτ_small; sigdigits=4)) days")
println("  ξ=0.25:  max|ΔV| = $(round(ΔV_mid; sigdigits=4)), max|Δτ| = $(round(Δτ_mid; sigdigits=4)) days")
