"""
    indemnity_xi_comparison.jl

Compare the indemnity I(τ) under the baseline formulation (ξ=0) against the
new opportunity-cost formulation with a small ξ (1e-3) to verify that the
new code recovers the old solution as ξ → 0.

For the medium-risk seasonal case, solves:
  Stage A: risk-averse model → policy (τ*, d*)
  Stage B: dollar continuation value W(t) (breakeven insurance)
  Stage C: three sequential ODE solves for both ξ=0 and ξ=1e-3

Evaluates I(τ) and OC(τ) on a dense grid within each cycle for 12 monthly
starting dates and exports the results for plotting.

Outputs:
- `indemnity_xi_comparison.csv` — columns: t0_month, tau, I_baseline, I_xi, OC
"""

using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "06_indemnity_with_opportunity_costs.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# 1. Solve Stage A: medium-risk seasonal model via homotopy
# ══════════════════════════════════════════════════════════════════════════════

p_seasonal = merge(default_params, (
    λ_coeffs = λ_medium_coeffs,
    γ        = 0.5,
    Y_MIN    = 0.0,
    ξ        = 0.0,
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

println("\nStage A: Solving seasonal model (medium risk)...")
for (i, α) in enumerate(homotopy_steps)
    p_scaled = scale_seasonal_params(p_seasonal, α)
    println("  Homotopy $i/$(length(homotopy_steps)): α = $α")

    global model_result = solve_seasonal_model(p_scaled;
        N        = N_solver,
        V_init   = V_coeffs,
        max_iter = 200,
        tol      = 1e-4,
        damping  = 0.5,
        verbose  = true,
    )

    global V_coeffs = model_result.V_coeffs
end

println("Stage A solved (converged = $(model_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Solve Stage B: dollar continuation value W(t)
# ══════════════════════════════════════════════════════════════════════════════

println("\nStage B: Solving dollar continuation value W(t)...")
W_result = solve_dollar_continuation_value(model_result, p_seasonal;
    N       = N_solver,
    max_iter = 200,
    tol     = 1e-4,
    damping = 0.5,
    verbose = true,
)
W_coeffs = W_result.W_coeffs
println("Stage B solved (converged = $(W_result.converged))")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Compare indemnity at ξ=0 vs ξ=1e-3 for 12 monthly starting dates
# ══════════════════════════════════════════════════════════════════════════════

ξ_small = 1e-3
ξ_full = 1.0
p_xi0 = merge(p_seasonal, (ξ = 0.0,))
p_xi_small = merge(p_seasonal, (ξ = ξ_small,))
p_xi_full = merge(p_seasonal, (ξ = ξ_full,))

n_pts = 100  # evaluation points within each cycle

# Output vectors
out_t0_month = Int[]
out_tau = Float64[]
out_t_calendar = Float64[]
out_I_baseline = Float64[]
out_I_xi_small = Float64[]
out_I_xi_full = Float64[]
out_OC = Float64[]
out_Y_H_baseline = Float64[]
out_Y_H_xi_small = Float64[]
out_Y_H_xi_full = Float64[]

t_inits = collect(0.0:30.0:330.0)

println("\nComparing indemnity for $(length(t_inits)) starting dates...")

for (j, t_start) in enumerate(t_inits)
    d_star = interpolate_d_star(t_start, model_result.nodes, model_result.d_values)
    t0 = t_start + d_star
    τ_star = spline_eval(t0, model_result.τ_star_coeffs)
    T_star = t0 + τ_star

    println("  Month $j: t_start=$(round(t_start; digits=0)), " *
            "t₀=$(round(t0; digits=1)), T*=$(round(T_star; digits=1)), " *
            "τ*=$(round(τ_star; digits=1))")

    L_sol = solve_length(t0, T_star, p_seasonal.L₀, p_seasonal)
    n_sol = solve_numbers(t0, T_star, p_seasonal.n₀, p_seasonal)

    # ── Baseline: ξ = 0 ──────────────────────────────────────────────────
    result_baseline = solve_indemnity_with_opportunity_costs(
        t0, T_star, W_coeffs, L_sol, n_sol, p_xi0; verbose=false)

    # ── Small ξ = 1e-3 ───────────────────────────────────────────────────
    result_xi_small = solve_indemnity_with_opportunity_costs(
        t0, T_star, W_coeffs, L_sol, n_sol, p_xi_small; verbose=false)

    # ── Full coverage: ξ = 1 ─────────────────────────────────────────────
    result_xi_full = solve_indemnity_with_opportunity_costs(
        t0, T_star, W_coeffs, L_sol, n_sol, p_xi_full; verbose=true)

    # Evaluate on dense grid
    τ_grid = range(0.0, τ_star, length=n_pts)
    for τ in τ_grid
        t_eval = t0 + τ
        I_base = result_baseline.I_sol(t_eval)
        I_small = result_xi_small.I_sol(t_eval)
        I_full = result_xi_full.I_sol(t_eval)
        oc_val = result_xi_full.OC_sol(t_eval)

        push!(out_t0_month, j)
        push!(out_tau, τ)
        push!(out_t_calendar, t_eval)
        push!(out_I_baseline, I_base)
        push!(out_I_xi_small, I_small)
        push!(out_I_xi_full, I_full)
        push!(out_OC, oc_val)
        push!(out_Y_H_baseline, result_baseline.Y_H_converged)
        push!(out_Y_H_xi_small, result_xi_small.Y_H_converged)
        push!(out_Y_H_xi_full, result_xi_full.Y_H_converged)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 4. Export CSV
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

df = DataFrame(
    t0_month      = out_t0_month,
    tau           = out_tau,
    t_calendar    = out_t_calendar,
    I_baseline    = out_I_baseline,
    I_xi_small    = out_I_xi_small,
    I_xi_full     = out_I_xi_full,
    OC            = out_OC,
    Y_H_baseline  = out_Y_H_baseline,
    Y_H_xi_small  = out_Y_H_xi_small,
    Y_H_xi_full   = out_Y_H_xi_full,
)
CSV.write(joinpath(outdir, "indemnity_xi_comparison.csv"), df)
println("\nWrote indemnity_xi_comparison.csv ($(nrow(df)) rows)")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n── Indemnity comparison summary ──────────────────────────")
println("  ξ values: 0.0, $ξ_small, $ξ_full")
for j in 1:length(t_inits)
    mask = out_t0_month .== j
    I_base_vals = out_I_baseline[mask]
    I_small_vals = out_I_xi_small[mask]
    I_full_vals = out_I_xi_full[mask]
    oc_vals = out_OC[mask]
    oc_t0 = oc_vals[1]
    oc_max = maximum(oc_vals)
    yh_base = out_Y_H_baseline[mask][1]
    yh_full = out_Y_H_xi_full[mask][1]
    max_rel_small = maximum(abs.(I_base_vals .- I_small_vals) ./ max.(abs.(I_base_vals), 1e-10))
    max_rel_full = maximum(abs.(I_base_vals .- I_full_vals) ./ max.(abs.(I_base_vals), 1e-10))
    println("  Month $j (t₀=$(round(t_inits[j]; digits=0))): " *
            "max|ΔI/I| ξ=0.001: $(round(max_rel_small * 100; digits=2))%, " *
            "ξ=1: $(round(max_rel_full * 100; digits=1))%, " *
            "OC(t₀)=$(round(oc_t0; digits=0)), OC_max=$(round(oc_max; digits=0)), " *
            "Y_H: $(round(yh_base; digits=0)) → $(round(yh_full; digits=0))")
end
