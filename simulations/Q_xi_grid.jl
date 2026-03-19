"""
    Q_xi_grid.jl

Grid comparison of insurer profit margin Q ∈ {0.1, 0.25, 0.5, 0.75} and
profit-coverage fraction ξ ∈ {0.0, 0.25, 0.5, 1.0} under medium (intermediate)
seasonal risk.

For each Q value, runs Stage A (breakeven, ξ=0) and Stage B (W(t)), then
loops Stage D over ξ values. Stage A homotopy up to α=0.75 is shared across
all Q values for efficiency.

Outputs:
- `Q_xi_grid.csv` — fine-grid V, τ*, d* for each (Q, ξ) combination
- `Q_xi_grid_policies.csv` — nodal optimal policies for each (Q, ξ)
"""

using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "07_continuation_value_solver_with_profit_coverage.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ══════════════════════════════════════════════════════════════════════════════
# 0. Grid definition
# ══════════════════════════════════════════════════════════════════════════════

Q_values = [0.1, 0.25, 0.5, 0.75]
ξ_values = [0.0, 0.25, 0.5, 1.0]

N_solver = 20
n_grid   = 200

# ══════════════════════════════════════════════════════════════════════════════
# 1. Shared homotopy warm-start (α = 0.25, 0.5, 0.75) using default Q
# ══════════════════════════════════════════════════════════════════════════════

p_base = merge(default_params, (
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

function scale_seasonal_params(p, α)
    scale_coeffs(c) = (a0 = c.a0, a = α .* c.a, b = α .* c.b)
    return merge(p, (
        λ_coeffs = scale_coeffs(p.λ_coeffs),
        m_coeffs = scale_coeffs(p.m_coeffs),
        k_coeffs = scale_coeffs(p.k_coeffs),
    ))
end

# Homotopy up to α=0.75 (shared warm start for all Q)
V_coeffs_warmstart = initialize_V_constant(V_hom; N=N_solver)
warmstart_result = nothing

println("\nShared homotopy warm-start (α = 0.25 → 0.75)...")
for (i, α) in enumerate([0.25, 0.5, 0.75])
    p_scaled = scale_seasonal_params(p_base, α)
    println("  Homotopy $i/3: α = $α")

    global warmstart_result = solve_seasonal_model(p_scaled;
        N        = N_solver,
        V_init   = V_coeffs_warmstart,
        max_iter = 25,
        tol      = 1e-3,
        damping  = 0.5,
        verbose  = true,
    )
    global V_coeffs_warmstart = warmstart_result.V_coeffs
end

# ══════════════════════════════════════════════════════════════════════════════
# 2. Loop over Q values: Stage A (α=1.0) → Stage B → Stage D grid
# ══════════════════════════════════════════════════════════════════════════════

grid_results = Dict{Tuple{Float64,Float64}, Any}()

for Q_val in Q_values
    println("\n" * "="^70)
    println("  Q = $Q_val: Solving Stage A + B + D")
    println("="^70)

    # Parameters for this Q
    p_Q = merge(p_base, (Q = Q_val, ξ = 0.0))

    # Stage A: finish homotopy at α=1.0 with this Q
    println("\nStage A (Q=$Q_val): Final homotopy step α = 1.0...")
    model_result = solve_seasonal_model(p_Q;
        N        = N_solver,
        V_init   = V_coeffs_warmstart,
        max_iter = 50,
        tol      = 1e-3,
        damping  = 0.5,
        verbose  = true,
    )
    println("Stage A solved (converged = $(model_result.converged))")

    # Stage B: dollar continuation value W(t)
    println("\nStage B (Q=$Q_val): Solving W(t)...")
    W_result = solve_dollar_continuation_value(model_result, p_Q;
        N       = N_solver,
        max_iter = 50,
        tol     = 1e-3,
        damping = 0.5,
        verbose = true,
    )
    W_coeffs = W_result.W_coeffs
    println("Stage B solved (converged = $(W_result.converged))")

    # Evaluate baseline (ξ=0)
    eval_A = evaluate_solution(model_result, p_Q; n_grid=n_grid)
    W_grid = [spline_eval(t, W_coeffs) for t in eval_A.t_grid]
    grid_results[(Q_val, 0.0)] = (result=model_result, eval=eval_A, W_grid=W_grid)

    # Stage D: loop over ξ > 0
    for ξ_val in ξ_values
        ξ_val == 0.0 && continue

        println("\n──────────────────────────────────────────────────────────")
        println("  Q = $Q_val, ξ = $ξ_val")
        println("──────────────────────────────────────────────────────────")

        p_grid = merge(p_Q, (ξ = ξ_val,))

        if ξ_val <= 0.25
            result = solve_stage_D(model_result, W_coeffs, p_grid;
                N        = N_solver,
                max_iter = 50,
                tol      = 1e-3,
                damping  = 0.5,
                verbose  = true,
            )
        else
            cont = solve_stage_D_continuation(model_result, W_coeffs, p_grid;
                ξ_target  = ξ_val,
                n_steps   = 5,
                N         = N_solver,
                max_iter  = 25,
                tol       = 1e-3,
                damping   = 0.5,
                verbose   = true,
            )
            result = cont.result
        end

        eval_grid = evaluate_solution(result, p_grid; n_grid=n_grid)
        grid_results[(Q_val, ξ_val)] = (result=result, eval=eval_grid, W_grid=W_grid)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Export fine-grid CSV (long format for easy plotting)
# ══════════════════════════════════════════════════════════════════════════════

outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

rows = []
for Q_val in Q_values
    for ξ_val in ξ_values
        gr = grid_results[(Q_val, ξ_val)]
        ev = gr.eval
        res = gr.result
        p_Q = merge(p_base, (Q = Q_val, ξ = ξ_val))
        for j in 1:length(ev.t_grid)
            t_j = ev.t_grid[j]
            Vt_prime = spline_derivative(t_j, res.Vtilde_coeffs)
            Vt_val   = spline_eval(t_j, res.Vtilde_coeffs)
            fallow_foc = Vt_prime - p_Q.δ * Vt_val
            push!(rows, (
                t          = t_j,
                Q          = Q_val,
                xi         = ξ_val,
                V          = ev.V_grid[j],
                Vtilde     = ev.Vtilde_grid[j],
                tau_star   = ev.τ_star_grid[j],
                d_star     = ev.d_grid[j],
                W          = gr.W_grid[j],
                fallow_foc = fallow_foc,
            ))
        end
    end
end

df = DataFrame(rows)
CSV.write(joinpath(outdir, "Q_xi_grid.csv"), df)
println("\nWrote Q_xi_grid.csv ($(nrow(df)) rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Export nodal policies CSV
# ══════════════════════════════════════════════════════════════════════════════

policy_rows = []
for Q_val in Q_values
    for ξ_val in ξ_values
        res = grid_results[(Q_val, ξ_val)].result
        nodes = res.nodes
        for j in 1:length(nodes)
            push!(policy_rows, (
                node     = nodes[j],
                Q        = Q_val,
                xi       = ξ_val,
                V        = res.V_values[j],
                tau_star = res.τ_values[j],
                d_star   = res.d_values[j],
            ))
        end
    end
end

policy_df = DataFrame(policy_rows)
CSV.write(joinpath(outdir, "Q_xi_grid_policies.csv"), policy_df)
println("Wrote Q_xi_grid_policies.csv ($(nrow(policy_df)) rows)")

# ── Summary ───────────────────────────────────────────────────────────────────

println("\n── Grid summary (medium risk) ──────────────────────────────")
for Q_val in Q_values
    for ξ_val in ξ_values
        ev = grid_results[(Q_val, ξ_val)].eval
        println("  Q=$Q_val, ξ=$ξ_val: V̄=$(round(mean(ev.V_grid); digits=0)), " *
                "τ̄=$(round(mean(ev.τ_star_grid); digits=1)), " *
                "d̄=$(round(mean(ev.d_grid); digits=1))")
    end
end
