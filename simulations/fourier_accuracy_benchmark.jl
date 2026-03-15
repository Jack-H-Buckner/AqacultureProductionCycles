"""
    fourier_accuracy_benchmark.jl

Benchmark the accuracy vs performance trade-off for Fourier approximation
of the harvest FOC solution τ*(t₀). Sweeps N from 5 to 30 in steps of 5,
measuring wall-clock time and MSE against a 100-point "exact" grid solution.

Outputs `fourier_benchmark.csv` with columns: N, n_nodes, time_s, mse, max_error.
"""

using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "02_first_order_conditions.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ── Setup: approximate V(t) ─────────────────────────────────────────────────
hom_p = merge(homogeneous_params, (γ = 0.1, Y_MIN = 1000.0))
T_hom = solve_insurance(hom_p)
I_sol_hom = solve_indemnity_homogeneous(T_hom, hom_p)
V_hom = insurance_value(T_hom, I_sol_hom, hom_p)

test_p = merge(default_params, (γ = 0.1, Y_MIN = 1000.0))
A_perturb = 0.01 * V_hom

# V_coeffs needs enough harmonics for the largest N we'll test
N_max = 30
V_coeffs = (
    a0 = V_hom,
    a  = vcat([A_perturb], zeros(N_max - 1)),
    b  = zeros(N_max),
)

# ── Compute "exact" solution on 100-point grid ──────────────────────────────
println("Computing exact solution on 100-point grid...")
exact = solve_harvest_on_grid(V_coeffs, test_p; n_grid=100)
println("Done. τ* range: $(round(minimum(exact.τ_grid), digits=1)) — $(round(maximum(exact.τ_grid), digits=1)) days")

# ── Sweep over N ─────────────────────────────────────────────────────────────
N_values = 5:5:30
results = DataFrame(N=Int[], n_nodes=Int[], time_s=Float64[], mse=Float64[], max_error=Float64[])

for N in N_values
    println("\nN=$N ($(2N+1) nodes)...")

    # Time the node solve + Fourier fit
    t_start = time()
    res = solve_harvest_at_nodes(V_coeffs, test_p; N=N)
    t_elapsed = time() - t_start

    τ_star_coeffs = res.τ_star_coeffs

    # Evaluate Fourier approximation at exact grid points
    τ_fourier = [fourier_eval(t, τ_star_coeffs) for t in exact.t₀_grid]
    errors = τ_fourier .- exact.τ_grid
    mse = sum(errors .^ 2) / length(errors)
    max_err = maximum(abs.(errors))

    println("  Time: $(round(t_elapsed, digits=2))s, MSE: $(round(mse, digits=2)), Max error: $(round(max_err, digits=2)) days")
    push!(results, (N=N, n_nodes=2N+1, time_s=t_elapsed, mse=mse, max_error=max_err))
end

# ── Write output ─────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)
CSV.write(joinpath(outdir, "fourier_benchmark.csv"), results)
println("\nWrote fourier_benchmark.csv ($(nrow(results)) rows)")
println(results)
