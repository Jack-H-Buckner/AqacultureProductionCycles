"""
    benchmark_performance.jl

Benchmark key solver functions to measure performance of the precomputed
cumulative ODE optimization. Times:
1. A single compute_Vtilde_decomposed call (the inner loop bottleneck)
2. A single solve_harvest_foc call (harvest bracket search)
3. One full iteration of the seasonal solver (harvest FOC + Ṽ + V update)
"""

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# Use seasonal parameters (the case that was slow)
p = default_params

# Warm start from homogeneous solution
T_hom = solve_insurance(merge(homogeneous_params, (γ = p.γ, Y_MIN = p.Y_MIN)))
I_hom = solve_indemnity_homogeneous(T_hom, merge(homogeneous_params, (γ = p.γ, Y_MIN = p.Y_MIN)))
V_hom = insurance_value(T_hom, I_hom, merge(homogeneous_params, (γ = p.γ, Y_MIN = p.Y_MIN)))

N_solver = 10
V_coeffs = initialize_V_constant(V_hom; N=N_solver)

# ── Warmup ────────────────────────────────────────────────────────────────────
println("Warming up...")
_ = compute_Vtilde_decomposed(0.0, 300.0, p)
_ = solve_harvest_foc(0.0, V_coeffs, p; τ_max=500.0)

# ── Benchmark 1: compute_Vtilde_decomposed ─────────────────────────────────
println("\n── Benchmark 1: compute_Vtilde_decomposed (21 calls) ──")
t₀_nodes = fourier_nodes(N_solver)
times_vtilde = Float64[]
for rep in 1:3
    t_start = time()
    for t₀ in t₀_nodes
        compute_Vtilde_decomposed(t₀, t₀ + 300.0, p)
    end
    elapsed = time() - t_start
    push!(times_vtilde, elapsed)
    println("  Rep $rep: $(round(elapsed; digits=3))s ($(round(elapsed/length(t₀_nodes)*1000; digits=1))ms/call)")
end
println("  Median: $(round(sort(times_vtilde)[2]; digits=3))s")

# ── Benchmark 2: solve_harvest_foc ──────────────────────────────────────────
println("\n── Benchmark 2: solve_harvest_foc (21 calls) ──")
times_harvest = Float64[]
for rep in 1:3
    t_start = time()
    for t₀ in t₀_nodes
        solve_harvest_foc(t₀, V_coeffs, p; τ_max=500.0)
    end
    elapsed = time() - t_start
    push!(times_harvest, elapsed)
    println("  Rep $rep: $(round(elapsed; digits=3))s ($(round(elapsed/length(t₀_nodes)*1000; digits=1))ms/call)")
end
println("  Median: $(round(sort(times_harvest)[2]; digits=3))s")

# ── Benchmark 3: Full solver iteration (3 iterations) ───────────────────────
println("\n── Benchmark 3: solve_seasonal_model (3 iterations) ──")
times_solver = Float64[]
for rep in 1:3
    t_start = time()
    result = solve_seasonal_model(p;
        N = N_solver,
        V_init = V_coeffs,
        max_iter = 3,
        tol = 1e-20,  # don't stop early
        damping = 0.5,
        τ_max = 500.0,
        verbose = false,
    )
    elapsed = time() - t_start
    push!(times_solver, elapsed)
    println("  Rep $rep: $(round(elapsed; digits=3))s ($(round(elapsed/3; digits=3))s/iter)")
end
println("  Median: $(round(sort(times_solver)[2]; digits=3))s")

println("\n── Summary ──")
println("  Vtilde_decomposed (21 calls): $(round(sort(times_vtilde)[2]; digits=3))s")
println("  harvest_foc (21 calls):        $(round(sort(times_harvest)[2]; digits=3))s")
println("  3 solver iterations:           $(round(sort(times_solver)[2]; digits=3))s")
