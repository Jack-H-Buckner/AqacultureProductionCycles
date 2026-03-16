"""
    test_continuation_value_solver.jl

Tests for the continuation value solver in src/03_continuation_value_solver.jl.

Takes a hypothetical continuation value V(t) = V_hom + A·sin(2πt/365)
(homogeneous value plus a periodic perturbation), performs one iteration of
the V(t) and Ṽ(t₀) update, fits Fourier series to the nodal values, and
compares the Fourier approximations against values computed on a fine grid
of n=100 points.

The V(t) fine grid uses linear interpolation of d*(t) from pre-computed nodal
values, making it cheap (one compute_Vtilde per grid point instead of ~1500).
"""

using Test
using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Setup: hypothetical V(t) = V_hom + A·sin(2πt/365)
# ──────────────────────────────────────────────────────────────────────────────

const test_p = merge(default_params, (
    γ     = 0.1,
    Y_MIN = 1000.0,
))

const hom_p = merge(homogeneous_params, (
    γ     = 0.1,
    Y_MIN = 1000.0,
))

const T_hom = solve_insurance(hom_p)
const I_sol_hom = solve_indemnity_homogeneous(T_hom, hom_p)
const V_hom = insurance_value(T_hom, I_sol_hom, hom_p)

# Perturbation: 1% of V_hom
const A_perturb = 0.01 * V_hom

# Number of harmonics for the test
const N_TEST = 40

# Hypothetical V(t) Fourier coefficients
const V_coeffs_hyp = (
    a0 = V_hom,
    a  = vcat([A_perturb], zeros(N_TEST - 1)),
    b  = zeros(N_TEST),
)

# ── Shared setup: solve harvest FOC and stocking FOC once ────────────────────
println("Setup: solving harvest FOC at $(2N_TEST+1) Fourier nodes...")
const harvest_result_shared = solve_harvest_at_nodes(V_coeffs_hyp, test_p; N=N_TEST)
const τ_star_coeffs_shared = harvest_result_shared.τ_star_coeffs
println("Setup: τ̄ = $(round(τ_star_coeffs_shared.a0; digits=1)) days")

println("Setup: computing Ṽ(t₀) Fourier series at $(2N_TEST+1) nodes...")
const Vtilde_shared = compute_Vtilde_at_nodes(τ_star_coeffs_shared, V_coeffs_hyp, test_p; N=N_TEST)
const Vtilde_coeffs_shared = Vtilde_shared.Vtilde_coeffs
println("Setup: Ṽ̄ = $(round(Vtilde_coeffs_shared.a0; digits=2))")

println("Setup: solving stocking FOC at $(2N_TEST+1) Fourier nodes...")
const stocking_shared = solve_stocking_at_V_nodes(Vtilde_coeffs_shared, test_p; N=N_TEST)
const d_nodes_shared = stocking_shared.d_values
const nodes_shared = stocking_shared.nodes
n_corner = count(d -> d == 0.0, d_nodes_shared)
println("Setup: d* computed at nodes ($n_corner corner solutions)")


# ══════════════════════════════════════════════════════════════════════════════
@testset "Continuation Value Solver Tests" begin
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
@testset "Ṽ(t₀) update: Fourier vs fine grid (n=100)" begin
    # Step 1: Compute Ṽ(t₀) at Fourier nodes and fit Fourier series
    println("  Computing Ṽ(t₀) at $(2N_TEST+1) Fourier nodes...")
    Vtilde_result = compute_Vtilde_at_nodes(τ_star_coeffs_shared, V_coeffs_hyp, test_p; N=N_TEST)
    Vtilde_coeffs = Vtilde_result.Vtilde_coeffs
    println("  Ṽ̄(Fourier) = $(round(Vtilde_coeffs.a0; digits=2))")

    # Step 2: Compute Ṽ(t₀) on a fine grid of 100 points
    n_fine = 100
    println("  Computing Ṽ(t₀) on $(n_fine)-point fine grid...")
    t0_fine = collect(range(0.0, PERIOD * (1 - 1/n_fine), length=n_fine))

    Vtilde_fine = Float64[]
    for t₀ in t0_fine
        τ_star = fourier_eval(t₀, τ_star_coeffs_shared)
        T_star = t₀ + τ_star
        Vtilde = compute_Vtilde(t₀, T_star, V_coeffs_hyp, test_p)
        push!(Vtilde_fine, Vtilde)
    end

    # Step 3: Evaluate the Fourier series at the fine grid points
    Vtilde_fourier_at_grid = [fourier_eval(t, Vtilde_coeffs) for t in t0_fine]

    # Step 4: Compare
    abs_errors = abs.(Vtilde_fourier_at_grid .- Vtilde_fine)
    rel_errors = abs_errors ./ abs.(Vtilde_fine)

    max_rel = maximum(rel_errors)
    mean_rel = sum(rel_errors) / length(rel_errors)

    println("  Ṽ(t₀) absolute error: max = $(round(maximum(abs_errors); sigdigits=4)), " *
            "mean = $(round(sum(abs_errors)/length(abs_errors); sigdigits=4))")
    println("  Ṽ(t₀) relative error: max = $(round(max_rel * 100; digits=3))%, " *
            "mean = $(round(mean_rel * 100; digits=3))%")

    @test all(isfinite, Vtilde_fine)
    @test all(v -> v > 0, Vtilde_fine)
    @test all(isfinite, Vtilde_fourier_at_grid)
    @test mean_rel < 0.05
end

# ──────────────────────────────────────────────────────────────────────────
@testset "V(t) update: Fourier vs fine grid (n=100)" begin
    # Step 1: Update V(t) at Fourier nodes and fit Fourier series
    println("  Updating V(t) at $(2N_TEST+1) Fourier nodes...")
    V_update = update_V_all_nodes(τ_star_coeffs_shared, Vtilde_coeffs_shared, Vtilde_shared, test_p; N=N_TEST)
    V_new_coeffs = V_update.V_new_coeffs
    println("  V̄(Fourier) = $(round(V_new_coeffs.a0; digits=2))")

    # Step 2: Compute V(t) on a 100-point fine grid using interpolated d*
    # Uses value linkage V(t) = exp(-δ·d*) · Ṽ(t₀*) matching update_V_all_nodes
    n_fine = 100
    println("  Computing V(t) on $(n_fine)-point fine grid (interpolated d*)...")
    t_fine = collect(range(0.0, PERIOD * (1 - 1/n_fine), length=n_fine))

    V_fine = Float64[]
    for t in t_fine
        d = interpolate_d_star(t, nodes_shared, d_nodes_shared)
        t0_star = t + d
        Vtilde_at_t0 = fourier_eval(t0_star, Vtilde_coeffs_shared)
        V_t = exp(-test_p.δ * d) * Vtilde_at_t0
        push!(V_fine, V_t)
    end

    # Step 3: Evaluate the Fourier series at the fine grid points
    V_fourier_at_grid = [fourier_eval(t, V_new_coeffs) for t in t_fine]

    # Step 4: Compare
    abs_errors = abs.(V_fourier_at_grid .- V_fine)
    rel_errors = abs_errors ./ abs.(V_fine)

    max_rel = maximum(rel_errors)
    mean_rel = sum(rel_errors) / length(rel_errors)

    println("  V(t) absolute error: max = $(round(maximum(abs_errors); sigdigits=4)), " *
            "mean = $(round(sum(abs_errors)/length(abs_errors); sigdigits=4))")
    println("  V(t) relative error: max = $(round(max_rel * 100; digits=3))%, " *
            "mean = $(round(mean_rel * 100; digits=3))%")

    @test all(isfinite, V_fine)
    @test all(v -> v > 0, V_fine)
    @test all(isfinite, V_fourier_at_grid)
    @test mean_rel < 0.15  # Fourier interpolation + linear system vs value linkage
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Value linkage consistency" begin
    # Verify V(t) = exp(-δ·d*) · Ṽ(t₀*) holds exactly
    println("  Checking value linkage consistency...")

    test_times = [0.0, 91.25, 182.5, 273.75]
    for t in test_times
        d = interpolate_d_star(t, nodes_shared, d_nodes_shared)
        res = compute_V_from_d(t, d, τ_star_coeffs_shared, V_coeffs_hyp, test_p)
        V_linkage = exp(-test_p.δ * d) * res.Vtilde
        @test res.V_t ≈ V_linkage rtol=1e-12

        println("  t=$(round(t; digits=1)): d*=$(round(d; digits=1)), " *
                "Ṽ=$(round(res.Vtilde; digits=2)), V=$(round(res.V_t; digits=2))")
    end
end

# ──────────────────────────────────────────────────────────────────────────
@testset "d* interpolation vs nodal values" begin
    # At each node, interpolation should recover the exact value
    println("  Checking d* interpolation at nodes...")
    for (i, t) in enumerate(nodes_shared)
        d_interp = interpolate_d_star(t, nodes_shared, d_nodes_shared)
        @test d_interp ≈ d_nodes_shared[i] atol=1e-10
    end
    println("  d* interpolation exact at all $(length(nodes_shared)) nodes")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Export comparison data" begin
    outdir = joinpath(@__DIR__, "..", "results", "simulations")
    mkpath(outdir)

    n_fine = 100

    # ── Ṽ(t₀) comparison ────────────────────────────────────────────────
    println("  Computing Ṽ(t₀) for export...")
    Vtilde_result = compute_Vtilde_at_nodes(τ_star_coeffs_shared, V_coeffs_hyp, test_p; N=N_TEST)
    Vtilde_coeffs = Vtilde_result.Vtilde_coeffs

    t0_fine = collect(range(0.0, PERIOD * (1 - 1/n_fine), length=n_fine))
    Vtilde_fine = Float64[]
    for t₀ in t0_fine
        τ_star = fourier_eval(t₀, τ_star_coeffs_shared)
        T_star = t₀ + τ_star
        push!(Vtilde_fine, compute_Vtilde(t₀, T_star, V_coeffs_hyp, test_p))
    end
    Vtilde_fourier = [fourier_eval(t, Vtilde_coeffs) for t in t0_fine]

    df_Vt = DataFrame(
        t0             = t0_fine,
        Vtilde_exact   = Vtilde_fine,
        Vtilde_fourier = Vtilde_fourier,
    )
    CSV.write(joinpath(outdir, "Vtilde_update_comparison.csv"), df_Vt)
    println("  Wrote Vtilde_update_comparison.csv ($(nrow(df_Vt)) rows)")

    # ── V(t) comparison ──────────────────────────────────────────────────
    println("  Computing V(t) for export...")
    V_update = update_V_all_nodes(τ_star_coeffs_shared, Vtilde_coeffs_shared, Vtilde_shared, test_p; N=N_TEST)
    V_new_coeffs = V_update.V_new_coeffs

    t_fine = collect(range(0.0, PERIOD * (1 - 1/n_fine), length=n_fine))
    V_fine = Float64[]
    d_fine = Float64[]
    Vtilde_linkage = Float64[]
    t0_star_fine = Float64[]
    for t in t_fine
        d = interpolate_d_star(t, nodes_shared, d_nodes_shared)
        res = compute_V_from_d(t, d, τ_star_coeffs_shared, V_coeffs_hyp, test_p)
        push!(V_fine, res.V_t)
        push!(d_fine, d)
        push!(Vtilde_linkage, res.Vtilde)
        push!(t0_star_fine, res.t0_star)
    end
    V_fourier = [fourier_eval(t, V_new_coeffs) for t in t_fine]

    df_V = DataFrame(
        t              = t_fine,
        V_exact        = V_fine,
        V_fourier      = V_fourier,
        d_star         = d_fine,
        t0_star        = t0_star_fine,
        Vtilde_linkage = Vtilde_linkage,
    )
    CSV.write(joinpath(outdir, "V_update_comparison.csv"), df_V)
    println("  Wrote V_update_comparison.csv ($(nrow(df_V)) rows)")

    # ── Node values ──────────────────────────────────────────────────────
    node_df = DataFrame(
        node           = V_update.nodes,
        V_at_node      = V_update.V_values,
        Vtilde_at_node = V_update.Vtilde_values,
        d_at_node      = V_update.d_values,
        t0_at_node     = V_update.t0_values,
    )
    CSV.write(joinpath(outdir, "V_update_nodes.csv"), node_df)
    println("  Wrote V_update_nodes.csv ($(nrow(node_df)) rows)")

    Vtilde_node_df = DataFrame(
        t0_node        = Vtilde_result.nodes,
        Vtilde_at_node = Vtilde_result.Vtilde_values,
    )
    CSV.write(joinpath(outdir, "Vtilde_update_nodes.csv"), Vtilde_node_df)
    println("  Wrote Vtilde_update_nodes.csv ($(nrow(Vtilde_node_df)) rows)")

    @test true
end

# ══════════════════════════════════════════════════════════════════════════════
end # top-level testset
