"""
    test_solver_convergence.jl

Validates the full iterative solver (`solve_seasonal_model`) against the
analytical homogeneous solution from 01_homogeneous_case.jl.

With constant hazard/growth/mortality rates (zero Fourier harmonics), the
seasonal solver should converge to:
- V(t) = V_analytical (constant across all t)
- τ*(t₀) = T*_analytical (constant across all t₀)
- d*(t) = 0 for all t (immediate restocking)
- All non-constant Fourier coefficients ≈ 0

This exercises the full numerical machinery: iterative convergence, Fourier
representation, survival/cost integrals, coupled FOCs, and value linkage.
See README § "Numerical Validation Procedure".
"""

using Test
using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Setup: homogeneous parameters for the seasonal solver
# ──────────────────────────────────────────────────────────────────────────────

# Build seasonal parameter set from the existing homogeneous constants.
# The seasonal solver needs (a0, a, b) Fourier coefficients; setting all
# harmonics to zero makes positive_periodic() return exp(a0) = the constant rate.
const seasonal_hom_params = merge(default_params, (
    λ_coeffs = (a0 = log(λ_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    m_coeffs = (a0 = log(m_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    k_coeffs = (a0 = log(k_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    γ     = 0.1,
    Y_MIN = 1000.0,
))

# Analytical benchmarks use the existing homogeneous_params from parameters.jl
const analytical_params = merge(homogeneous_params, (
    γ     = 0.1,
    Y_MIN = 1000.0,
))

# ── Analytical benchmarks ────────────────────────────────────────────────────
const T_star_analytical = solve_insurance(analytical_params)
const I_sol_analytical = solve_indemnity_homogeneous(T_star_analytical, analytical_params)
const V_analytical = insurance_value(T_star_analytical, I_sol_analytical, analytical_params)

println("Analytical benchmarks:")
println("  T* = $(round(T_star_analytical; digits=2)) days")
println("  V  = $(round(V_analytical; digits=2))")
println()


# ══════════════════════════════════════════════════════════════════════════════
@testset "Solver Convergence: Homogeneous Validation" begin
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
@testset "Full iterative solver converges" begin
    println("  Running seasonal solver with constant parameters...")
    result = solve_seasonal_model(seasonal_hom_params;
        N        = 10,
        max_iter = 200,
        tol      = 1e-4,
        damping  = 0.5,
        verbose  = false,
    )

    @test result.converged
    println("  Converged in $(result.iterations) iterations")

    # ── V(t) should be constant ──────────────────────────────────────────
    V_mean = result.V_coeffs.a0
    V_harmonics = vcat(result.V_coeffs.a, result.V_coeffs.b)
    max_harmonic_V = maximum(abs.(V_harmonics))

    println("  V Fourier: a0 = $(round(V_mean; digits=2)), " *
            "max |harmonic| = $(round(max_harmonic_V; sigdigits=3))")

    # Non-constant coefficients should be negligible relative to a0
    @test max_harmonic_V / abs(V_mean) < 0.01

    # ── V(t) ≈ V_analytical ──────────────────────────────────────────────
    V_rel_error = abs(V_mean - V_analytical) / abs(V_analytical)
    println("  V_mean = $(round(V_mean; digits=2)), " *
            "V_analytical = $(round(V_analytical; digits=2)), " *
            "relative error = $(round(V_rel_error * 100; digits=4))%")
    @test V_rel_error < 0.05  # within 5%

    # ── τ*(t₀) should be constant ────────────────────────────────────────
    τ_mean = result.τ_star_coeffs.a0
    τ_harmonics = vcat(result.τ_star_coeffs.a, result.τ_star_coeffs.b)
    max_harmonic_τ = maximum(abs.(τ_harmonics))

    println("  τ* Fourier: a0 = $(round(τ_mean; digits=2)), " *
            "max |harmonic| = $(round(max_harmonic_τ; sigdigits=3))")

    @test max_harmonic_τ / abs(τ_mean) < 0.01

    # ── τ* ≈ T*_analytical ───────────────────────────────────────────────
    τ_rel_error = abs(τ_mean - T_star_analytical) / T_star_analytical
    println("  τ*_mean = $(round(τ_mean; digits=2)), " *
            "T*_analytical = $(round(T_star_analytical; digits=2)), " *
            "relative error = $(round(τ_rel_error * 100; digits=4))%")
    @test τ_rel_error < 0.05

    # ── d*(t) should be zero everywhere (immediate restocking) ───────────
    max_d = maximum(result.d_values)
    println("  max d*(t) = $(round(max_d; digits=2)) days")
    @test max_d < 1.0  # effectively zero fallow

    # ── Ṽ(t₀) should be constant and ≈ V ─────────────────────────────────
    Vtilde_mean = result.Vtilde_coeffs.a0
    Vtilde_harmonics = vcat(result.Vtilde_coeffs.a, result.Vtilde_coeffs.b)
    max_harmonic_Vt = maximum(abs.(Vtilde_harmonics))

    println("  Ṽ Fourier: a0 = $(round(Vtilde_mean; digits=2)), " *
            "max |harmonic| = $(round(max_harmonic_Vt; sigdigits=3))")

    @test max_harmonic_Vt / abs(Vtilde_mean) < 0.01

    # With d*=0, V(t) = Ṽ(t₀), so Ṽ_mean ≈ V_mean
    Vtilde_V_gap = abs(Vtilde_mean - V_mean) / abs(V_mean)
    println("  |Ṽ_mean - V_mean| / V_mean = $(round(Vtilde_V_gap * 100; digits=4))%")
    @test Vtilde_V_gap < 0.05

    # ── V should be constant at all nodes ────────────────────────────────
    V_range = maximum(result.V_values) - minimum(result.V_values)
    V_cv = V_range / abs(V_mean)
    println("  V range at nodes: $(round(V_range; sigdigits=3)), " *
            "CV = $(round(V_cv * 100; digits=4))%")
    @test V_cv < 0.01

    # ── τ* should be constant at all nodes ───────────────────────────────
    τ_range = maximum(result.τ_values) - minimum(result.τ_values)
    τ_cv = τ_range / abs(τ_mean)
    println("  τ* range at nodes: $(round(τ_range; sigdigits=3)), " *
            "CV = $(round(τ_cv * 100; digits=4))%")
    @test τ_cv < 0.01
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Convergence history" begin
    println("  Running solver to check convergence rate...")
    result = solve_seasonal_model(seasonal_hom_params;
        N        = 10,
        max_iter = 200,
        tol      = 1e-6,
        damping  = 0.5,
        verbose  = false,
    )

    # Convergence should be monotonic (or nearly so) after initial iterations
    println("  Convergence history:")
    for (k, Δ) in result.history
        println("    iter $k: ΔV = $(round(Δ; sigdigits=4))")
    end

    @test result.converged
    @test result.iterations < 50

    # ΔV should decrease overall
    first_Δ = result.history[1][2]
    last_Δ = result.history[end][2]
    @test last_Δ < first_Δ
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Export convergence data" begin
    outdir = joinpath(@__DIR__, "..", "results", "simulations")
    mkpath(outdir)

    result = solve_seasonal_model(seasonal_hom_params;
        N        = 10,
        max_iter = 200,
        tol      = 1e-6,
        damping  = 0.5,
        verbose  = false,
    )

    # ── Convergence history ──────────────────────────────────────────────
    hist_df = DataFrame(
        iteration = [h[1] for h in result.history],
        delta_V   = [h[2] for h in result.history],
    )
    CSV.write(joinpath(outdir, "convergence_history.csv"), hist_df)
    println("  Wrote convergence_history.csv ($(nrow(hist_df)) rows)")

    # ── Nodal values vs analytical ───────────────────────────────────────
    node_df = DataFrame(
        node           = result.nodes,
        V              = result.V_values,
        V_analytical   = fill(V_analytical, length(result.nodes)),
        tau_star        = result.τ_values,
        T_analytical   = fill(T_star_analytical, length(result.nodes)),
        d_star         = result.d_values,
        Vtilde         = result.Vtilde_at_V_nodes,
    )
    CSV.write(joinpath(outdir, "homogeneous_validation.csv"), node_df)
    println("  Wrote homogeneous_validation.csv ($(nrow(node_df)) rows)")

    @test true
end

# ══════════════════════════════════════════════════════════════════════════════
end # top-level testset
