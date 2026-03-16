"""
    test_solver_convergence.jl

Validates the full iterative solver (`solve_seasonal_model`) against the
analytical homogeneous solution from 01_homogeneous_case.jl.

With constant hazard/growth/mortality rates (zero Fourier harmonics), the
seasonal solver should converge to:
- V(t) = V_analytical (constant across all t)
- τ*(t₀) = T*_analytical (constant across all t₀)
- d*(t) = 0 for all t (immediate restocking)
- All nodal values approximately equal (no seasonal variation)

This exercises the full numerical machinery: iterative convergence, spline
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
        N        = 20,
        max_iter = 200,
        tol      = 1e-4,
        damping  = 0.5,
        verbose  = false,
    )

    @test result.converged
    println("  Converged in $(result.iterations) iterations")

    # ── V(t) should be constant ──────────────────────────────────────────
    V_mean = sum(result.V_coeffs.values) / length(result.V_coeffs.values)
    V_range = maximum(result.V_coeffs.values) - minimum(result.V_coeffs.values)

    println("  V spline: mean = $(round(V_mean; digits=2)), " *
            "range = $(round(V_range; sigdigits=3))")

    # Nodal values should be nearly constant (range negligible relative to mean)
    @test V_range / abs(V_mean) < 0.01

    # ── V(t) ≈ V_analytical ──────────────────────────────────────────────
    V_rel_error = abs(V_mean - V_analytical) / abs(V_analytical)
    println("  V_mean = $(round(V_mean; digits=2)), " *
            "V_analytical = $(round(V_analytical; digits=2)), " *
            "relative error = $(round(V_rel_error * 100; digits=4))%")
    @test V_rel_error < 0.05  # within 5%

    # ── τ*(t₀) should be constant ────────────────────────────────────────
    τ_mean = sum(result.τ_star_coeffs.values) / length(result.τ_star_coeffs.values)
    τ_range = maximum(result.τ_star_coeffs.values) - minimum(result.τ_star_coeffs.values)

    println("  τ* spline: mean = $(round(τ_mean; digits=2)), " *
            "range = $(round(τ_range; sigdigits=3))")

    @test τ_range / abs(τ_mean) < 0.01

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
    Vtilde_mean = sum(result.Vtilde_coeffs.values) / length(result.Vtilde_coeffs.values)
    Vtilde_range = maximum(result.Vtilde_coeffs.values) - minimum(result.Vtilde_coeffs.values)

    println("  Ṽ spline: mean = $(round(Vtilde_mean; digits=2)), " *
            "range = $(round(Vtilde_range; sigdigits=3))")

    @test Vtilde_range / abs(Vtilde_mean) < 0.01

    # With d*=0, V(t) = Ṽ(t₀), so Ṽ_mean ≈ V_mean
    Vtilde_V_gap = abs(Vtilde_mean - V_mean) / abs(V_mean)
    println("  |Ṽ_mean - V_mean| / V_mean = $(round(Vtilde_V_gap * 100; digits=4))%")
    @test Vtilde_V_gap < 0.05

    # ── V should be constant at all nodes ────────────────────────────────
    V_node_range = maximum(result.V_values) - minimum(result.V_values)
    V_cv = V_node_range / abs(V_mean)
    println("  V range at nodes: $(round(V_node_range; sigdigits=3)), " *
            "CV = $(round(V_cv * 100; digits=4))%")
    @test V_cv < 0.01

    # ── τ* should be constant at all nodes ───────────────────────────────
    τ_node_range = maximum(result.τ_values) - minimum(result.τ_values)
    τ_cv = τ_node_range / abs(τ_mean)
    println("  τ* range at nodes: $(round(τ_node_range; sigdigits=3)), " *
            "CV = $(round(τ_cv * 100; digits=4))%")
    @test τ_cv < 0.01
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Convergence history" begin
    println("  Running solver to check convergence rate...")
    result = solve_seasonal_model(seasonal_hom_params;
        N        = 20,
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
        N        = 20,
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

# ──────────────────────────────────────────────────────────────────────────
@testset "Fixed-point consistency: V(t) = e^{-δd}·Ṽ(t₀*) via full Bellman" begin
    # After the direct linear solver converges, verify that the solution is
    # also a fixed point of the exact Bellman equation:
    #   Ṽ(t₀) = S·e^{-δτ}·[u(Y_H) + V(T*)] + ∫ S·λ·e^{-δs}·[u(Y_L) + V(s)] ds
    # The direct solver uses an f/g decomposition (Ṽ ≈ f + g·V(T*)) that
    # collapses the integral over V(s) to a single evaluation at T*. This test
    # checks that the full integral (which evaluates V(s) at every loss time)
    # produces the same Ṽ, confirming the approximation is valid at convergence.

    println("  Running seasonal solver for fixed-point check...")
    result = solve_seasonal_model(seasonal_hom_params;
        N        = 20,
        max_iter = 200,
        tol      = 1e-6,
        damping  = 0.5,
        verbose  = false,
    )
    @test result.converged

    # Recompute Ṽ at each node using the full Bellman equation
    # (compute_Vtilde integrates V(s) at every quadrature point in the loss
    # integral, unlike compute_Vtilde_decomposed which factors out V(T*))
    nodes = result.nodes
    V_coeffs = result.V_coeffs
    τ_star_coeffs = result.τ_star_coeffs

    println("  Recomputing Ṽ(t₀) via full Bellman at $(length(nodes)) nodes...")
    max_V_error = 0.0
    max_Vtilde_error = 0.0

    for (i, t) in enumerate(nodes)
        d_star = result.d_values[i]
        t0_star = t + d_star
        τ_star = spline_eval(t0_star, τ_star_coeffs)
        T_star = t0_star + τ_star

        # Full Bellman Ṽ: evaluates V(s) inside the loss integral at every s
        Vtilde_full = compute_Vtilde(t0_star, T_star, V_coeffs, seasonal_hom_params)

        # V from full Bellman via value linkage
        V_full = exp(-seasonal_hom_params.δ * d_star) * Vtilde_full

        # V from the converged spline
        V_spline = spline_eval(t, V_coeffs)

        # Ṽ from the converged spline
        Vtilde_spline = spline_eval(t0_star, result.Vtilde_coeffs)

        V_err = abs(V_full - V_spline) / abs(V_spline)
        Vtilde_err = abs(Vtilde_full - Vtilde_spline) / abs(Vtilde_spline)

        max_V_error = max(max_V_error, V_err)
        max_Vtilde_error = max(max_Vtilde_error, Vtilde_err)
    end

    println("  Max relative error in Ṽ (full Bellman vs spline): $(round(max_Vtilde_error * 100; digits=6))%")
    println("  Max relative error in V (full Bellman vs spline):  $(round(max_V_error * 100; digits=6))%")

    @test max_Vtilde_error < 1e-4
    @test max_V_error < 1e-4
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Fixed-point consistency: seasonal parameters" begin
    # Same check as above but with the full seasonal parameters (non-constant
    # λ, m, k). This is the case where V(s) genuinely varies over the cycle,
    # so the f/g decomposition's approximation is non-trivial.

    seasonal_p = merge(default_params, (
        γ     = 0.1,
        Y_MIN = 1000.0,
    ))

    println("  Running seasonal solver (seasonal params) for fixed-point check...")
    result = solve_seasonal_model(seasonal_p;
        N        = 10,
        max_iter = 200,
        tol      = 1e-4,
        damping  = 0.5,
        verbose  = false,
    )
    @test result.converged
    println("  Converged in $(result.iterations) iterations")

    nodes = result.nodes
    V_coeffs = result.V_coeffs
    τ_star_coeffs = result.τ_star_coeffs

    println("  Recomputing Ṽ(t₀) via full Bellman at $(length(nodes)) nodes...")
    max_V_error = 0.0
    max_Vtilde_error = 0.0

    for (i, t) in enumerate(nodes)
        d_star = result.d_values[i]
        t0_star = t + d_star
        τ_star = spline_eval(t0_star, τ_star_coeffs)
        T_star = t0_star + τ_star

        # Full Bellman Ṽ: evaluates V(s) inside the loss integral at every s
        Vtilde_full = compute_Vtilde(t0_star, T_star, V_coeffs, seasonal_p)

        # V from full Bellman via value linkage
        V_full = exp(-seasonal_p.δ * d_star) * Vtilde_full

        # V from the converged spline
        V_spline = spline_eval(t, V_coeffs)

        # Ṽ from the converged spline
        Vtilde_spline = spline_eval(t0_star, result.Vtilde_coeffs)

        V_err = abs(V_full - V_spline) / abs(V_spline)
        Vtilde_err = abs(Vtilde_full - Vtilde_spline) / abs(Vtilde_spline)

        max_V_error = max(max_V_error, V_err)
        max_Vtilde_error = max(max_Vtilde_error, Vtilde_err)
    end

    println("  Max relative error in Ṽ (full Bellman vs spline): $(round(max_Vtilde_error * 100; digits=6))%")
    println("  Max relative error in V (full Bellman vs spline):  $(round(max_V_error * 100; digits=6))%")

    # The f/g decomposition approximates ∫ S·λ·e^{-δs}·V(s) ds ≈ g·V(T*),
    # so with seasonal V(s) there is a non-zero discrepancy. These tolerances
    # confirm the approximation error is small (< 1% for both Ṽ and V).
    @test max_Vtilde_error < 0.01
    @test max_V_error < 0.01
end

# ══════════════════════════════════════════════════════════════════════════════
end # top-level testset
