"""
    test_homogeneous_validation.jl

Validates the full seasonal solver (03_continuation_value_solver.jl) against
the homogeneous analytical solution (01_homogeneous_case.jl).

When all seasonal Fourier coefficients have zero higher harmonics (only a0),
the seasonal model should reduce exactly to the homogeneous (constant-rate)
case. This test:

1. Constructs a seasonal parameter set with constant rates (zero harmonics).
2. Solves the homogeneous case analytically → T*, V*, d* = 0.
3. Runs the full seasonal solver with the same constant rates.
4. Compares: τ*(t₀) ≈ T* (constant), V(t) ≈ V* (constant), d*(t) = 0 (all corner).
"""

using Test

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "03_continuation_value_solver.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Setup: homogeneous parameter sets
# ──────────────────────────────────────────────────────────────────────────────

# Seasonal parameter set with ZERO higher harmonics (constant rates)
# This makes the seasonal functions reduce to exp(a0) = constant
const hom_seasonal_params = merge(default_params, (
    λ_coeffs = (a0 = log(λ_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    m_coeffs = (a0 = log(m_const), a = [0.0, 0.0], b = [0.0, 0.0]),
    k_coeffs = (a0 = log(k_const), a = [0.0, 0.0], b = [0.0, 0.0]),
))

# ──────────────────────────────────────────────────────────────────────────────
# Homogeneous analytical solution (benchmark)
# ──────────────────────────────────────────────────────────────────────────────

println("Solving homogeneous case analytically...")
const T_star_hom = solve_insurance(homogeneous_params)
const I_sol_hom = solve_indemnity_homogeneous(T_star_hom, homogeneous_params)
const V_hom = insurance_value(T_star_hom, I_sol_hom, homogeneous_params)
println("  T* = $(round(T_star_hom; digits=2)) days")
println("  V* = $(round(V_hom; digits=2))")

# ──────────────────────────────────────────────────────────────────────────────
# Run full seasonal solver with constant rates
# ──────────────────────────────────────────────────────────────────────────────

const N_TEST = 10

# Warm-start with the known homogeneous V (constant Fourier series)
const V_init_hom = initialize_V_constant(V_hom; N=N_TEST)

println("\nRunning seasonal solver with constant rates (N=$N_TEST)...")
const result = solve_seasonal_model(hom_seasonal_params;
    N = N_TEST,
    V_init = V_init_hom,
    max_iter = 200,
    tol = 1e-4,
    damping = 1.0,
    verbose = true,
)

# ══════════════════════════════════════════════════════════════════════════════
@testset "Homogeneous Validation Tests" begin
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
@testset "Solver convergence" begin
    @test result.converged
    println("  Converged in $(result.iterations) iterations")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Harvest time τ*(t₀) ≈ T* (constant)" begin
    # The spline for τ* should be approximately constant = T*_hom
    τ_values = result.τ_values
    τ_mean = sum(result.τ_star_coeffs.values) / length(result.τ_star_coeffs.values)

    # Mean should match homogeneous T*
    rel_err_mean = abs(τ_mean - T_star_hom) / T_star_hom
    println("  τ̄ (spline mean) = $(round(τ_mean; digits=2)) days")
    println("  T* (homogeneous) = $(round(T_star_hom; digits=2)) days")
    println("  Relative error = $(round(rel_err_mean * 100; digits=4))%")
    @test rel_err_mean < 0.01  # < 1% error

    # All nodal values should be close to T*
    max_node_err = maximum(abs.(τ_values .- T_star_hom)) / T_star_hom
    println("  Max nodal relative error = $(round(max_node_err * 100; digits=4))%")
    @test max_node_err < 0.01

    # Variation should be near zero (constant in homogeneous case)
    τ_range = maximum(result.τ_star_coeffs.values) - minimum(result.τ_star_coeffs.values)
    println("  τ* range (should be ≈ 0) = $(round(τ_range; sigdigits=4))")
    @test τ_range < 1.0  # < 1 day of variation
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Continuation value V(t) ≈ V* (constant)" begin
    V_values = result.V_values
    V_mean = sum(result.V_coeffs.values) / length(result.V_coeffs.values)

    # Mean should match homogeneous V*
    rel_err_mean = abs(V_mean - V_hom) / abs(V_hom)
    println("  V̄ (spline mean) = $(round(V_mean; digits=2))")
    println("  V* (homogeneous) = $(round(V_hom; digits=2))")
    println("  Relative error = $(round(rel_err_mean * 100; digits=4))%")
    @test rel_err_mean < 0.01  # < 1% error

    # All nodal values should be close to V*
    max_node_err = maximum(abs.(V_values .- V_hom)) / abs(V_hom)
    println("  Max nodal relative error = $(round(max_node_err * 100; digits=4))%")
    @test max_node_err < 0.01

    # Variation should be near zero (constant in homogeneous case)
    V_range = maximum(result.V_coeffs.values) - minimum(result.V_coeffs.values)
    rel_range = V_range / abs(V_mean)
    println("  Relative range = $(round(rel_range * 100; digits=4))%")
    @test rel_range < 0.01
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Fallow duration d*(t) = 0 (all corner solutions)" begin
    d_values = result.d_values

    n_corner = count(d -> d == 0.0, d_values)
    n_total = length(d_values)
    max_d = maximum(d_values)

    println("  Corner solutions: $n_corner / $n_total")
    println("  Max fallow duration = $(round(max_d; digits=4)) days")

    # In the homogeneous case, Ṽ'(t₀) = 0 (constant), so residual = -δ·Ṽ < 0
    # which means d* = 0 at every node (corner solution)
    @test n_corner == n_total
    @test max_d == 0.0
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Ṽ(t₀) ≈ constant" begin
    Vtilde_values = result.Vtilde_at_t0_nodes
    Vtilde_mean = sum(result.Vtilde_coeffs.values) / length(result.Vtilde_coeffs.values)

    # Ṽ should be approximately constant
    Vtilde_range = maximum(Vtilde_values) - minimum(Vtilde_values)
    rel_range = Vtilde_range / abs(Vtilde_mean)
    println("  Ṽ̄ (spline mean) = $(round(Vtilde_mean; digits=2))")
    println("  Ṽ range = $(round(Vtilde_range; sigdigits=4))")
    println("  Relative range = $(round(rel_range * 100; digits=4))%")
    @test rel_range < 0.01

    # Variation should be near zero
    Vt_range = maximum(result.Vtilde_coeffs.values) - minimum(result.Vtilde_coeffs.values)
    rel_vt_range = Vt_range / abs(Vtilde_mean)
    println("  Relative spline range = $(round(rel_vt_range * 100; digits=4))%")
    @test rel_vt_range < 0.01
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Value linkage: V = Ṽ when d* = 0" begin
    # When d* = 0, V(t) = e^{-δ·0} · Ṽ(t) = Ṽ(t)
    # So the V and Ṽ splines should be nearly identical
    V_mean = sum(result.V_coeffs.values) / length(result.V_coeffs.values)
    Vtilde_mean = sum(result.Vtilde_coeffs.values) / length(result.Vtilde_coeffs.values)

    rel_diff = abs(V_mean - Vtilde_mean) / abs(V_mean)
    println("  V̄ = $(round(V_mean; digits=2))")
    println("  Ṽ̄ = $(round(Vtilde_mean; digits=2))")
    println("  Relative difference = $(round(rel_diff * 100; digits=4))%")
    @test rel_diff < 0.01
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Cold-start convergence" begin
    # Run from default initial guess (no warm start) to verify the solver
    # can find the correct solution without prior knowledge
    println("  Running cold-start solver...")
    result_cold = solve_seasonal_model(hom_seasonal_params;
        N = N_TEST,
        max_iter = 500,
        tol = 1e-4,
        damping = 1.0,
        verbose = false,
    )

    @test result_cold.converged
    println("  Converged in $(result_cold.iterations) iterations")

    # V should match homogeneous V* within 1%
    V_mean = sum(result_cold.V_coeffs.values) / length(result_cold.V_coeffs.values)
    V_err = abs(V_mean - V_hom) / abs(V_hom)
    println("  V̄ = $(round(V_mean; digits=2)), " *
            "V* = $(round(V_hom; digits=2)), error = $(round(V_err * 100; digits=4))%")
    @test V_err < 0.01

    # τ should match homogeneous T* within 1%
    τ_mean = sum(result_cold.τ_star_coeffs.values) / length(result_cold.τ_star_coeffs.values)
    τ_err = abs(τ_mean - T_star_hom) / T_star_hom
    println("  τ̄ = $(round(τ_mean; digits=2)), " *
            "T* = $(round(T_star_hom; digits=2)), error = $(round(τ_err * 100; digits=4))%")
    @test τ_err < 0.01

    # All d* should be 0
    n_corner = count(d -> d == 0.0, result_cold.d_values)
    println("  Corner solutions: $n_corner / $(length(result_cold.d_values))")
    @test n_corner == length(result_cold.d_values)
end

# ══════════════════════════════════════════════════════════════════════════════
end # top-level testset
