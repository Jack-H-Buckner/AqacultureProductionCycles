"""
    test_homogeneous.jl

Unit tests for the four homogeneous (constant hazard) validation cases
defined in src/01_homogeneous_case.jl. Each test verifies that the solver
runs successfully, returns a finite optimal rotation T*, and that the FOC
residual is near zero at the solution.
"""

using Test

# Load model code (paths relative to project root)
include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build case-specific parameter sets from homogeneous_params
# ──────────────────────────────────────────────────────────────────────────────

# Case 1 — Classical Reed: risk-neutral (γ → 0 gives u(x)=x via CRRA limit),
# no stocking cost, no feed costs, no insurance.
# With γ=0 the CRRA u(Y) = Y^(1-0)/(1-0) = Y (linear utility).
const reed_params = merge(homogeneous_params, (
    γ   = 0.0,      # risk-neutral (u(x) = x)
    c_s = 0.0,      # no stocking cost
    η   = 0.0,      # no feed costs
    Y_MIN = 1.0,    # positive loss income to avoid u(0) issues
))

# Case 2 — Risk aversion: CRRA with γ = 0.1 (mild risk aversion).
# Note: higher γ values (e.g. 0.5) can eliminate the interior optimum
# with these growth parameters, making immediate harvest optimal.
const ra_params = merge(homogeneous_params, (
    γ     = 0.1,    # mild risk aversion
    η     = 0.0,    # no feed costs
    Y_MIN = 100.0,  # positive constant loss income
))

# Case 3 — Feed costs: same as Case 2 but with η > 0
const feed_params = merge(homogeneous_params, (
    γ     = 0.1,    # mild risk aversion
    Y_MIN = 100.0,  # positive constant loss income
))

# Case 4 — Insurance: full model with breakeven coverage
const ins_params = merge(homogeneous_params, (
    γ     = 0.1,    # mild risk aversion
    Y_MIN = 1000.0, # higher floor to keep loss income positive with insurance costs
))


# ══════════════════════════════════════════════════════════════════════════════
@testset "Homogeneous Model Cases" begin
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
@testset "Case 1: Classical Reed" begin
    # Growth function should be positive and increasing at early ages
    @test v_homogeneous(0.0, reed_params) > 0
    @test v_homogeneous(100.0, reed_params) > v_homogeneous(0.0, reed_params)

    # Solve for optimal rotation
    T_star = solve_reed(reed_params)
    @test isfinite(T_star)
    @test T_star > 0

    # FOC residual should be near zero at the solution
    @test abs(reed_foc(T_star, reed_params)) < 1e-6

    # Continuation value should be finite and positive
    V = reed_value(T_star, reed_params)
    @test isfinite(V)
    @test V > 0

    println("  Reed T* = $(round(T_star, digits=1)) days, V = $(round(V, digits=2))")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Case 2: Risk Aversion (CRRA)" begin
    # Harvest income should be positive at some reasonable T
    @test Y_H_homogeneous(300.0, ra_params) > 0

    # Solve for optimal rotation
    T_star = solve_risk_aversion(ra_params)
    @test isfinite(T_star)
    @test T_star > 0

    # FOC residual should be near zero
    @test abs(risk_aversion_foc(T_star, ra_params)) < 1e-6

    # Continuation value should be finite
    V = risk_aversion_value(T_star, ra_params)
    @test isfinite(V)

    println("  Risk-aversion T* = $(round(T_star, digits=1)) days, V = $(round(V, digits=2))")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Case 3: Feed Costs" begin
    # Accumulated feed cost should be positive
    Φ = Φ_homogeneous(300.0, feed_params)
    @test isfinite(Φ)
    @test Φ > 0

    # Harvest income with feed costs should be less than without
    Y_no_feed = Y_H_homogeneous(300.0, feed_params)
    Y_with_feed = Y_H_feed(300.0, feed_params)
    @test Y_with_feed < Y_no_feed

    # Solve for optimal rotation
    T_star = solve_feed_cost(feed_params)
    @test isfinite(T_star)
    @test T_star > 0

    # FOC residual should be near zero
    @test abs(feed_cost_foc(T_star, feed_params)) < 1e-6

    # Continuation value should be finite
    V = feed_cost_value(T_star, feed_params)
    @test isfinite(V)

    println("  Feed-cost T* = $(round(T_star, digits=1)) days, V = $(round(V, digits=2))")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Case 4: Insurance" begin
    # Solve indemnity ODE over a test horizon
    T_test = 300.0
    I_sol = solve_indemnity_homogeneous(T_test, ins_params)

    # Indemnity should start at Y_MIN + c_s + c₂ and increase
    I_0 = I_sol(0.0)
    @test I_0 ≈ ins_params.Y_MIN + ins_params.c_s + ins_params.c₂
    @test I_sol(T_test) > I_0

    # Premium rate should be positive
    π_val = π_homogeneous(100.0, I_sol, ins_params)
    @test isfinite(π_val)
    @test π_val > 0

    # Accumulated premiums should be positive
    Π_val = Π_homogeneous(T_test, I_sol, ins_params)
    @test isfinite(Π_val)
    @test Π_val > 0

    # Solve for optimal rotation
    T_star = solve_insurance(ins_params)
    @test isfinite(T_star)
    @test T_star > 0

    # FOC residual should be near zero
    @test abs(insurance_foc(T_star, ins_params)) < 1e-6

    # Continuation value should be finite
    I_sol_star = solve_indemnity_homogeneous(T_star, ins_params)
    V = insurance_value(T_star, I_sol_star, ins_params)
    @test isfinite(V)

    println("  Insurance T* = $(round(T_star, digits=1)) days, V = $(round(V, digits=2))")
end

# ══════════════════════════════════════════════════════════════════════════════
end # top-level testset
