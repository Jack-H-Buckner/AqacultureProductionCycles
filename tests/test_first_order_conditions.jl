"""
    test_first_order_conditions.jl

Tests for the seasonal first-order conditions in src/02_first_order_conditions.jl.

Tests the accuracy of Fourier interpolation by:
1. Solving the harvest FOC at Fourier nodes and fitting a Fourier series
2. Solving the harvest FOC on a fine grid (the "exact" solution)
3. Comparing the Fourier series to the fine-grid solution
4. Evaluating the stocking FOC residual (expected to be a corner solution)

Uses an approximate continuation value V(t) = V_homogeneous + A·sin(2πt/365)
to introduce seasonal variation without requiring the full iterative solver.
"""

using Test
using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "02_first_order_conditions.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Setup: construct approximate V(t) from homogeneous solution + perturbation
# ──────────────────────────────────────────────────────────────────────────────

# Use insurance case parameters for the homogeneous baseline
const test_p = merge(default_params, (
    γ     = 0.1,
    Y_MIN = 1000.0,
))

# Get homogeneous continuation value as baseline
const hom_p = merge(homogeneous_params, (
    γ     = 0.1,
    Y_MIN = 1000.0,
))
const T_hom = solve_insurance(hom_p)
const I_sol_hom = solve_indemnity_homogeneous(T_hom, hom_p)
const V_hom = insurance_value(T_hom, I_sol_hom, hom_p)

# Perturbation amplitude: 1% of V_hom
const A_perturb = 0.01 * V_hom

# Number of harmonics for FOC approximations
const N_FOC = 40

# Approximate V(t) = V_hom + A·sin(2πt/365)
const V_coeffs_test = (
    a0 = V_hom,
    a  = vcat([A_perturb], zeros(N_FOC - 1)),
    b  = zeros(N_FOC),
)


# ══════════════════════════════════════════════════════════════════════════════
@testset "Seasonal FOC Tests" begin
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
@testset "Fourier fitting roundtrip" begin
    true_coeffs = (
        a0 = 300.0,
        a  = vcat([5.0, -2.0, 1.0], zeros(N_FOC - 3)),
        b  = vcat([3.0, 1.5], zeros(N_FOC - 2)),
    )

    nodes = fourier_nodes(N_FOC)
    values = [fourier_eval(t, true_coeffs) for t in nodes]
    recovered = fit_fourier(nodes, values, N_FOC)

    @test recovered.a0 ≈ true_coeffs.a0 atol=1e-10
    for k in 1:N_FOC
        @test recovered.a[k] ≈ true_coeffs.a[k] atol=1e-10
        @test recovered.b[k] ≈ true_coeffs.b[k] atol=1e-10
    end
    println("  Fourier roundtrip: passed")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Harvest FOC at nodes" begin
    t₀_test = 100.0  # pick a node away from edge cases
    T_star = solve_harvest_foc(t₀_test, V_coeffs_test, test_p)
    τ_star = T_star - t₀_test

    @test isfinite(T_star)
    @test τ_star > 100
    @test τ_star < 1500.0

    # Check FOC residual is near zero at solution
    cycle = prepare_cycle(t₀_test, T_star + 10.0, test_p)
    resid = harvest_foc_residual(T_star, t₀_test, V_coeffs_test,
                                  cycle.L_sol, cycle.n_sol, cycle.I_sol, test_p)
    @test abs(resid) < 100.0  # numerical tolerance for seasonal case

    println("  Harvest FOC at t₀=100: τ* = $(round(τ_star, digits=1)) days, residual = $(round(resid, sigdigits=3))")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Harvest FOC Fourier interpolation" begin
    # Step 1: Solve at Fourier nodes
    println("  Solving harvest FOC at $(2N_FOC+1) Fourier nodes...")
    result = solve_harvest_at_nodes(V_coeffs_test, test_p; N=N_FOC)
    τ_star_coeffs = result.τ_star_coeffs

    println("  τ* range at nodes: $(round(minimum(result.τ_values), digits=1)) — $(round(maximum(result.τ_values), digits=1)) days")

    # Step 2: Solve on a fine grid for comparison
    n_test = 20
    println("  Solving harvest FOC on $(n_test)-point comparison grid...")
    grid = solve_harvest_on_grid(V_coeffs_test, test_p; n_grid=n_test)

    # Step 3: Compare Fourier interpolation to grid solutions
    errors = Float64[]
    for i in 1:length(grid.t₀_grid)
        t₀ = grid.t₀_grid[i]
        τ_fourier = fourier_eval(t₀, τ_star_coeffs)
        τ_exact = grid.τ_grid[i]
        push!(errors, abs(τ_fourier - τ_exact))
    end

    max_err = maximum(errors)
    mean_err = sum(errors) / length(errors)
    println("  Interpolation error: max = $(round(max_err, digits=2)) days, mean = $(round(mean_err, digits=2)) days")

    # The Fourier interpolation should capture most of the variation
    @test mean_err < 20.0  # mean error under 20 days
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Cycle value Ṽ(t₀)" begin
    t₀_test = 100.0
    T_star = solve_harvest_foc(t₀_test, V_coeffs_test, test_p)

    Vtilde = compute_Vtilde(t₀_test, T_star, V_coeffs_test, test_p)
    @test isfinite(Vtilde)
    @test Vtilde > 0

    println("  Ṽ(t₀=100) = $(round(Vtilde, digits=2)) with T*=$(round(T_star, digits=1))")
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Stocking FOC (corner solution)" begin
    # Solve harvest FOC first
    println("  Solving harvest FOC at nodes for stocking test...")
    result = solve_harvest_at_nodes(V_coeffs_test, test_p; N=N_FOC)
    τ_star_coeffs = result.τ_star_coeffs

    # Evaluate stocking FOC at a few test points
    test_t0s = [0.0, 100.0, 200.0]
    for t₀ in test_t0s
        τ_star = fourier_eval(t₀, τ_star_coeffs)
        T_star = t₀ + τ_star
        Vtilde = compute_Vtilde(t₀, T_star, V_coeffs_test, test_p)
        @test isfinite(Vtilde)
        @test Vtilde > 0

        resid = stocking_foc_residual(t₀, τ_star_coeffs, V_coeffs_test, test_p)
        println("  Stocking FOC at t₀=$(round(t₀)): Ṽ=$(round(Vtilde, digits=1)), " *
                "residual=$(round(resid, sigdigits=3)) " *
                "($(resid < 0 ? "corner: restock immediately" : "interior solution"))")
    end
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Stocking FOC Fourier interpolation" begin
    # Step 1: Solve harvest FOC at nodes
    println("  Solving harvest FOC at nodes...")
    harvest_result = solve_harvest_at_nodes(V_coeffs_test, test_p; N=N_FOC)
    τ_star_coeffs = harvest_result.τ_star_coeffs

    # Step 2: Evaluate stocking FOC at Fourier nodes
    println("  Evaluating stocking FOC at $(2N_FOC+1) nodes...")
    stocking_nodes = evaluate_stocking_foc_at_nodes(τ_star_coeffs, V_coeffs_test, test_p; N=N_FOC)

    # Fit Fourier series to stocking FOC residuals and Ṽ values
    nodes = stocking_nodes.nodes
    resid_coeffs = fit_fourier(nodes, stocking_nodes.residuals, N_FOC)
    Vtilde_coeffs = fit_fourier(nodes, stocking_nodes.Vtilde_values, N_FOC)

    println("  Ṽ range: $(round(minimum(stocking_nodes.Vtilde_values), digits=1)) — " *
            "$(round(maximum(stocking_nodes.Vtilde_values), digits=1))")
    println("  Residual range: $(round(minimum(stocking_nodes.residuals), sigdigits=3)) — " *
            "$(round(maximum(stocking_nodes.residuals), sigdigits=3))")

    # Step 3: Evaluate on a fine grid for comparison
    n_test = 20
    println("  Evaluating stocking FOC on $(n_test)-point grid...")
    stocking_grid = evaluate_stocking_foc_on_grid(τ_star_coeffs, V_coeffs_test, test_p; n_grid=n_test)

    # Step 4: Compare Fourier interpolation to grid values
    resid_errors = Float64[]
    Vtilde_errors = Float64[]
    for i in 1:length(stocking_grid.t₀_grid)
        t₀ = stocking_grid.t₀_grid[i]
        # Fourier-interpolated values
        resid_fourier = fourier_eval(t₀, resid_coeffs)
        Vtilde_fourier = fourier_eval(t₀, Vtilde_coeffs)
        # Exact values
        resid_exact = stocking_grid.residuals[i]
        Vtilde_exact = stocking_grid.Vtilde_values[i]

        push!(resid_errors, abs(resid_fourier - resid_exact))
        push!(Vtilde_errors, abs(Vtilde_fourier - Vtilde_exact) / abs(Vtilde_exact))  # relative
    end

    println("  Ṽ relative error: max = $(round(maximum(Vtilde_errors)*100, digits=3))%, " *
            "mean = $(round(sum(Vtilde_errors)/length(Vtilde_errors)*100, digits=3))%")
    println("  Residual abs error: max = $(round(maximum(resid_errors), sigdigits=3)), " *
            "mean = $(round(sum(resid_errors)/length(resid_errors), sigdigits=3))")

    # Ṽ should be well-approximated
    @test sum(Vtilde_errors) / length(Vtilde_errors) < 0.01  # mean relative error < 1%

    # Check that most stocking dates are corner solutions
    n_corner = count(r -> r < 0, stocking_grid.residuals)
    println("  Corner solutions: $(n_corner)/$(length(stocking_grid.residuals)) dates")
    @test n_corner > 0  # at least some corner solutions
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Fallow duration d*(T) Fourier interpolation" begin
    # Step 1: Solve harvest FOC at nodes
    println("  Solving harvest FOC at nodes...")
    harvest_result = solve_harvest_at_nodes(V_coeffs_test, test_p; N=N_FOC)
    τ_star_coeffs = harvest_result.τ_star_coeffs

    # Step 2: Solve stocking FOC at Fourier nodes
    println("  Solving stocking FOC at $(2N_FOC+1) nodes...")
    stocking_result = solve_stocking_at_nodes(τ_star_coeffs, V_coeffs_test, test_p; N=N_FOC)
    d_star_coeffs = stocking_result.d_star_coeffs

    n_interior = count(d -> d > 0, stocking_result.d_values)
    n_corner = count(d -> d == 0, stocking_result.d_values)
    println("  Interior solutions: $(n_interior)/$(length(stocking_result.d_values)), " *
            "Corner solutions: $(n_corner)/$(length(stocking_result.d_values))")

    if n_interior > 0
        d_pos = filter(d -> d > 0, stocking_result.d_values)
        println("  Fallow duration range (interior): $(round(minimum(d_pos), digits=1)) — $(round(maximum(d_pos), digits=1)) days")
    end

    # Step 3: Compare Fourier interpolation to grid solutions
    n_test = 50
    println("  Solving stocking FOC on $(n_test)-point grid...")
    stocking_grid = solve_stocking_on_grid(τ_star_coeffs, V_coeffs_test, test_p; n_grid=n_test)

    errors = Float64[]
    for i in 1:length(stocking_grid.t₀_grid)
        t₀ = stocking_grid.t₀_grid[i]
        d_fourier = max(0.0, fourier_eval(t₀, d_star_coeffs))  # clamp negative to 0
        d_exact = stocking_grid.d_grid[i]
        push!(errors, abs(d_fourier - d_exact))
    end

    max_err = maximum(errors)
    mean_err = sum(errors) / length(errors)
    println("  Interpolation error: max = $(round(max_err, digits=2)) days, mean = $(round(mean_err, digits=2)) days")

    @test mean_err < 30.0  # mean error under 30 days (sharp transition is hard to approximate)
    @test n_interior > 0   # at least some interior solutions exist
end

# ──────────────────────────────────────────────────────────────────────────
@testset "Export comparison data" begin
    outdir = joinpath(@__DIR__, "..", "results", "simulations")
    mkpath(outdir)

    # ── Harvest FOC ──────────────────────────────────────────────────────
    println("  Solving harvest FOC at nodes for export...")
    result = solve_harvest_at_nodes(V_coeffs_test, test_p; N=N_FOC)
    τ_star_coeffs = result.τ_star_coeffs

    n_grid = 100
    println("  Solving harvest FOC on $(n_grid)-point grid for export...")
    grid = solve_harvest_on_grid(V_coeffs_test, test_p; n_grid=n_grid)

    τ_fourier = [fourier_eval(t, τ_star_coeffs) for t in grid.t₀_grid]

    df = DataFrame(
        t0          = grid.t₀_grid,
        tau_exact   = grid.τ_grid,
        tau_fourier = τ_fourier,
    )
    CSV.write(joinpath(outdir, "harvest_foc_comparison.csv"), df)

    node_df = DataFrame(t0 = result.nodes, tau = result.τ_values)
    CSV.write(joinpath(outdir, "harvest_foc_nodes.csv"), node_df)

    println("  Wrote harvest_foc_comparison.csv ($(nrow(df)) rows)")
    println("  Wrote harvest_foc_nodes.csv ($(nrow(node_df)) rows)")

    # ── Stocking FOC ─────────────────────────────────────────────────────
    println("  Evaluating stocking FOC on $(n_grid)-point grid for export...")
    stocking_grid = evaluate_stocking_foc_on_grid(τ_star_coeffs, V_coeffs_test, test_p; n_grid=n_grid)

    println("  Evaluating stocking FOC at nodes for export...")
    stocking_nodes = evaluate_stocking_foc_at_nodes(τ_star_coeffs, V_coeffs_test, test_p; N=N_FOC)
    resid_coeffs = fit_fourier(stocking_nodes.nodes, stocking_nodes.residuals, N_FOC)
    Vtilde_coeffs_fit = fit_fourier(stocking_nodes.nodes, stocking_nodes.Vtilde_values, N_FOC)

    stocking_df = DataFrame(
        t0             = stocking_grid.t₀_grid,
        residual_exact = stocking_grid.residuals,
        residual_fourier = [fourier_eval(t, resid_coeffs) for t in stocking_grid.t₀_grid],
        Vtilde_exact   = stocking_grid.Vtilde_values,
        Vtilde_fourier = [fourier_eval(t, Vtilde_coeffs_fit) for t in stocking_grid.t₀_grid],
    )
    CSV.write(joinpath(outdir, "stocking_foc_comparison.csv"), stocking_df)

    stocking_node_df = DataFrame(
        t0       = stocking_nodes.nodes,
        residual = stocking_nodes.residuals,
        Vtilde   = stocking_nodes.Vtilde_values,
    )
    CSV.write(joinpath(outdir, "stocking_foc_nodes.csv"), stocking_node_df)

    println("  Wrote stocking_foc_comparison.csv ($(nrow(stocking_df)) rows)")
    println("  Wrote stocking_foc_nodes.csv ($(nrow(stocking_node_df)) rows)")

    # ── Fallow duration d*(T) ───────────────────────────────────────────
    println("  Solving stocking FOC (fallow duration) at nodes for export...")
    stocking_sol = solve_stocking_at_nodes(τ_star_coeffs, V_coeffs_test, test_p; N=N_FOC)
    d_star_coeffs = stocking_sol.d_star_coeffs

    println("  Solving stocking FOC (fallow duration) on $(n_grid)-point grid...")
    fallow_grid = solve_stocking_on_grid(τ_star_coeffs, V_coeffs_test, test_p; n_grid=n_grid)

    d_fourier = [max(0.0, fourier_eval(t, d_star_coeffs)) for t in fallow_grid.t₀_grid]

    fallow_df = DataFrame(
        T_harvest     = fallow_grid.t₀_grid,
        d_exact       = fallow_grid.d_grid,
        d_fourier     = d_fourier,
        Vtilde        = fallow_grid.Vtilde_values,
        residual_at_t0star = fallow_grid.residuals,
    )
    CSV.write(joinpath(outdir, "fallow_duration_comparison.csv"), fallow_df)

    fallow_node_df = DataFrame(
        T_harvest = stocking_sol.nodes,
        d_star    = stocking_sol.d_values,
        Vtilde    = stocking_sol.Vtilde_values,
        residual  = stocking_sol.residuals,
    )
    CSV.write(joinpath(outdir, "fallow_duration_nodes.csv"), fallow_node_df)

    println("  Wrote fallow_duration_comparison.csv ($(nrow(fallow_df)) rows)")
    println("  Wrote fallow_duration_nodes.csv ($(nrow(fallow_node_df)) rows)")
    @test true
end

# ══════════════════════════════════════════════════════════════════════════════
end # top-level testset
