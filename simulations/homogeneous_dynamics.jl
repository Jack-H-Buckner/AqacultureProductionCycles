"""
    homogeneous_dynamics.jl

Generate time-series data for the homogeneous model dynamics, exported as
CSV for plotting in R. Produces two output files:

1. `growth_dynamics.csv` — survival, weight, f_value, stock value v over time
2. `cost_dynamics.csv` — accumulated feed and insurance costs over time
"""

using CSV, DataFrames

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "01_homogeneous_case.jl"))

# Use the insurance parameter set (full model with all cost components)
p = merge(homogeneous_params, (
    γ     = 0.1,
    Y_MIN = 1000.0,
))

# Solve for T* to know the relevant time horizon
T_star = solve_insurance(p)
T_max = 450.0  # fixed horizon for growth and cost plots

# Time grid
ts = range(0.0, T_max, length=500)

# ── Growth dynamics ───────────────────────────────────────────────────────────
survival = [exp(-p.λ_const * t) for t in ts]
lengths  = [p.L∞ - (p.L∞ - p.L₀) * exp(-p.k_const * t) for t in ts]
weights  = [W_weight(L, p) for L in lengths]
fvalues  = [f_value(L, p) for L in lengths]
numbers  = [p.n₀ * exp(-p.m_const * t) for t in ts]
vvalues  = [v_homogeneous(t, p) for t in ts]

growth_df = DataFrame(
    day       = collect(ts),
    survival  = survival,
    length_cm = lengths,
    weight_g  = weights,
    f_value   = fvalues,
    n_fish    = numbers,
    v_stock   = vvalues,
)

# ── Cost dynamics ─────────────────────────────────────────────────────────────
# Solve indemnity ODE over full horizon
I_sol = solve_indemnity_homogeneous(T_max, p)

feed_acc = Float64[]
prem_acc = Float64[]
indemnity = Float64[]

for t in ts
    if t ≈ 0.0
        push!(feed_acc, 0.0)
        push!(prem_acc, 0.0)
    else
        push!(feed_acc, Φ_homogeneous(t, p))
        push!(prem_acc, Π_homogeneous(t, I_sol, p))
    end
    push!(indemnity, I_sol(t))
end

cost_df = DataFrame(
    day                = collect(ts),
    feed_accumulated   = feed_acc,
    premium_accumulated = prem_acc,
    indemnity          = indemnity,
    stocking_compounded = [p.c_s * exp(p.δ_b * t) for t in ts],
)

# ── FOC components for all four cases ────────────────────────────────────────
δλ = p.δ + p.λ_const

# Build case-specific parameter sets (matching test_homogeneous.jl)
reed_p = merge(homogeneous_params, (γ = 0.0, c_s = 0.0, η = 0.0, Y_MIN = 1.0))
ra_p   = merge(homogeneous_params, (γ = 0.1, η = 0.0, Y_MIN = 100.0))
feed_p = merge(homogeneous_params, (γ = 0.1, Y_MIN = 100.0))
ins_p  = p  # already has γ=0.1, Y_MIN=1000

# Solve for T* in each case
T_reed = solve_reed(reed_p)
T_ra   = solve_risk_aversion(ra_p)
T_feed = solve_feed_cost(feed_p)
T_ins  = T_star  # already solved above

# Use common time grid spanning all cases
T_foc_max = max(T_reed, T_ra, T_feed, T_ins) * 1.2
ts_foc = range(1.0, T_foc_max, length=500)

# Case 1: Reed — LHS = v'(T), RHS = (δ+λ)·v(T) / (1 − exp(−(δ+λ)T))
reed_δλ = reed_p.δ + reed_p.λ_const
reed_lhs = [v_homogeneous_prime(t, reed_p) for t in ts_foc]
reed_rhs = [reed_δλ * v_homogeneous(t, reed_p) / (1 - exp(-reed_δλ * t)) for t in ts_foc]

# Case 2: Risk aversion — LHS = Y_H'·u'(Y_H), RHS = (δ+λ)·u(Y_H) / (1 − exp(−(δ+λ)T))
ra_δλ = ra_p.δ + ra_p.λ_const
ra_lhs = Float64[]
ra_rhs = Float64[]
for t in ts_foc
    yh = Y_H_homogeneous(t, ra_p)
    if yh > 0
        push!(ra_lhs, Y_H_homogeneous_prime(t, ra_p) * u_prime(yh, ra_p))
        push!(ra_rhs, ra_δλ * u(yh, ra_p) / (1 - exp(-ra_δλ * t)))
    else
        push!(ra_lhs, NaN)
        push!(ra_rhs, NaN)
    end
end

# Case 3: Feed costs — same form but Y_H includes feed costs
feed_δλ = feed_p.δ + feed_p.λ_const
feed_lhs = Float64[]
feed_rhs = Float64[]
for t in ts_foc
    yh = Y_H_feed(t, feed_p)
    if yh > 0
        push!(feed_lhs, Y_H_feed_prime(t, feed_p) * u_prime(yh, feed_p))
        push!(feed_rhs, feed_δλ * u(yh, feed_p) / (1 - exp(-feed_δλ * t)))
    else
        push!(feed_lhs, NaN)
        push!(feed_rhs, NaN)
    end
end

# Case 4: Insurance — same form but Y_H includes feed + insurance costs
ins_δλ = ins_p.δ + ins_p.λ_const
I_sol_foc = solve_indemnity_homogeneous(T_foc_max, ins_p)
ins_lhs = Float64[]
ins_rhs = Float64[]
for t in ts_foc
    yh = Y_H_insurance(t, I_sol_foc, ins_p)
    if yh > 0
        push!(ins_lhs, Y_H_insurance_prime(t, I_sol_foc, ins_p) * u_prime(yh, ins_p))
        push!(ins_rhs, ins_δλ * u(yh, ins_p) / (1 - exp(-ins_δλ * t)))
    else
        push!(ins_lhs, NaN)
        push!(ins_rhs, NaN)
    end
end

# Combine into long-format DataFrame
n_ts = length(ts_foc)
foc_df = DataFrame(
    day  = repeat(collect(ts_foc), 4),
    LHS  = vcat(reed_lhs, ra_lhs, feed_lhs, ins_lhs),
    RHS  = vcat(reed_rhs, ra_rhs, feed_rhs, ins_rhs),
    case = vcat(
        fill("Case 1: Classical Reed", n_ts),
        fill("Case 2: Risk Aversion", n_ts),
        fill("Case 3: Feed Costs", n_ts),
        fill("Case 4: Insurance", n_ts),
    ),
    T_star = vcat(
        fill(T_reed, n_ts),
        fill(T_ra, n_ts),
        fill(T_feed, n_ts),
        fill(T_ins, n_ts),
    ),
)

# ── Write outputs ─────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "..", "results", "simulations")
mkpath(outdir)

CSV.write(joinpath(outdir, "growth_dynamics.csv"), growth_df)
CSV.write(joinpath(outdir, "cost_dynamics.csv"), cost_df)
CSV.write(joinpath(outdir, "foc_all_cases.csv"), foc_df)

println("T* (Reed)      = $(round(T_reed, digits=1)) days")
println("T* (Risk Av.)  = $(round(T_ra, digits=1)) days")
println("T* (Feed)      = $(round(T_feed, digits=1)) days")
println("T* (Insurance) = $(round(T_ins, digits=1)) days")
println("Wrote growth_dynamics.csv ($(nrow(growth_df)) rows)")
println("Wrote cost_dynamics.csv ($(nrow(cost_df)) rows)")
println("Wrote foc_all_cases.csv ($(nrow(foc_df)) rows)")
