"""
    seasonal_parameters.jl

Export the seasonal parameter functions (growth rate k(t), mortality rate m(t),
and catastrophic hazard rate λ(t)) evaluated over a 365-day cycle to CSV
for visualization.
"""

include(joinpath(@__DIR__, "..", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "00_model_functions.jl"))

using CSV, DataFrames

# Evaluate seasonal functions on a daily grid
days = 0:364
grid = DataFrame(
    t = collect(days),
    k = [k_growth(t, default_params) for t in days],
    m = [m_rate(t, default_params) for t in days],
    lambda = [λ(t, default_params) for t in days],
)

# Also add constant (homogeneous) reference values
grid.k_const .= k_const
grid.m_const .= m_const
grid.lambda_const .= λ_const

outdir = "results/simulations"
mkpath(outdir)
CSV.write(joinpath(outdir, "seasonal_parameters.csv"), grid)
println("Saved seasonal_parameters.csv ($(nrow(grid)) rows)")
