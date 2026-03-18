"""
    04_simulate_production_cycles.jl

Monte Carlo simulation of aquaculture production cycles under the optimal
policy derived by the solver in `03_continuation_value_solver.jl`.

Starting from an initial calendar date, each cycle proceeds as follows:

1. **Fallow**: Look up the optimal fallow duration d*(t) from the converged
   policy splines and wait d* days before restocking.
2. **Stock**: At t₀ = t + d*, stock n₀ fingerlings of length L₀.
3. **Grow**: Solve the growth (L), mortality (n), indemnity (I), feed-cost (Φ),
   and premium (Π) ODEs forward from t₀ to the planned harvest T* = t₀ + τ*(t₀).
4. **Loss sampling**: Draw a catastrophic loss event from the inhomogeneous
   Poisson process with intensity λ(t) using the inverse-CDF method on the
   cumulative hazard.  If the sampled loss time τ_loss falls before T*, the
   cycle ends in a loss at τ_loss; otherwise the firm harvests at T*.
5. **Income**: Compute Y_H (harvest) or Y_L (loss) from the precomputed ODE
   solutions (see README §§ 7–9).
6. **Repeat**: The next cycle begins at the end of the current one (T* or τ_loss).

The main entry point is `simulate_production_cycles`, which returns a vector of
`CycleOutcome` records suitable for conversion to a CSV via DataFrames or
manual column extraction.

Functions provided:
- `sample_loss_time(t₀, T, p, rng)` — inverse-CDF sampling from λ(t)
- `simulate_cycle(t_start, model_result, p, rng)` — simulate one cycle
- `simulate_production_cycles(model_result, p; ...)` — full Monte Carlo run
"""

using Random

include("03_continuation_value_solver.jl")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Loss event sampling via inverse CDF on cumulative hazard
# ──────────────────────────────────────────────────────────────────────────────

"""
    sample_loss_time(t₀, T, p, rng)

Sample the time of the first catastrophic loss event during [t₀, T] from the
inhomogeneous Poisson process with intensity λ(t).

Uses the inverse-CDF method:
1. Draw U ~ Uniform(0, 1).
2. Compute the cumulative hazard Λ(T) = ∫_{t₀}^{T} λ(s) ds.
3. If −log(U) > Λ(T), no event occurs before T → return `nothing`.
4. Otherwise, find τ such that Λ(τ) = −log(U) via bisection on the
   precomputed cumulative hazard ODE solution.

# Returns
- `τ_loss::Float64` if a loss event occurs in [t₀, T], or
- `nothing` if the cycle completes without loss.
"""
function sample_loss_time(t₀, T, Λ_sol, rng)
    U = rand(rng)
    target = -log(U)

    Λ_T = Λ_sol(T)
    if target > Λ_T
        return nothing  # no loss before harvest
    end

    # Bisect to find τ such that Λ(τ) = target
    lo, hi = t₀, T
    for _ in 1:15  # ~2^{-15} precision
        mid = (lo + hi) / 2
        if Λ_sol(mid) < target
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Single-cycle simulation
# ──────────────────────────────────────────────────────────────────────────────

"""
    CycleOutcome

Record of a single simulated production cycle.

# Fields
- `cycle`        : cycle index (1-based)
- `t_start`      : calendar date at start of fallow (end of previous cycle)
- `d_star`       : fallow duration (days)
- `t0`           : stocking date (= t_start + d_star)
- `tau_star`     : planned cycle duration τ*(t₀) (days)
- `T_planned`    : planned harvest date (= t₀ + τ*)
- `t_end`        : actual end date (= T_planned if harvest, τ_loss if loss)
- `duration`     : actual growing days (t_end − t₀)
- `loss`         : whether a catastrophic loss occurred
- `income`       : end-of-cycle income (Y_H or Y_L)
- `harvest_value`: gross stock value v(t_end) at cycle end (0 if loss)
- `length_cm`    : mean fish length at cycle end (cm)
- `weight_g`     : mean fish weight at cycle end (g)
- `numbers`      : surviving fish count at cycle end
- `biomass_kg`   : total biomass at cycle end (kg)
- `feed_cost`    : accumulated feed cost Φ compounded to cycle end
- `premium_cost` : accumulated insurance premium Π compounded to cycle end
- `indemnity`    : insurance indemnity I(t_end) (relevant if loss)
"""
struct CycleOutcome
    cycle::Int
    t_start::Float64
    d_star::Float64
    t0::Float64
    tau_star::Float64
    T_planned::Float64
    t_end::Float64
    duration::Float64
    loss::Bool
    income::Float64
    harvest_value::Float64
    length_cm::Float64
    weight_g::Float64
    numbers::Float64
    biomass_kg::Float64
    feed_cost::Float64
    premium_cost::Float64
    indemnity::Float64
end

"""
    simulate_cycle(t_start, cycle_idx, model_result, p, rng)

Simulate a single production cycle starting from calendar date `t_start`.

1. Look up optimal fallow d*(t_start) and cycle duration τ*(t₀) from the
   converged policy splines in `model_result`.
2. Solve all ODEs from t₀ to T* via `prepare_cycle`.
3. Sample a loss event from the hazard rate λ(t).
4. Compute income and biological state at the cycle endpoint.

# Returns
A `CycleOutcome` struct.
"""
function simulate_cycle(t_start, cycle_idx, model_result, p, rng)
    # Optimal policy from solved model
    d_star = interpolate_d_star(t_start, model_result.nodes, model_result.d_values)
    t0 = t_start + d_star
    tau_star = spline_eval(t0, model_result.τ_star_coeffs)
    T_planned = t0 + tau_star

    # Pre-solve all ODEs for this cycle
    cycle = prepare_cycle(t0, T_planned, p)

    # Sample loss event
    τ_loss = sample_loss_time(t0, T_planned, cycle.Λ_sol, rng)
    loss = !isnothing(τ_loss)
    t_end = loss ? τ_loss : T_planned

    # Biological state at cycle end
    L_end = cycle.L_sol(t_end)
    n_end = cycle.n_sol(t_end)
    W_end = W_weight(L_end, p)
    biomass_kg = n_end * W_end / 1000.0

    # Costs at cycle end
    Φ_end = cycle.Φ_sol(t_end)
    Π_end = cycle.Π_sol(t_end)
    I_end = cycle.I_sol(t_end)

    if loss
        income = Y_L_seasonal(t_end, t0, cycle, p)
        harvest_val = 0.0
    else
        income = Y_H_seasonal(t_end, t0, cycle, p)
        harvest_val = v_seasonal(t_end, cycle.L_sol, cycle.n_sol, p)
    end

    return CycleOutcome(
        cycle_idx, t_start, d_star, t0, tau_star, T_planned, t_end,
        t_end - t0, loss, income, harvest_val,
        L_end, W_end, n_end, biomass_kg,
        Φ_end, Π_end, I_end,
    )
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. Full Monte Carlo simulation
# ──────────────────────────────────────────────────────────────────────────────

"""
    simulate_production_cycles(model_result, p;
        n_cycles = 100,
        n_sims   = 1000,
        t_init   = 0.0,
        seed     = SEED
    )

Simulate `n_sims` independent sample paths, each running `n_cycles` production
cycles under the optimal policy encoded in `model_result`.

# Arguments
- `model_result` : output of `solve_seasonal_model` (must contain V_coeffs,
   τ_star_coeffs, d_values, nodes)
- `p`            : parameter NamedTuple
- `n_cycles`     : number of production cycles per sample path
- `n_sims`       : number of independent sample paths
- `t_init`       : initial calendar date (day of year) for the first cycle
- `seed`         : random seed for reproducibility

# Returns
A `Vector{Vector{CycleOutcome}}` — one inner vector per sample path, each
containing `n_cycles` cycle records.
"""
function simulate_production_cycles(model_result, p;
        n_cycles::Int = 100,
        n_sims::Int   = 1000,
        t_init::Float64 = 0.0,
        seed::Int = SEED,
    )
    rng = MersenneTwister(seed)
    all_paths = Vector{Vector{CycleOutcome}}(undef, n_sims)

    for sim in 1:n_sims
        path = CycleOutcome[]
        t_current = t_init

        for c in 1:n_cycles
            outcome = simulate_cycle(t_current, c, model_result, p, rng)
            push!(path, outcome)
            t_current = outcome.t_end
        end

        all_paths[sim] = path
    end

    return all_paths
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

"""
    summarize_simulations(all_paths)

Compute summary statistics across all simulated sample paths.

# Returns
A NamedTuple with:
- `n_sims`              : number of sample paths
- `n_cycles`            : cycles per path
- `loss_rate`           : fraction of cycles ending in loss
- `mean_income`         : mean cycle income (across all cycles and paths)
- `std_income`          : std. dev. of cycle income
- `mean_harvest_income` : mean income conditional on harvest
- `mean_loss_income`    : mean income conditional on loss
- `mean_duration`       : mean growing days per cycle
- `mean_fallow`         : mean fallow duration
- `mean_biomass_kg`     : mean biomass at cycle end (kg)
- `mean_weight_g`       : mean individual fish weight at cycle end (g)
- `expected_utility`    : mean across paths of Σ_c e^{−δ·t_end_c} · u(Y_c),
                          the discounted sum of per-cycle utilities
"""
function summarize_simulations(all_paths, p)
    incomes = Float64[]
    harvest_incomes = Float64[]
    loss_incomes = Float64[]
    durations = Float64[]
    fallows = Float64[]
    biomasses = Float64[]
    weights = Float64[]
    n_losses = 0
    n_total = 0

    # Expected present utility: for each path, sum e^{-δ·t_end} · u(Y) over cycles
    path_utilities = Float64[]

    for path in all_paths
        path_u = 0.0
        for outcome in path
            n_total += 1
            push!(incomes, outcome.income)
            push!(durations, outcome.duration)
            push!(fallows, outcome.d_star)
            push!(biomasses, outcome.biomass_kg)
            push!(weights, outcome.weight_g)

            if outcome.loss
                n_losses += 1
                push!(loss_incomes, outcome.income)
            else
                push!(harvest_incomes, outcome.income)
            end

            # Discounted utility: discount to time of first cycle start
            Y = max(outcome.income, 1e-10)
            path_u += exp(-p.δ * outcome.t_end) * u(Y, p)
        end
        push!(path_utilities, path_u)
    end

    return (
        n_sims              = length(all_paths),
        n_cycles            = length(all_paths[1]),
        loss_rate           = n_losses / n_total,
        mean_income         = sum(incomes) / length(incomes),
        std_income          = std(incomes),
        mean_harvest_income = isempty(harvest_incomes) ? NaN : sum(harvest_incomes) / length(harvest_incomes),
        mean_loss_income    = isempty(loss_incomes) ? NaN : sum(loss_incomes) / length(loss_incomes),
        mean_duration       = sum(durations) / length(durations),
        mean_fallow         = sum(fallows) / length(fallows),
        mean_biomass_kg     = sum(biomasses) / length(biomasses),
        mean_weight_g       = sum(weights) / length(weights),
        expected_utility    = sum(path_utilities) / length(path_utilities),
    )
end

"""
    cycles_to_vectors(all_paths)

Flatten all cycle outcomes across all sample paths into column vectors suitable
for writing to CSV.

# Returns
A NamedTuple of `Vector`s, one per field of `CycleOutcome`, plus a `sim` column
identifying the sample path.
"""
function cycles_to_vectors(all_paths)
    sim_ids     = Int[]
    cycles      = Int[]
    t_starts    = Float64[]
    d_stars     = Float64[]
    t0s         = Float64[]
    tau_stars    = Float64[]
    T_planneds  = Float64[]
    t_ends      = Float64[]
    durations   = Float64[]
    losses      = Bool[]
    incomes     = Float64[]
    harvest_vals = Float64[]
    lengths     = Float64[]
    weights_g   = Float64[]
    numbers     = Float64[]
    biomasses   = Float64[]
    feed_costs  = Float64[]
    premium_costs = Float64[]
    indemnities = Float64[]

    for (sim, path) in enumerate(all_paths)
        for o in path
            push!(sim_ids, sim)
            push!(cycles, o.cycle)
            push!(t_starts, o.t_start)
            push!(d_stars, o.d_star)
            push!(t0s, o.t0)
            push!(tau_stars, o.tau_star)
            push!(T_planneds, o.T_planned)
            push!(t_ends, o.t_end)
            push!(durations, o.duration)
            push!(losses, o.loss)
            push!(incomes, o.income)
            push!(harvest_vals, o.harvest_value)
            push!(lengths, o.length_cm)
            push!(weights_g, o.weight_g)
            push!(numbers, o.numbers)
            push!(biomasses, o.biomass_kg)
            push!(feed_costs, o.feed_cost)
            push!(premium_costs, o.premium_cost)
            push!(indemnities, o.indemnity)
        end
    end

    return (
        sim = sim_ids, cycle = cycles, t_start = t_starts, d_star = d_stars,
        t0 = t0s, tau_star = tau_stars, T_planned = T_planneds, t_end = t_ends,
        duration = durations, loss = losses, income = incomes,
        harvest_value = harvest_vals, length_cm = lengths, weight_g = weights_g,
        numbers = numbers, biomass_kg = biomasses, feed_cost = feed_costs,
        premium_cost = premium_costs, indemnity = indemnities,
    )
end


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dense within-cycle time series for a single sample path
# ──────────────────────────────────────────────────────────────────────────────

"""
    simulate_path_timeseries(model_result, p;
        n_cycles = 10, t_init = 0.0, seed = SEED, n_pts = 50)

Simulate a single sample path of `n_cycles` production cycles and return dense
within-cycle time series data suitable for plotting.

For each cycle, the growth/mortality/cost ODEs are evaluated at `n_pts` equally
spaced points between stocking (t₀) and cycle end (T* or τ_loss). Fallow
periods appear as gaps between cycles.

# Returns
A NamedTuple of `Vector`s with one row per time point:
- `t`             : absolute calendar time (days)
- `cycle`         : cycle index
- `biomass_kg`    : n(t)·W(L(t)) / 1000
- `stock_value`   : n(t)·f(L(t))
- `feed_cost`     : accumulated feed cost Φ(t)
- `premium_cost`  : accumulated insurance premium Π(t)
- `premium_rate`  : instantaneous premium rate π(t)
- `hazard_rate`   : catastrophic hazard λ(t)
- `indemnity`     : insurance indemnity I(t)

Plus a NamedTuple `cycle_endpoints` with one row per cycle for point data:
- `t_end`, `cycle`, `loss`, `income`, `utility`, `d_star`, `cumulative_fallow`
"""
function simulate_path_timeseries(model_result, p;
        n_cycles::Int = 10,
        t_init::Float64 = 0.0,
        seed::Int = SEED,
        n_pts::Int = 50,
    )
    rng = MersenneTwister(seed)

    # Time series vectors (within-cycle dense data + fallow periods)
    ts_t            = Float64[]
    ts_cycle        = Int[]
    ts_biomass      = Float64[]
    ts_value        = Float64[]
    ts_feed         = Float64[]
    ts_premium      = Float64[]
    ts_premium_rate = Float64[]
    ts_hazard       = Float64[]
    ts_indemnity    = Float64[]
    ts_fallow_days  = Float64[]
    ts_phase        = String[]   # "fallow" or "production"

    # Cycle endpoint vectors (one per cycle)
    ep_t_end   = Float64[]
    ep_cycle   = Int[]
    ep_loss    = Bool[]
    ep_income  = Float64[]
    ep_utility = Float64[]
    ep_d_star  = Float64[]

    t_current = t_init
    n_fallow_pts = max(div(n_pts, 5), 3)  # fewer points for fallow stretches

    for c in 1:n_cycles
        # Optimal policy
        d_star = interpolate_d_star(t_current, model_result.nodes, model_result.d_values)
        t0 = t_current + d_star
        tau_star = spline_eval(t0, model_result.τ_star_coeffs)
        T_planned = t0 + tau_star

        # ── Fallow period [t_current, t₀] ────────────────────────────────────
        if d_star > 0.5  # only emit points for non-trivial fallow
            fallow_grid = range(t_current, t0, length=n_fallow_pts)
            for t in fallow_grid
                push!(ts_t, t)
                push!(ts_cycle, c)
                push!(ts_biomass, 0.0)
                push!(ts_value, 0.0)
                push!(ts_feed, 0.0)
                push!(ts_premium, 0.0)
                push!(ts_premium_rate, 0.0)
                push!(ts_hazard, λ(t, p))
                push!(ts_indemnity, 0.0)
                push!(ts_fallow_days, t - t_current)
                push!(ts_phase, "fallow")
            end
        end

        # ── Production period [t₀, t_end] ────────────────────────────────────
        # Solve ODEs
        cycle = prepare_cycle(t0, T_planned, p)

        # Sample loss
        τ_loss = sample_loss_time(t0, T_planned, cycle.Λ_sol, rng)
        loss = !isnothing(τ_loss)
        t_end = loss ? τ_loss : T_planned

        # Dense evaluation grid within [t₀, t_end]
        t_grid = range(t0, t_end, length=n_pts)
        for t in t_grid
            L_t = cycle.L_sol(t)
            n_t = cycle.n_sol(t)
            W_t = W_weight(L_t, p)

            push!(ts_t, t)
            push!(ts_cycle, c)
            push!(ts_biomass, n_t * W_t / 1000.0)
            push!(ts_value, n_t * f_value(L_t, p))
            push!(ts_feed, cycle.Φ_sol(t))
            push!(ts_premium, cycle.Π_sol(t))
            push!(ts_premium_rate, π_premium(t, cycle.I_sol, p))
            push!(ts_hazard, λ(t, p))
            push!(ts_indemnity, cycle.I_sol(t))
            push!(ts_fallow_days, 0.0)
            push!(ts_phase, "production")
        end

        # Endpoint income and utility
        if loss
            income = Y_L_seasonal(t_end, t0, cycle, p)
        else
            income = Y_H_seasonal(t_end, t0, cycle, p)
        end
        Y = max(income, 1e-10)

        push!(ep_t_end, t_end)
        push!(ep_cycle, c)
        push!(ep_loss, loss)
        push!(ep_income, income)
        push!(ep_utility, u(Y, p))
        push!(ep_d_star, d_star)

        t_current = t_end
    end

    timeseries = (
        t = ts_t, cycle = ts_cycle,
        biomass_kg = ts_biomass, stock_value = ts_value,
        feed_cost = ts_feed, premium_cost = ts_premium,
        premium_rate = ts_premium_rate, hazard_rate = ts_hazard,
        indemnity = ts_indemnity,
        fallow_days = ts_fallow_days, phase = ts_phase,
    )

    cycle_endpoints = (
        t_end = ep_t_end, cycle = ep_cycle, loss = ep_loss,
        income = ep_income, utility = ep_utility,
        d_star = ep_d_star,
    )

    return (timeseries = timeseries, cycle_endpoints = cycle_endpoints)
end
