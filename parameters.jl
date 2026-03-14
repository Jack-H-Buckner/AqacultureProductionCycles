"""
    parameters.jl

Central parameter definitions for the aquaculture bioeconomic model.
All simulation and optimization scripts should import from this file.
"""

# Number of harmonics for seasonal Fourier series
const N_HARMONICS = 2

# ── Seasonal Fourier coefficients ─────────────────────────────────────────────
# Each seasonal function is evaluated as  exp(a0 + Σ aₖ sin + bₖ cos)
# via positive_periodic() in src/00_model_functions.jl.
# Coefficient vectors have length N_HARMONICS.

# Catastrophic hazard rate λ(t)
const λ_coeffs = (
    a0 = -3.0,                              # baseline log-hazard
    a  = [0.3, 0.0],                         # sine coefficients
    b  = [0.2, 0.0],                         # cosine coefficients
)

# Natural (background) mortality rate m(t) on fish numbers
const m_coeffs = (
    a0 = -4.0,
    a  = [0.2, 0.0],
    b  = [0.1, 0.0],
)

# Von Bertalanffy growth rate k(t)
const k_coeffs = (
    a0 = -2.5,
    a  = [0.4, 0.0],
    b  = [0.3, 0.0],
)

# ── Growth parameters ─────────────────────────────────────────────────────────
const L∞  = 60.0       # asymptotic mean length (cm)
const L₀  = 5.0        # initial fingerling length (cm) at stocking
const n₀  = 10000.0    # initial number of fish stocked

# ── Allometric value function  f(L) = α · L^β ────────────────────────────────
const α = 0.01          # allometric scalar
const β = 3.0           # allometric exponent (≈ cubic for weight-length)

# ── Cost structure ────────────────────────────────────────────────────────────
const c_s = 5000.0      # stocking cost (paid at t₀)
const c_h = 2000.0      # harvest cost (paid at T)
const c₂  = 3000.0      # clearing cost (paid at loss event)
const η   = 0.02        # feed cost rate (fraction of stock value per day)

# ── Insurance ─────────────────────────────────────────────────────────────────
const c_I   = 50.0      # administrative cost component in premium
const Q     = 0.10      # insurer profit margin
const Y_MIN = 0.0       # minimum-income margin (0 = break-even coverage)

# ── Discounting ───────────────────────────────────────────────────────────────
const δ   = 0.05 / 365  # daily discount rate
const δ_b = 0.07 / 365  # daily borrowing rate

# ── Risk aversion (CRRA) ─────────────────────────────────────────────────────
const γ = 0.5            # coefficient of relative risk aversion

# ── Random seed ───────────────────────────────────────────────────────────────
const SEED = 5491

# ══════════════════════════════════════════════════════════════════════════════
# Collect everything into a single NamedTuple for passing to model functions
# ══════════════════════════════════════════════════════════════════════════════

const default_params = (
    # seasonal Fourier coefficients
    λ_coeffs = λ_coeffs,
    m_coeffs = m_coeffs,
    k_coeffs = k_coeffs,
    # growth
    L∞  = L∞,
    L₀  = L₀,
    n₀  = n₀,
    # allometric value
    α = α,
    β = β,
    # costs
    c_s = c_s,
    c_h = c_h,
    c₂  = c₂,
    η   = η,
    # insurance
    c_I   = c_I,
    Q     = Q,
    Y_MIN = Y_MIN,
    # discounting
    δ   = δ,
    δ_b = δ_b,
    # utility
    γ = γ,
)
