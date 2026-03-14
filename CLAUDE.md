# Aquaculture Bioeconomic Model

Numerical optimization of aquaculture production cycles under seasonal mortality risk, with risk aversion and insurance. See `README.md` for full model specification and equations.

## Project Structure

- `src/` — Model functions and numerical optimization routines (Julia)
- `simulations/` — Simulation analysis scripts (Julia)
- `visualizations/` — Plotting scripts (R with ggplot2)
- `tests/` — Unit and integration tests
- `results/figures/` — Saved figure outputs (PNG, 400 dpi)
- `results/models/` — Serialized optimized model objects
- `results/simulations/` — Simulation output (CSV)
- `parameters.jl` — Central parameter definitions shared across all runs
- `run.sh` — Runs the full analysis pipeline end-to-end

## Language and Tooling

- **Optimization and simulation**: Julia. All model code, solvers, and simulation scripts must be Julia.
- **Visualization**: R using `ggplot2` and `tidyr`. No Julia or Python plotting.
- **Data exchange**: Export all model solutions and simulation results as `.csv` files.
- **Figures**: Save as PNG with `dpi = 400`. Use `ggsave(..., dpi = 400)` in R scripts.

## Code Conventions

- Store all model parameters in `parameters.jl`. Never hard-code parameter values in simulation or optimization scripts — always import from `parameters.jl`.
- Use `seed = 5491` for all random number generation in simulations.
- Name bash runner scripts `run_{analysis}.sh` (e.g., `run_baseline.sh`, `run_sensitivity.sh`). Each script should contain the full sequence of bash commands to reproduce that analysis.
- When creating new source files in `src/`, include a module docstring explaining what functions the file provides and how they connect to the model equations in `README.md`.

## Key Model Components

The solver finds periodic functions V(t), T*(t₀), and t₀*(T) over a 365-day cycle by iterating between:

1. Solving the harvest and stocking FOCs given current continuation values
2. Updating continuation values Ṽ and V from the objective function and value linkage

All periodic unknowns are represented as truncated Fourier series. The insurance indemnity I(t) is solved via a first-order linear ODE within each cycle evaluation. See `README.md` § "Numerical Procedure" for the full algorithm.
