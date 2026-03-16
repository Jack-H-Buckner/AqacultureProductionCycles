library(ggplot2)
library(tidyr)

if (!requireNamespace("gridExtra", quietly = TRUE)) {
  install.packages("gridExtra", repos = "https://cloud.r-project.org")
}
library(gridExtra)

# ── Read data ─────────────────────────────────────────────────────────────────
params <- read.csv("results/simulations/risk_scenarios_params.csv")
grid   <- read.csv("results/simulations/risk_scenarios_grid.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ── Labels ────────────────────────────────────────────────────────────────────
scenario_labels <- c(
  "baseline"    = "Baseline",
  "medium_risk" = "Medium risk",
  "high_risk"   = "High risk"
)

params$scenario_label <- factor(scenario_labels[params$scenario],
  levels = c("Baseline", "Medium risk", "High risk"))

grid$scenario_label <- factor(scenario_labels[grid$scenario],
  levels = c("Baseline", "Medium risk", "High risk"))

grid$fallow_label <- factor(
  ifelse(grid$fallow == "optimal_fallow", "Optimal fallow", "No fallow (d* = 0)"),
  levels = c("Optimal fallow", "No fallow (d* = 0)")
)

scenario_colors <- c(
  "Baseline"    = "steelblue",
  "Medium risk" = "darkorange",
  "High risk"   = "firebrick"
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Seasonal parameters — k(t), m(t), λ(t) by scenario
# ══════════════════════════════════════════════════════════════════════════════

params_long <- data.frame(
  t = rep(params$t, 3),
  value = c(params$k, params$m, params$lambda),
  scenario = rep(params$scenario_label, 3),
  panel = factor(rep(
    c("k(t): Growth rate (day\u207b\u00b9)",
      "m(t): Mortality rate (day\u207b\u00b9)",
      "\u03bb(t): Catastrophic hazard (day\u207b\u00b9)"),
    each = nrow(params)),
    levels = c("k(t): Growth rate (day\u207b\u00b9)",
               "m(t): Mortality rate (day\u207b\u00b9)",
               "\u03bb(t): Catastrophic hazard (day\u207b\u00b9)"))
)

p1 <- ggplot(params_long, aes(x = t, y = value, color = scenario)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ panel, scales = "free_y", ncol = 1) +
  scale_color_manual(values = scenario_colors) +
  labs(x = "Calendar date (day of year)",
       y = NULL,
       color = NULL,
       title = "Seasonal parameter functions by risk scenario") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "risk_scenarios_params.png"), p1,
       width = 12, height = 14, dpi = 400)
cat("Saved risk_scenarios_params.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Continuation value V(t) — scenarios × fallow conditions
# ══════════════════════════════════════════════════════════════════════════════

grid$group <- interaction(grid$scenario_label, grid$fallow_label, sep = " \u2014 ")

group_colors <- c(
  "Baseline \u2014 Optimal fallow"          = "steelblue",
  "Baseline \u2014 No fallow (d* = 0)"      = "steelblue4",
  "Medium risk \u2014 Optimal fallow"       = "darkorange",
  "Medium risk \u2014 No fallow (d* = 0)"   = "darkorange4",
  "High risk \u2014 Optimal fallow"         = "firebrick",
  "High risk \u2014 No fallow (d* = 0)"     = "firebrick4"
)

group_lines <- c(
  "Baseline \u2014 Optimal fallow"          = "solid",
  "Baseline \u2014 No fallow (d* = 0)"      = "dashed",
  "Medium risk \u2014 Optimal fallow"       = "solid",
  "Medium risk \u2014 No fallow (d* = 0)"   = "dashed",
  "High risk \u2014 Optimal fallow"         = "solid",
  "High risk \u2014 No fallow (d* = 0)"     = "dashed"
)

p2 <- ggplot(grid, aes(x = t, y = V, color = group, linetype = group)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = group_colors) +
  scale_linetype_manual(values = group_lines) +
  labs(x = "Calendar date (day of year)",
       y = "V(t)",
       color = NULL, linetype = NULL,
       title = "Continuation value by risk scenario") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.direction = "vertical",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "risk_scenarios_V.png"), p2,
       width = 12, height = 8, dpi = 400)
cat("Saved risk_scenarios_V.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Optimal policies — τ*(t₀), d*(t), stocking FOC by scenario
#          Rows = scenarios, with fallow conditions overlaid
# ══════════════════════════════════════════════════════════════════════════════

fallow_colors <- c("Optimal fallow" = "steelblue",
                    "No fallow (d* = 0)" = "firebrick")
fallow_lines  <- c("Optimal fallow" = "solid",
                    "No fallow (d* = 0)" = "dashed")

# ── τ*(t₀) ───────────────────────────────────────────────────────────────────
p3_tau <- ggplot(grid, aes(x = t, y = tau_star, color = fallow_label,
                            linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ scenario_label, ncol = 1, scales = "free_y") +
  scale_color_manual(values = fallow_colors) +
  scale_linetype_manual(values = fallow_lines) +
  labs(x = NULL,
       y = expression(paste(tau, "*(t"[0], ") (days)")),
       color = NULL, linetype = NULL,
       title = "Optimal rotation length by risk scenario") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

# ── d*(t) ─────────────────────────────────────────────────────────────────────
p3_d <- ggplot(grid, aes(x = t, y = d_star, color = fallow_label,
                          linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ scenario_label, ncol = 1, scales = "free_y") +
  scale_color_manual(values = fallow_colors) +
  scale_linetype_manual(values = fallow_lines) +
  labs(x = NULL,
       y = "d*(t) (days)",
       color = NULL, linetype = NULL,
       title = "Optimal fallow duration by risk scenario") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

# ── Stocking FOC ──────────────────────────────────────────────────────────────
foc_long <- data.frame(
  t = rep(grid$t, 2),
  value = c(grid$Vtilde_prime, grid$delta_Vtilde),
  component = factor(rep(
    c("d\u1e7c/dt\u2080", "\u03b4 \u00b7 \u1e7c(t\u2080)"),
    each = nrow(grid)),
    levels = c("d\u1e7c/dt\u2080", "\u03b4 \u00b7 \u1e7c(t\u2080)")),
  scenario = rep(grid$scenario_label, 2),
  fallow = rep(grid$fallow_label, 2)
)
foc_long$group <- interaction(foc_long$component, foc_long$fallow, sep = " \u2014 ")

p3_foc <- ggplot(foc_long, aes(x = t, y = value, color = group, linetype = group)) +
  geom_line(linewidth = 1.5) +
  geom_hline(yintercept = 0, color = "grey50", linewidth = 0.5) +
  facet_wrap(~ scenario, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(
    "d\u1e7c/dt\u2080 \u2014 Optimal fallow" = "steelblue",
    "\u03b4 \u00b7 \u1e7c(t\u2080) \u2014 Optimal fallow" = "steelblue4",
    "d\u1e7c/dt\u2080 \u2014 No fallow (d* = 0)" = "firebrick",
    "\u03b4 \u00b7 \u1e7c(t\u2080) \u2014 No fallow (d* = 0)" = "darkorange")) +
  scale_linetype_manual(values = c(
    "d\u1e7c/dt\u2080 \u2014 Optimal fallow" = "solid",
    "\u03b4 \u00b7 \u1e7c(t\u2080) \u2014 Optimal fallow" = "dashed",
    "d\u1e7c/dt\u2080 \u2014 No fallow (d* = 0)" = "solid",
    "\u03b4 \u00b7 \u1e7c(t\u2080) \u2014 No fallow (d* = 0)" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = "Stocking FOC",
       color = NULL, linetype = NULL,
       title = "Stocking first-order condition by risk scenario") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.direction = "vertical",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank()
  )

# Save each as a separate figure (too tall for one combined figure)
ggsave(file.path(outdir, "risk_scenarios_tau.png"), p3_tau,
       width = 12, height = 14, dpi = 400)
cat("Saved risk_scenarios_tau.png\n")

ggsave(file.path(outdir, "risk_scenarios_d.png"), p3_d,
       width = 12, height = 14, dpi = 400)
cat("Saved risk_scenarios_d.png\n")

ggsave(file.path(outdir, "risk_scenarios_foc.png"), p3_foc,
       width = 12, height = 14, dpi = 400)
cat("Saved risk_scenarios_foc.png\n")
