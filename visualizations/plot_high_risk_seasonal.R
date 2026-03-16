library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
params <- read.csv("results/simulations/high_risk_seasonal_params.csv")
grid   <- read.csv("results/simulations/high_risk_seasonal_grid.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ── Fallow labels ────────────────────────────────────────────────────────────
grid$fallow_label <- factor(
  ifelse(grid$fallow == "optimal_fallow", "Optimal fallow", "No fallow (d* = 0)"),
  levels = c("Optimal fallow", "No fallow (d* = 0)")
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Seasonal parameters — k(t), m(t), λ(t) with baseline overlay
# ══════════════════════════════════════════════════════════════════════════════

params_long <- data.frame(
  t = rep(params$t, 6),
  value = c(params$k, params$m, params$lambda,
            params$k_baseline, params$m_baseline, params$lambda_baseline),
  scenario = rep(c(rep("High risk", nrow(params) * 3),
                   rep("Baseline", nrow(params) * 3))),
  panel = factor(rep(
    rep(c("k(t): Growth rate (day\u207b\u00b9)",
          "m(t): Mortality rate (day\u207b\u00b9)",
          "\u03bb(t): Catastrophic hazard (day\u207b\u00b9)"),
        each = nrow(params)), 2),
    levels = c("k(t): Growth rate (day\u207b\u00b9)",
               "m(t): Mortality rate (day\u207b\u00b9)",
               "\u03bb(t): Catastrophic hazard (day\u207b\u00b9)"))
)

p1 <- ggplot(params_long, aes(x = t, y = value,
                               color = scenario, linetype = scenario)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ panel, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c(
    "High risk" = "firebrick",
    "Baseline" = "steelblue")) +
  scale_linetype_manual(values = c(
    "High risk" = "solid",
    "Baseline" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = NULL,
       color = NULL, linetype = NULL,
       title = "Seasonal parameter functions: high risk vs baseline") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "high_risk_seasonal_params.png"), p1,
       width = 12, height = 14, dpi = 400)
cat("Saved high_risk_seasonal_params.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Continuation value V(t) — optimal fallow vs no fallow
# ══════════════════════════════════════════════════════════════════════════════

p2 <- ggplot(grid, aes(x = t, y = V, color = fallow_label,
                        linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = c(
    "Optimal fallow" = "steelblue",
    "No fallow (d* = 0)" = "firebrick")) +
  scale_linetype_manual(values = c(
    "Optimal fallow" = "solid",
    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = "V(t)",
       color = NULL, linetype = NULL,
       title = "Continuation value: high-risk seasonal scenario") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "high_risk_seasonal_V.png"), p2,
       width = 12, height = 8, dpi = 400)
cat("Saved high_risk_seasonal_V.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Optimal policies — τ*(t₀), d*(t), and stocking FOC
# ══════════════════════════════════════════════════════════════════════════════

if (!requireNamespace("gridExtra", quietly = TRUE)) {
  install.packages("gridExtra", repos = "https://cloud.r-project.org")
}
library(gridExtra)

fallow_colors <- c("Optimal fallow" = "steelblue",
                    "No fallow (d* = 0)" = "firebrick")
fallow_lines  <- c("Optimal fallow" = "solid",
                    "No fallow (d* = 0)" = "dashed")

# τ*(t₀) panel
p3_tau <- ggplot(grid, aes(x = t, y = tau_star, color = fallow_label,
                            linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = fallow_colors) +
  scale_linetype_manual(values = fallow_lines) +
  labs(x = NULL,
       y = expression(paste(tau, "*(t"[0], ") (days)")),
       color = NULL, linetype = NULL,
       title = "Optimal policies: high-risk seasonal scenario") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

# d*(t) panel
p3_d <- ggplot(grid, aes(x = t, y = d_star, color = fallow_label,
                          linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = fallow_colors) +
  scale_linetype_manual(values = fallow_lines) +
  labs(x = NULL,
       y = "d*(t) (days)",
       color = NULL, linetype = NULL) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "none",
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

# Stocking FOC panel: Ṽ'(t₀) and δ·Ṽ(t₀) overlaid
foc_long <- data.frame(
  t = rep(grid$t, 2),
  value = c(grid$Vtilde_prime, grid$delta_Vtilde),
  component = factor(rep(
    c("d\u1e7c/dt\u2080", "\u03b4 \u00b7 \u1e7c(t\u2080)"),
    each = nrow(grid)),
    levels = c("d\u1e7c/dt\u2080", "\u03b4 \u00b7 \u1e7c(t\u2080)")),
  fallow = rep(grid$fallow_label, 2)
)
foc_long$group <- interaction(foc_long$component, foc_long$fallow, sep = " \u2014 ")

p3_foc <- ggplot(foc_long, aes(x = t, y = value, color = group, linetype = group)) +
  geom_line(linewidth = 1.5) +
  geom_hline(yintercept = 0, color = "grey50", linewidth = 0.5) +
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
       color = NULL, linetype = NULL) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.direction = "vertical"
  )

p3 <- grid.arrange(p3_tau, p3_d, p3_foc, ncol = 1, heights = c(0.8, 0.8, 1.2))

ggsave(file.path(outdir, "high_risk_seasonal_policies.png"), p3,
       width = 12, height = 16, dpi = 400)
cat("Saved high_risk_seasonal_policies.png\n")
