library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
risk_grid <- read.csv("results/simulations/risk_scenarios_grid.csv")
ymin_grid <- read.csv("results/simulations/minimum_income_grid.csv")
scalars   <- read.csv("results/simulations/minimum_income_scalars.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

Y_MIN_5  <- round(scalars$Y_MIN_5pct, 0)
Y_MIN_10 <- round(scalars$Y_MIN_10pct, 0)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Risk scenario comparison (optimal fallow + no fallow overlaid)
# Solid = optimal fallow, dashed = no fallow (d* = 0)
# ══════════════════════════════════════════════════════════════════════════════

scenario_labels <- c(
  "baseline"    = "Baseline",
  "medium_risk" = "Medium risk",
  "high_risk"   = "High risk"
)
risk_grid$scenario_label <- factor(scenario_labels[risk_grid$scenario],
  levels = c("Baseline", "Medium risk", "High risk"))
risk_grid$fallow_label <- factor(
  ifelse(risk_grid$fallow == "optimal_fallow", "Optimal fallow", "No fallow (d* = 0)"),
  levels = c("Optimal fallow", "No fallow (d* = 0)")
)

scenario_colors <- c(
  "Baseline"    = "steelblue",
  "Medium risk" = "darkorange",
  "High risk"   = "firebrick"
)

# ── V(t) by risk ──────────────────────────────────────────────────────────────
p_risk_V <- ggplot(risk_grid, aes(x = t, y = V, color = scenario_label,
                                   linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = scenario_colors) +
  scale_linetype_manual(values = c("Optimal fallow" = "solid",
                                    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Day of year",
       y = "V(t)",
       color = NULL, linetype = NULL,
       title = "Continuation value by risk level") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_risk_V.png"), p_risk_V,
       width = 12, height = 8, dpi = 400)
cat("Saved comparison_risk_V.png\n")

# ── tau*(t0) by risk ──────────────────────────────────────────────────────────
p_risk_tau <- ggplot(risk_grid, aes(x = t, y = tau_star, color = scenario_label,
                                     linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = scenario_colors) +
  scale_linetype_manual(values = c("Optimal fallow" = "solid",
                                    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Day of year",
       y = expression(paste(tau, "*(t"[0], ") (days)")),
       color = NULL, linetype = NULL,
       title = "Optimal rotation length by risk level") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_risk_tau.png"), p_risk_tau,
       width = 12, height = 8, dpi = 400)
cat("Saved comparison_risk_tau.png\n")

# ── d*(t) by risk (optimal fallow only — d*=0 is trivially zero) ─────────────
risk_opt <- risk_grid[risk_grid$fallow == "optimal_fallow", ]

p_risk_d <- ggplot(risk_opt, aes(x = t, y = d_star, color = scenario_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = scenario_colors) +
  labs(x = "Day of year",
       y = "d*(t) (days)",
       color = NULL,
       title = "Optimal fallow duration by risk level") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_risk_d.png"), p_risk_d,
       width = 12, height = 8, dpi = 400)
cat("Saved comparison_risk_d.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Insurance scenario comparison (optimal fallow + no fallow overlaid)
# ══════════════════════════════════════════════════════════════════════════════

ymin_labels <- c(
  "ymin_zero"  = "No guarantee (Y_MIN = 0)",
  "ymin_5pct"  = paste0("5% guarantee (Y_MIN = ", Y_MIN_5, ")"),
  "ymin_10pct" = paste0("10% guarantee (Y_MIN = ", Y_MIN_10, ")")
)
ymin_grid$ymin_label <- factor(ymin_labels[ymin_grid$ymin_case],
  levels = ymin_labels)
ymin_grid$fallow_label <- factor(
  ifelse(ymin_grid$fallow == "optimal_fallow", "Optimal fallow", "No fallow (d* = 0)"),
  levels = c("Optimal fallow", "No fallow (d* = 0)")
)

ymin_colors <- c(
  setNames("steelblue", ymin_labels["ymin_zero"]),
  setNames("darkorange", ymin_labels["ymin_5pct"]),
  setNames("firebrick", ymin_labels["ymin_10pct"])
)

# ── V(t) by insurance ────────────────────────────────────────────────────────
p_ins_V <- ggplot(ymin_grid, aes(x = t, y = V, color = ymin_label,
                                  linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = ymin_colors) +
  scale_linetype_manual(values = c("Optimal fallow" = "solid",
                                    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Day of year",
       y = "V(t)",
       color = NULL, linetype = NULL,
       title = "Continuation value by insurance coverage (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_insurance_V.png"), p_ins_V,
       width = 12, height = 8, dpi = 400)
cat("Saved comparison_insurance_V.png\n")

# ── tau*(t0) by insurance ─────────────────────────────────────────────────────
p_ins_tau <- ggplot(ymin_grid, aes(x = t, y = tau_star, color = ymin_label,
                                    linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = ymin_colors) +
  scale_linetype_manual(values = c("Optimal fallow" = "solid",
                                    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Day of year",
       y = expression(paste(tau, "*(t"[0], ") (days)")),
       color = NULL, linetype = NULL,
       title = "Optimal rotation length by insurance coverage (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_insurance_tau.png"), p_ins_tau,
       width = 12, height = 8, dpi = 400)
cat("Saved comparison_insurance_tau.png\n")

# ── d*(t) by insurance (optimal fallow only — d*=0 is trivially zero) ────────
ymin_opt <- ymin_grid[ymin_grid$fallow == "optimal_fallow", ]

p_ins_d <- ggplot(ymin_opt, aes(x = t, y = d_star, color = ymin_label)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = ymin_colors) +
  labs(x = "Day of year",
       y = "d*(t) (days)",
       color = NULL,
       title = "Optimal fallow duration by insurance coverage (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_insurance_d.png"), p_ins_d,
       width = 12, height = 8, dpi = 400)
cat("Saved comparison_insurance_d.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Combined panel — risk vs insurance side-by-side
# V(t), tau*(t0), d*(t) as rows; risk and insurance as columns
# ══════════════════════════════════════════════════════════════════════════════

# Build a unified data frame with a "dimension" column
risk_unified <- data.frame(
  t            = risk_grid$t,
  V            = risk_grid$V,
  tau_star     = risk_grid$tau_star,
  d_star       = risk_grid$d_star,
  label        = risk_grid$scenario_label,
  fallow_label = risk_grid$fallow_label,
  dimension    = "Risk level"
)

ins_unified <- data.frame(
  t            = ymin_grid$t,
  V            = ymin_grid$V,
  tau_star     = ymin_grid$tau_star,
  d_star       = ymin_grid$d_star,
  label        = ymin_grid$ymin_label,
  fallow_label = ymin_grid$fallow_label,
  dimension    = "Insurance coverage"
)

combined <- rbind(risk_unified, ins_unified)
combined$dimension <- factor(combined$dimension,
  levels = c("Risk level", "Insurance coverage"))

# Use a single color palette mapped by label
all_colors <- c(
  "Baseline"    = "steelblue",
  "Medium risk" = "darkorange",
  "High risk"   = "firebrick",
  setNames("steelblue",  ymin_labels["ymin_zero"]),
  setNames("darkorange", ymin_labels["ymin_5pct"]),
  setNames("firebrick",  ymin_labels["ymin_10pct"])
)

fallow_linetypes <- c("Optimal fallow" = "solid",
                       "No fallow (d* = 0)" = "dashed")

# ── V(t) panel ────────────────────────────────────────────────────────────────
p_combined_V <- ggplot(combined, aes(x = t, y = V, color = label,
                                      linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ dimension, scales = "free_y") +
  scale_color_manual(values = all_colors) +
  scale_linetype_manual(values = fallow_linetypes) +
  labs(x = "Day of year", y = "V(t)", color = NULL, linetype = NULL,
       title = "Continuation value: risk vs insurance") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 13),
    legend.direction = "vertical",
    strip.text = element_text(face = "bold", size = 18),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_combined_V.png"), p_combined_V,
       width = 16, height = 8, dpi = 400)
cat("Saved comparison_combined_V.png\n")

# ── tau*(t0) panel ────────────────────────────────────────────────────────────
p_combined_tau <- ggplot(combined, aes(x = t, y = tau_star, color = label,
                                        linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ dimension, scales = "free_y") +
  scale_color_manual(values = all_colors) +
  scale_linetype_manual(values = fallow_linetypes) +
  labs(x = "Day of year",
       y = expression(paste(tau, "*(t"[0], ") (days)")),
       color = NULL, linetype = NULL,
       title = "Optimal rotation length: risk vs insurance") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 13),
    legend.direction = "vertical",
    strip.text = element_text(face = "bold", size = 18),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_combined_tau.png"), p_combined_tau,
       width = 16, height = 8, dpi = 400)
cat("Saved comparison_combined_tau.png\n")

# ── d*(t) panel (optimal fallow only — d*=0 is trivially zero) ───────────────
combined_opt <- combined[combined$fallow_label == "Optimal fallow", ]

p_combined_d <- ggplot(combined_opt, aes(x = t, y = d_star, color = label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ dimension, scales = "free_y") +
  scale_color_manual(values = all_colors) +
  labs(x = "Day of year", y = "d*(t) (days)", color = NULL,
       title = "Optimal fallow duration: risk vs insurance") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 13),
    legend.direction = "vertical",
    strip.text = element_text(face = "bold", size = 18),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "comparison_combined_d.png"), p_combined_d,
       width = 16, height = 8, dpi = 400)
cat("Saved comparison_combined_d.png\n")
