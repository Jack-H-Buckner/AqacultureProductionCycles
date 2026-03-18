library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
ts <- read.csv("results/simulations/production_cycle_timeseries.csv")
ep <- read.csv("results/simulations/production_cycle_endpoints.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Assign alternating cycle shading
cycle_bounds <- aggregate(t ~ cycle, data = ts, FUN = range)
shade <- data.frame(
  xmin  = sapply(cycle_bounds$t, `[`, 1),
  xmax  = sapply(cycle_bounds$t, `[`, 2),
  cycle = cycle_bounds$cycle
)
shade$fill <- ifelse(shade$cycle %% 2 == 1, "odd", "even")

# Endpoint markers: shape by harvest vs loss
ep$event <- ifelse(ep$loss, "Loss", "Harvest")

# ── Helper: add cycle shading to a ggplot ─────────────────────────────────────
add_cycle_shading <- function(p) {
  p + geom_rect(data = shade,
                aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = fill),
                alpha = 0.08, inherit.aes = FALSE) +
    scale_fill_manual(values = c("odd" = "grey50", "even" = "grey80"), guide = "none")
}

# Common theme for panels without x-axis
theme_panel <- function() {
  theme_classic(base_size = 20) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
}

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1: Biomass n(t)·W(L(t))
# ══════════════════════════════════════════════════════════════════════════════
p1 <- ggplot(ts, aes(x = t, y = biomass_kg, group = cycle)) +
  geom_line(color = "steelblue", linewidth = 1.5)
p1 <- add_cycle_shading(p1) +
  labs(x = NULL, y = "Biomass (kg)") +
  theme_panel()

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2: Stock value n(t)·f(L(t))
# ══════════════════════════════════════════════════════════════════════════════
p2 <- ggplot(ts, aes(x = t, y = stock_value, group = cycle)) +
  geom_line(color = "darkgreen", linewidth = 1.5)
p2 <- add_cycle_shading(p2) +
  labs(x = NULL, y = "Stock value") +
  theme_panel()

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3: Utility at cycle end (point plot)
# ══════════════════════════════════════════════════════════════════════════════
p3 <- ggplot(ep, aes(x = t_end, y = utility, shape = event, color = event)) +
  geom_point(size = 4, stroke = 1.2)
p3 <- add_cycle_shading(p3) +
  scale_shape_manual(values = c("Harvest" = 16, "Loss" = 4)) +
  scale_color_manual(values = c("Harvest" = "darkgreen", "Loss" = "firebrick")) +
  labs(x = NULL, y = "Utility u(Y)", shape = NULL, color = NULL) +
  theme_panel() +
  theme(legend.position.inside = c(0.9, 0.85))

# ══════════════════════════════════════════════════════════════════════════════
# Panel 4: Cumulative fallow days (step function)
# ══════════════════════════════════════════════════════════════════════════════
p4 <- ggplot(ep, aes(x = t_end, y = cumulative_fallow)) +
  geom_step(color = "grey30", linewidth = 1.5, direction = "mid") +
  geom_point(color = "grey30", size = 3)
p4 <- add_cycle_shading(p4) +
  labs(x = NULL, y = "Cumulative\nfallow (days)") +
  theme_panel()

# ══════════════════════════════════════════════════════════════════════════════
# Panel 5: Cumulative feed cost
# ══════════════════════════════════════════════════════════════════════════════
p5 <- ggplot(ts, aes(x = t, y = feed_cost, group = cycle)) +
  geom_line(color = "darkorange", linewidth = 1.5)
p5 <- add_cycle_shading(p5) +
  labs(x = NULL, y = "Feed cost \u03a6(t)") +
  theme_panel()

# ══════════════════════════════════════════════════════════════════════════════
# Panel 6: Cumulative insurance premium
# ══════════════════════════════════════════════════════════════════════════════
p6 <- ggplot(ts, aes(x = t, y = premium_cost, group = cycle)) +
  geom_line(color = "purple4", linewidth = 1.5)
p6 <- add_cycle_shading(p6) +
  labs(x = NULL, y = "Premium cost \u03a0(t)") +
  theme_panel()

# ══════════════════════════════════════════════════════════════════════════════
# Panel 7: Instantaneous insurance premium rate
# ══════════════════════════════════════════════════════════════════════════════
p7 <- ggplot(ts, aes(x = t, y = premium_rate, group = cycle)) +
  geom_line(color = "purple4", linewidth = 1.5)
p7 <- add_cycle_shading(p7) +
  labs(x = NULL, y = "Premium rate \u03c0(t)") +
  theme_panel()

# ══════════════════════════════════════════════════════════════════════════════
# Panel 8: Hazard rate
# ══════════════════════════════════════════════════════════════════════════════
p8 <- ggplot(ts, aes(x = t, y = hazard_rate, group = cycle)) +
  geom_line(color = "firebrick", linewidth = 1.5)
p8 <- add_cycle_shading(p8) +
  labs(x = "Time (days)", y = "Hazard rate \u03bb(t)") +
  theme_classic(base_size = 20)

# ══════════════════════════════════════════════════════════════════════════════
# Combine into single-column multi-panel figure
# ══════════════════════════════════════════════════════════════════════════════

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  combined <- p1 / p2 / p3 / p4 / p5 / p6 / p7 / p8
  ggsave(file.path(outdir, "production_cycle_timeseries.png"), combined,
         width = 14, height = 32, dpi = 400)
} else if (requireNamespace("cowplot", quietly = TRUE)) {
  library(cowplot)
  combined <- plot_grid(p1, p2, p3, p4, p5, p6, p7, p8,
                        ncol = 1, align = "v")
  ggsave(file.path(outdir, "production_cycle_timeseries.png"), combined,
         width = 14, height = 32, dpi = 400)
} else {
  cat("patchwork/cowplot not available — install one for combined plot\n")
}

cat("Saved production_cycle_timeseries.png\n")
