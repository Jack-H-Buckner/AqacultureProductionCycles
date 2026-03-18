library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
ts <- read.csv("results/simulations/production_cycle_comparison_ts.csv")
ep <- read.csv("results/simulations/production_cycle_comparison_ep.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Order scenarios for faceting
ts$scenario <- factor(ts$scenario,
                      levels = c("Baseline", "Medium risk", "High risk"))
ep$scenario <- factor(ep$scenario,
                      levels = c("Baseline", "Medium risk", "High risk"))

# Endpoint markers
ep$event <- ifelse(ep$loss, "Loss", "Harvest")

# Production-only data (exclude fallow points for biological variables)
ts_prod <- ts[ts$phase == "production", ]

# Fallow-only data
ts_fallow <- ts[ts$phase == "fallow", ]

# ── Common theme ──────────────────────────────────────────────────────────────
theme_panel <- function() {
  theme_classic(base_size = 16) +
    theme(
      axis.text.x   = element_blank(),
      axis.ticks.x  = element_blank(),
      strip.text     = element_text(face = "bold", size = 14),
      strip.background = element_blank()
    )
}

theme_bottom <- function() {
  theme_classic(base_size = 16) +
    theme(
      strip.text     = element_text(face = "bold", size = 14),
      strip.background = element_blank()
    )
}

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1: Biomass
# ══════════════════════════════════════════════════════════════════════════════
p1 <- ggplot(ts_prod, aes(x = t, y = biomass_kg, group = cycle)) +
  geom_line(color = "steelblue", linewidth = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x") +
  labs(x = NULL, y = "Biomass (kg)") +
  theme_panel()

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2: Stock value
# ══════════════════════════════════════════════════════════════════════════════
p2 <- ggplot(ts_prod, aes(x = t, y = stock_value, group = cycle)) +
  geom_line(color = "darkgreen", linewidth = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x") +
  labs(x = NULL, y = "Stock value") +
  theme_panel() +
  theme(strip.text = element_blank())

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3: Utility at cycle end
# ══════════════════════════════════════════════════════════════════════════════
p3 <- ggplot(ep, aes(x = t_end, y = utility, shape = event, color = event)) +
  geom_point(size = 3, stroke = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x") +
  scale_shape_manual(values = c("Harvest" = 16, "Loss" = 4)) +
  scale_color_manual(values = c("Harvest" = "darkgreen", "Loss" = "firebrick")) +
  labs(x = NULL, y = "Utility u(Y)", shape = NULL, color = NULL) +
  theme_panel() +
  theme(strip.text = element_blank(),
        legend.position = "bottom")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 4: Fallow days (resetting each cycle)
# ══════════════════════════════════════════════════════════════════════════════
p4 <- ggplot(ts_fallow, aes(x = t, y = fallow_days, group = cycle)) +
  geom_line(color = "grey30", linewidth = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x", drop = FALSE) +
  labs(x = NULL, y = "Fallow days") +
  theme_panel() +
  theme(strip.text = element_blank())

# ══════════════════════════════════════════════════════════════════════════════
# Panel 5: Feed cost
# ══════════════════════════════════════════════════════════════════════════════
p5 <- ggplot(ts_prod, aes(x = t, y = feed_cost, group = cycle)) +
  geom_line(color = "darkorange", linewidth = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x") +
  labs(x = NULL, y = "Feed cost \u03a6(t)") +
  theme_panel() +
  theme(strip.text = element_blank())

# ══════════════════════════════════════════════════════════════════════════════
# Panel 6: Cumulative insurance premium
# ══════════════════════════════════════════════════════════════════════════════
p6 <- ggplot(ts_prod, aes(x = t, y = premium_cost, group = cycle)) +
  geom_line(color = "purple4", linewidth = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x") +
  labs(x = NULL, y = "Premium cost \u03a0(t)") +
  theme_panel() +
  theme(strip.text = element_blank())

# ══════════════════════════════════════════════════════════════════════════════
# Panel 7: Instantaneous premium rate
# ══════════════════════════════════════════════════════════════════════════════
p7 <- ggplot(ts_prod, aes(x = t, y = premium_rate, group = cycle)) +
  geom_line(color = "purple4", linewidth = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x") +
  labs(x = NULL, y = "Premium rate \u03c0(t)") +
  theme_panel() +
  theme(strip.text = element_blank())

# ══════════════════════════════════════════════════════════════════════════════
# Panel 8: Hazard rate (continuous across fallow + production)
# ══════════════════════════════════════════════════════════════════════════════
p8 <- ggplot(ts, aes(x = t, y = hazard_rate, group = cycle)) +
  geom_line(color = "firebrick", linewidth = 1.0) +
  facet_wrap(~ scenario, nrow = 1, scales = "free_x") +
  labs(x = "Time (days)", y = "Hazard rate \u03bb(t)") +
  theme_bottom() +
  theme(strip.text = element_blank())

# ══════════════════════════════════════════════════════════════════════════════
# Combine
# ══════════════════════════════════════════════════════════════════════════════

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  combined <- p1 / p2 / p3 / p4 / p5 / p6 / p7 / p8 +
    plot_layout(heights = c(1, 1, 1, 0.7, 1, 1, 1, 1))
  ggsave(file.path(outdir, "production_cycle_comparison.png"), combined,
         width = 18, height = 32, dpi = 400)
} else if (requireNamespace("cowplot", quietly = TRUE)) {
  library(cowplot)
  combined <- plot_grid(p1, p2, p3, p4, p5, p6, p7, p8,
                        ncol = 1, align = "v",
                        rel_heights = c(1, 1, 1, 0.7, 1, 1, 1, 1))
  ggsave(file.path(outdir, "production_cycle_comparison.png"), combined,
         width = 18, height = 32, dpi = 400)
} else {
  cat("patchwork/cowplot not available — install one for combined plot\n")
}

cat("Saved production_cycle_comparison.png\n")
