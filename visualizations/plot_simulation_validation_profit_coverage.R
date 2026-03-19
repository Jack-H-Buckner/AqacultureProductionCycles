library(ggplot2)

# ── Read data ─────────────────────────────────────────────────────────────────
df <- read.csv("results/simulations/simulation_validation_profit_coverage.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# Simulated E[U] vs analytical V(t₀): profit-coverage validation
# ══════════════════════════════════════════════════════════════════════════════

# Create xi labels for faceting
df$xi_label <- paste0("\u03be = ", df$xi)
df$xi_label <- factor(df$xi_label,
                      levels = paste0("\u03be = ", sort(unique(df$xi))))

p <- ggplot(df) +
  geom_line(aes(x = t_init, y = V_analytical,
                color = "Analytical V(t)"),
            linewidth = 1.5) +
  geom_point(aes(x = t_init, y = V_analytical,
                 color = "Analytical V(t)"),
             size = 2.5) +
  geom_errorbar(aes(x = t_init,
                    ymin = sim_lower_2se,
                    ymax = sim_upper_2se,
                    color = "Simulated E[U] \u00b1 2SE"),
                width = 8, linewidth = 1.0) +
  geom_point(aes(x = t_init,
                 y = sim_mean,
                 color = "Simulated E[U] \u00b1 2SE"),
             size = 3) +
  facet_wrap(~ xi_label, ncol = 2, scales = "free_y") +
  scale_color_manual(values = c(
    "Analytical V(t)" = "steelblue",
    "Simulated E[U] \u00b1 2SE" = "firebrick")) +
  labs(x = "Starting date (day of year)",
       y = "Expected present utility",
       color = NULL,
       title = "Simulation validation: profit-coverage insurance (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 16)
  )

ggsave(file.path(outdir, "simulation_validation_profit_coverage.png"), p,
       width = 14, height = 7, dpi = 400)
cat("Saved simulation_validation_profit_coverage.png\n")
