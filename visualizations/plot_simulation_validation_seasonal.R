library(ggplot2)

# ── Read data ─────────────────────────────────────────────────────────────────
df <- read.csv("results/simulations/simulation_validation_seasonal.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# Simulated E[U] vs analytical V(t₀): medium-risk seasonal case
# ══════════════════════════════════════════════════════════════════════════════

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
  scale_color_manual(values = c(
    "Analytical V(t)" = "steelblue",
    "Simulated E[U] \u00b1 2SE" = "firebrick")) +
  labs(x = "Starting date (day of year)",
       y = "Expected present utility",
       color = NULL,
       title = "Simulation validation: medium-risk seasonal case") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "simulation_validation_seasonal.png"), p,
       width = 10, height = 7, dpi = 400)
cat("Saved simulation_validation_seasonal.png\n")
