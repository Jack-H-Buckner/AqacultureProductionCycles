library(ggplot2)

# ── Read data ─────────────────────────────────────────────────────────────────
df <- read.csv("results/simulations/simulation_validation.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# Simulated E[U] vs analytical continuation value
# ══════════════════════════════════════════════════════════════════════════════

p <- ggplot(df) +
  geom_hline(aes(yintercept = Vtilde_analytical,
                 color = "Analytical V\u0303"),
             linewidth = 1.5) +
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
    "Analytical V\u0303" = "steelblue",
    "Simulated E[U] \u00b1 2SE" = "firebrick")) +
  labs(x = "Starting date (day of year)",
       y = "Expected present utility",
       color = NULL,
       title = "Simulation validation: homogeneous case",
       subtitle = paste0(df$sim_mean[1], " does not appear — see CSV for values")) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_blank()
  )

ggsave(file.path(outdir, "simulation_validation.png"), p,
       width = 10, height = 7, dpi = 400)
cat("Saved simulation_validation.png\n")
