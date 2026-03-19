library(ggplot2)

# ── Read data ─────────────────────────────────────────────────────────────────
df <- read.csv("results/simulations/dollar_cv_validation_seasonal.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# Simulated E[Y] vs solver W(t): medium-risk seasonal case
# ══════════════════════════════════════════════════════════════════════════════

p <- ggplot(df) +
  geom_line(aes(x = t_init, y = W_solver,
                color = "Solver W(t)"),
            linewidth = 1.5) +
  geom_point(aes(x = t_init, y = W_solver,
                 color = "Solver W(t)"),
             size = 2.5) +
  geom_errorbar(aes(x = t_init,
                    ymin = sim_lower_2se,
                    ymax = sim_upper_2se,
                    color = "Simulated E[Y] \u00b1 2SE"),
                width = 8, linewidth = 1.0) +
  geom_point(aes(x = t_init,
                 y = sim_mean,
                 color = "Simulated E[Y] \u00b1 2SE"),
             size = 3) +
  scale_color_manual(values = c(
    "Solver W(t)" = "steelblue",
    "Simulated E[Y] \u00b1 2SE" = "firebrick")) +
  labs(x = "Starting date (day of year)",
       y = "Expected present dollar value",
       color = NULL,
       title = "Dollar continuation value validation: medium-risk seasonal case") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "dollar_cv_validation_seasonal.png"), p,
       width = 10, height = 7, dpi = 400)
cat("Saved dollar_cv_validation_seasonal.png\n")
