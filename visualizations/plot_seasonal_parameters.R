library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
grid <- read.csv("results/simulations/seasonal_parameters.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# Three-panel figure: k(t), m(t), λ(t)
# ══════════════════════════════════════════════════════════════════════════════

grid_long <- data.frame(
  t = rep(grid$t, 3),
  value = c(grid$k, grid$m, grid$lambda),
  constant = c(grid$k_const, grid$m_const, grid$lambda_const),
  panel = factor(rep(
    c("k(t): Growth rate (day\u207b\u00b9)",
      "m(t): Mortality rate (day\u207b\u00b9)",
      "\u03bb(t): Hazard rate (day\u207b\u00b9)"),
    each = nrow(grid)),
    levels = c("k(t): Growth rate (day\u207b\u00b9)",
               "m(t): Mortality rate (day\u207b\u00b9)",
               "\u03bb(t): Hazard rate (day\u207b\u00b9)"))
)

p <- ggplot(grid_long, aes(x = t)) +
  geom_line(aes(y = value, color = "Seasonal"),
            linewidth = 1.5) +
  geom_line(aes(y = constant, color = "Constant (mean)"),
            linewidth = 1.5, linetype = "dashed") +
  facet_wrap(~ panel, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c(
    "Seasonal" = "steelblue",
    "Constant (mean)" = "firebrick")) +
  labs(x = "Calendar date (day of year)",
       y = NULL,
       color = NULL,
       title = "Seasonal parameter functions") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "seasonal_parameters.png"), p,
       width = 10, height = 12, dpi = 400)
cat("Saved seasonal_parameters.png\n")
