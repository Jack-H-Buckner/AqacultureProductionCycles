library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
comparison <- read.csv("results/simulations/harvest_foc_comparison.csv")
nodes      <- read.csv("results/simulations/harvest_foc_nodes.csv")

# ── Figure: Harvest FOC Fourier approximation vs exact ────────────────────────

p1 <- ggplot() +
  geom_line(data = comparison, aes(x = t0, y = tau_exact, color = "Exact (grid)"),
            linewidth = 2) +
  geom_line(data = comparison, aes(x = t0, y = tau_fourier, color = "Fourier (N=40)"),
            linewidth = 2, linetype = "dashed") +
  geom_point(data = nodes, aes(x = t0, y = tau, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Exact (grid)" = "steelblue",
    "Fourier (N=40)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Stocking date t\u2080 (day of year)",
       y = "Optimal cycle duration \u03c4* (days)",
       color = NULL,
       title = "Harvest FOC: Fourier approximation vs exact solution") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/harvest_foc_comparison.png", p1,
       width = 10, height = 7, dpi = 400)

cat("Saved results/figures/harvest_foc_comparison.png\n")

# ── Figure: Interpolation error ──────────────────────────────────────────────

comparison$error <- comparison$tau_fourier - comparison$tau_exact

p2 <- ggplot(comparison, aes(x = t0, y = error)) +
  geom_line(linewidth = 2, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = "Stocking date t\u2080 (day of year)",
       y = "Error (days)",
       title = "Fourier interpolation error in \u03c4*") +
  theme_classic(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/harvest_foc_error.png", p2,
       width = 10, height = 5, dpi = 400)

cat("Saved results/figures/harvest_foc_error.png\n")
