library(ggplot2)

# ── Read data ─────────────────────────────────────────────────────────────────
comparison <- read.csv("results/simulations/fallow_duration_comparison.csv")
nodes      <- read.csv("results/simulations/fallow_duration_nodes.csv")

# ── Figure 1: Fallow duration d*(T) comparison ───────────────────────────────

p1 <- ggplot() +
  geom_line(data = comparison, aes(x = T_harvest, y = d_exact, color = "Exact (grid)"),
            linewidth = 2) +
  geom_line(data = comparison, aes(x = T_harvest, y = d_fourier, color = "Fourier (N=40)"),
            linewidth = 2, linetype = "dashed") +
  geom_point(data = nodes, aes(x = T_harvest, y = d_star, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Exact (grid)" = "steelblue",
    "Fourier (N=40)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Harvest date T (day of year)",
       y = "Optimal fallow duration d* (days)",
       color = NULL,
       title = "Stocking FOC: optimal fallow duration",
       subtitle = "d* = 0 is a corner solution (restock immediately)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/fallow_duration_comparison.png", p1,
       width = 10, height = 7, dpi = 400)

cat("Saved results/figures/fallow_duration_comparison.png\n")

# ── Figure 2: Interpolation error ────────────────────────────────────────────

comparison$error <- comparison$d_fourier - comparison$d_exact

p2 <- ggplot(comparison, aes(x = T_harvest, y = error)) +
  geom_line(linewidth = 2, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = "Harvest date T (day of year)",
       y = "Error (days)",
       title = "Fourier interpolation error in fallow duration d*") +
  theme_classic(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/fallow_duration_error.png", p2,
       width = 10, height = 5, dpi = 400)

cat("Saved results/figures/fallow_duration_error.png\n")
