library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
bench    <- read.csv("results/simulations/fourier_benchmark.csv")
stocking <- read.csv("results/simulations/stocking_foc_comparison.csv")
st_nodes <- read.csv("results/simulations/stocking_foc_nodes.csv")

# ── Figure 1: Accuracy vs performance trade-off ─────────────────────────────

p1 <- ggplot(bench, aes(x = time_s, y = mse)) +
  geom_point(size = 4, color = "steelblue") +
  geom_line(linewidth = 2, color = "steelblue") +
  geom_text(aes(label = paste0("N=", N)), vjust = -1.2, size = 5) +
  labs(x = "Computation time (seconds)",
       y = "Mean squared error (days\u00b2)",
       title = "Fourier approximation: accuracy vs performance") +
  theme_classic(base_size = 20) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/figures/fourier_benchmark.png", p1,
       width = 10, height = 7, dpi = 400)

cat("Saved results/figures/fourier_benchmark.png\n")

# ── Figure 2: Stocking FOC comparison ───────────────────────────────────────

# Panel 1: Ṽ(t₀)
p2a <- ggplot() +
  geom_line(data = stocking, aes(x = t0, y = Vtilde_exact, color = "Exact (grid)"),
            linewidth = 2) +
  geom_line(data = stocking, aes(x = t0, y = Vtilde_fourier, color = "Fourier (N=40)"),
            linewidth = 2, linetype = "dashed") +
  geom_point(data = st_nodes, aes(x = t0, y = Vtilde, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Exact (grid)" = "steelblue",
    "Fourier (N=40)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Stocking date t\u2080 (day of year)",
       y = expression(tilde(V)(t[0])),
       color = NULL,
       title = "Cycle value") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/stocking_Vtilde.png", p2a,
       width = 10, height = 7, dpi = 400)

cat("Saved results/figures/stocking_Vtilde.png\n")

# Panel 2: Stocking FOC residual
p2b <- ggplot() +
  geom_line(data = stocking, aes(x = t0, y = residual_exact, color = "Exact (grid)"),
            linewidth = 2) +
  geom_line(data = stocking, aes(x = t0, y = residual_fourier, color = "Fourier (N=40)"),
            linewidth = 2, linetype = "dashed") +
  geom_point(data = st_nodes, aes(x = t0, y = residual, color = "Fourier nodes"),
             size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c(
    "Exact (grid)" = "steelblue",
    "Fourier (N=40)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Stocking date t\u2080 (day of year)",
       y = expression(tilde(V)*"'(t"[0]*") - "*delta %.% tilde(V)*"(t"[0]*")"),
       color = NULL,
       title = "Stocking FOC residual",
       subtitle = "Negative = immediate restocking optimal (corner solution)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/stocking_foc_residual.png", p2b,
       width = 10, height = 7, dpi = 400)

cat("Saved results/figures/stocking_foc_residual.png\n")
