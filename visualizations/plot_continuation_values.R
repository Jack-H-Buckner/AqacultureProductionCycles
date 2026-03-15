library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
Vtilde_comp <- read.csv("results/simulations/Vtilde_update_comparison.csv")
Vtilde_nodes <- read.csv("results/simulations/Vtilde_update_nodes.csv")
V_comp <- read.csv("results/simulations/V_update_comparison.csv")
V_nodes <- read.csv("results/simulations/V_update_nodes.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Vtilde(t0) — Fourier vs fine grid
# ══════════════════════════════════════════════════════════════════════════════

p1 <- ggplot() +
  geom_line(data = Vtilde_comp,
            aes(x = t0, y = Vtilde_exact / 1e6, color = "Fine grid (n=100)"),
            linewidth = 1.5) +
  geom_line(data = Vtilde_comp,
            aes(x = t0, y = Vtilde_fourier / 1e6, color = "Fourier (N=40)"),
            linewidth = 1.5, linetype = "dashed") +
  geom_point(data = Vtilde_nodes,
             aes(x = t0_node, y = Vtilde_at_node / 1e6, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Fine grid (n=100)" = "steelblue",
    "Fourier (N=40)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = expression(paste("Stocking date ", t[0], " (day of year)")),
       y = expression(paste(tilde(V)(t[0]), " (millions)")),
       color = NULL,
       title = expression(paste("Cycle value ", tilde(V)(t[0]),
                                " = J(T*, ", t[0], ", ", t[0], ")"))) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "Vtilde_fourier_comparison.png"), p1,
       width = 10, height = 7, dpi = 400)
cat("Saved results/figures/Vtilde_fourier_comparison.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Vtilde(t0) — Fourier approximation error
# ══════════════════════════════════════════════════════════════════════════════

Vtilde_comp$error <- Vtilde_comp$Vtilde_fourier - Vtilde_comp$Vtilde_exact
Vtilde_comp$rel_error <- Vtilde_comp$error / Vtilde_comp$Vtilde_exact * 100

p2 <- ggplot(Vtilde_comp, aes(x = t0, y = rel_error)) +
  geom_line(linewidth = 1.5, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = expression(paste("Stocking date ", t[0], " (day of year)")),
       y = "Relative error (%)",
       title = expression(paste("Fourier interpolation error in ",
                                tilde(V)(t[0])))) +
  theme_classic(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "Vtilde_fourier_error.png"), p2,
       width = 10, height = 5, dpi = 400)
cat("Saved results/figures/Vtilde_fourier_error.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: V(t) — Fourier vs fine grid
# ══════════════════════════════════════════════════════════════════════════════

p3 <- ggplot() +
  geom_line(data = V_comp,
            aes(x = t, y = V_exact / 1e6, color = "Fine grid (n=100)"),
            linewidth = 1.5) +
  geom_line(data = V_comp,
            aes(x = t, y = V_fourier / 1e6, color = "Fourier (N=40)"),
            linewidth = 1.5, linetype = "dashed") +
  geom_point(data = V_nodes,
             aes(x = node, y = V_at_node / 1e6, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Fine grid (n=100)" = "steelblue",
    "Fourier (N=40)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "End-of-cycle date t (day of year)",
       y = "V(t) (millions)",
       color = NULL,
       title = "Continuation value V(t): Fourier approximation vs fine grid") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "V_fourier_comparison.png"), p3,
       width = 10, height = 7, dpi = 400)
cat("Saved results/figures/V_fourier_comparison.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: V(t) — Fourier approximation error
# ══════════════════════════════════════════════════════════════════════════════

V_comp$error <- V_comp$V_fourier - V_comp$V_exact
V_comp$rel_error <- V_comp$error / V_comp$V_exact * 100

p4 <- ggplot(V_comp, aes(x = t, y = rel_error)) +
  geom_line(linewidth = 1.5, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = "End-of-cycle date t (day of year)",
       y = "Relative error (%)",
       title = "Fourier interpolation error in V(t)") +
  theme_classic(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "V_fourier_error.png"), p4,
       width = 10, height = 5, dpi = 400)
cat("Saved results/figures/V_fourier_error.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Fallow duration d*(t) and stocking date t0*(t) at nodes
# ══════════════════════════════════════════════════════════════════════════════

# Use the fine-grid d* from V_comp (interpolated) plus nodal values
p5 <- ggplot() +
  geom_line(data = V_comp,
            aes(x = t, y = d_star, color = "Interpolated"),
            linewidth = 1.5) +
  geom_point(data = V_nodes,
             aes(x = node, y = d_at_node, color = "Nodal values"),
             size = 3) +
  scale_color_manual(values = c(
    "Interpolated" = "steelblue",
    "Nodal values" = "firebrick")) +
  labs(x = "End-of-cycle date t (day of year)",
       y = "Fallow duration d* (days)",
       color = NULL,
       title = "Optimal fallow duration d*(t)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fallow_duration_nodes.png"), p5,
       width = 10, height = 7, dpi = 400)
cat("Saved results/figures/fallow_duration_nodes.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Both V(t) and Vtilde(t0) on the same panel
# ══════════════════════════════════════════════════════════════════════════════

# Build a long-format data frame for both functions
V_long <- data.frame(
  t = V_comp$t,
  value = V_comp$V_exact / 1e6,
  type = "fine grid",
  fn = "V(t)"
)
V_long_fourier <- data.frame(
  t = V_comp$t,
  value = V_comp$V_fourier / 1e6,
  type = "Fourier",
  fn = "V(t)"
)
# Use facets with Unicode labels
both_fine <- data.frame(
  t = c(Vtilde_comp$t0, V_comp$t),
  value = c(Vtilde_comp$Vtilde_exact / 1e6, V_comp$V_exact / 1e6),
  fourier = c(Vtilde_comp$Vtilde_fourier / 1e6, V_comp$V_fourier / 1e6),
  panel = c(rep("Cycle value V\u0303(t\u2080)", nrow(Vtilde_comp)),
            rep("Continuation value V(t)", nrow(V_comp)))
)

# Node data for both panels
both_nodes <- data.frame(
  t = c(Vtilde_nodes$t0_node, V_nodes$node),
  value = c(Vtilde_nodes$Vtilde_at_node / 1e6, V_nodes$V_at_node / 1e6),
  panel = c(rep("Cycle value V\u0303(t\u2080)", nrow(Vtilde_nodes)),
            rep("Continuation value V(t)", nrow(V_nodes)))
)

p6 <- ggplot() +
  geom_line(data = both_fine,
            aes(x = t, y = value, color = "Fine grid"),
            linewidth = 1.5) +
  geom_line(data = both_fine,
            aes(x = t, y = fourier, color = "Fourier"),
            linewidth = 1.5, linetype = "dashed") +
  geom_point(data = both_nodes,
             aes(x = t, y = value, color = "Nodes"),
             size = 2.5) +
  facet_wrap(~ panel, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c(
    "Fine grid" = "steelblue",
    "Fourier" = "darkorange",
    "Nodes" = "firebrick")) +
  labs(x = "Calendar date (day of year)",
       y = "Value (millions)",
       color = NULL) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 18),
    strip.background = element_blank()
  )

ggsave(file.path(outdir, "continuation_values_panel.png"), p6,
       width = 10, height = 12, dpi = 400)
cat("Saved results/figures/continuation_values_panel.png\n")
