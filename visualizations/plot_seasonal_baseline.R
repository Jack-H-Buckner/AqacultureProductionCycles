library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
grid <- read.csv("results/simulations/seasonal_baseline_grid.csv")
nodes <- read.csv("results/simulations/seasonal_baseline_nodes.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Four-panel seasonal solution
# ══════════════════════════════════════════════════════════════════════════════

grid_long <- data.frame(
  t = rep(grid$t, 4),
  value = c(grid$V, grid$Vtilde, grid$tau_star, grid$d_star),
  panel = factor(rep(
    c("V(t): Continuation value",
      "\u1e7c(t\u2080): Cycle value",
      "\u03c4*(t\u2080): Harvest time (days)",
      "d*(t): Fallow duration (days)"),
    each = nrow(grid)),
    levels = c("V(t): Continuation value",
               "\u1e7c(t\u2080): Cycle value",
               "\u03c4*(t\u2080): Harvest time (days)",
               "d*(t): Fallow duration (days)"))
)

node_long <- data.frame(
  t = rep(nodes$node, 4),
  value = c(nodes$V_at_node, nodes$Vtilde_at_node,
            nodes$tau_at_node, nodes$d_at_node),
  panel = factor(rep(
    c("V(t): Continuation value",
      "\u1e7c(t\u2080): Cycle value",
      "\u03c4*(t\u2080): Harvest time (days)",
      "d*(t): Fallow duration (days)"),
    each = nrow(nodes)),
    levels = c("V(t): Continuation value",
               "\u1e7c(t\u2080): Cycle value",
               "\u03c4*(t\u2080): Harvest time (days)",
               "d*(t): Fallow duration (days)"))
)

p1 <- ggplot() +
  geom_line(data = grid_long,
            aes(x = t, y = value, color = "Fourier series"),
            linewidth = 1.5) +
  geom_point(data = node_long,
             aes(x = t, y = value, color = "Fourier nodes"),
             size = 2) +
  facet_wrap(~ panel, scales = "free_y", ncol = 2) +
  scale_color_manual(values = c(
    "Fourier series" = "steelblue",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Calendar date (day of year)",
       y = NULL,
       color = NULL,
       title = "Seasonal baseline solution") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "seasonal_baseline_panel.png"), p1,
       width = 14, height = 10, dpi = 400)
cat("Saved seasonal_baseline_panel.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Harvest time τ*(t₀) close-up
# ══════════════════════════════════════════════════════════════════════════════

p2 <- ggplot() +
  geom_line(data = grid,
            aes(x = t, y = tau_star, color = "Seasonal \u03c4*(t\u2080)"),
            linewidth = 1.5) +
  geom_point(data = nodes,
             aes(x = node, y = tau_at_node, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Seasonal \u03c4*(t\u2080)" = "steelblue",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Stocking date (day of year)",
       y = "Optimal cycle duration (days)",
       color = NULL,
       title = "Seasonal variation in harvest timing") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "seasonal_baseline_tau.png"), p2,
       width = 10, height = 7, dpi = 400)
cat("Saved seasonal_baseline_tau.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: V(t) and self-consistency check
# ══════════════════════════════════════════════════════════════════════════════

p3 <- ggplot() +
  geom_line(data = grid,
            aes(x = t, y = V, color = "V(t) Fourier"),
            linewidth = 1.5) +
  geom_line(data = grid,
            aes(x = t, y = V_recomputed, color = "V recomputed"),
            linewidth = 1.5, linetype = "dashed") +
  geom_point(data = nodes,
             aes(x = node, y = V_at_node, color = "V nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "V(t) Fourier" = "steelblue",
    "V recomputed" = "darkorange",
    "V nodes" = "firebrick")) +
  labs(x = "Calendar date (day of year)",
       y = "Continuation value V",
       color = NULL,
       title = "Continuation value V(t): Fourier vs recomputed") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "seasonal_baseline_V.png"), p3,
       width = 10, height = 7, dpi = 400)
cat("Saved seasonal_baseline_V.png\n")
