library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
grid <- read.csv("results/simulations/homogeneous_validation_grid.csv")
nodes <- read.csv("results/simulations/homogeneous_validation_nodes.csv")
scalars <- read.csv("results/simulations/homogeneous_validation_scalars.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Extract scalar benchmarks
T_hom <- scalars$homogeneous[scalars$quantity == "T_star"]
V_hom <- scalars$homogeneous[scalars$quantity == "V_star"]
Vtilde_hom <- scalars$homogeneous[scalars$quantity == "Vtilde_star"]

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Four-panel comparison
# ══════════════════════════════════════════════════════════════════════════════

# Build long-format data for the four quantities
grid_long <- data.frame(
  t = rep(grid$t, 4),
  seasonal = c(grid$V_seasonal, grid$Vtilde_seasonal,
               grid$tau_seasonal, grid$d_seasonal),
  homogeneous = c(grid$V_homogeneous, grid$Vtilde_homogeneous,
                  grid$tau_homogeneous, grid$d_homogeneous),
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
            aes(x = t, y = homogeneous, color = "Homogeneous (analytical)"),
            linewidth = 1.5) +
  geom_line(data = grid_long,
            aes(x = t, y = seasonal, color = "Seasonal solver"),
            linewidth = 1.5, linetype = "dashed") +
  geom_point(data = node_long,
             aes(x = t, y = value, color = "Fourier nodes"),
             size = 2) +
  facet_wrap(~ panel, scales = "free_y", ncol = 2) +
  scale_color_manual(values = c(
    "Homogeneous (analytical)" = "steelblue",
    "Seasonal solver" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Calendar date (day of year)",
       y = NULL,
       color = NULL) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank()
  )

ggsave(file.path(outdir, "homogeneous_validation_panel.png"), p1,
       width = 14, height = 10, dpi = 400)
cat("Saved homogeneous_validation_panel.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: V(t) close-up with error band
# ══════════════════════════════════════════════════════════════════════════════

grid$V_rel_error <- (grid$V_seasonal - grid$V_homogeneous) /
                     abs(grid$V_homogeneous) * 100

p2 <- ggplot() +
  geom_hline(aes(yintercept = V_hom, color = "Homogeneous V*"),
             linewidth = 1.5) +
  geom_line(data = grid,
            aes(x = t, y = V_seasonal, color = "Seasonal V(t)"),
            linewidth = 1.5, linetype = "dashed") +
  geom_point(data = nodes,
             aes(x = node, y = V_at_node, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Homogeneous V*" = "steelblue",
    "Seasonal V(t)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Calendar date (day of year)",
       y = "Continuation value V",
       color = NULL,
       title = "Continuation value: seasonal solver vs homogeneous benchmark") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "homogeneous_validation_V.png"), p2,
       width = 10, height = 7, dpi = 400)
cat("Saved homogeneous_validation_V.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Harvest time comparison
# ══════════════════════════════════════════════════════════════════════════════

p3 <- ggplot() +
  geom_hline(aes(yintercept = T_hom, color = "Homogeneous T*"),
             linewidth = 1.5) +
  geom_line(data = grid,
            aes(x = t, y = tau_seasonal, color = "Seasonal \u03c4*(t\u2080)"),
            linewidth = 1.5, linetype = "dashed") +
  geom_point(data = nodes,
             aes(x = node, y = tau_at_node, color = "Fourier nodes"),
             size = 3) +
  scale_color_manual(values = c(
    "Homogeneous T*" = "steelblue",
    "Seasonal \u03c4*(t\u2080)" = "darkorange",
    "Fourier nodes" = "firebrick")) +
  labs(x = "Calendar date (day of year)",
       y = "Optimal cycle duration (days)",
       color = NULL,
       title = "Harvest time: seasonal solver vs homogeneous benchmark") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "homogeneous_validation_tau.png"), p3,
       width = 10, height = 7, dpi = 400)
cat("Saved homogeneous_validation_tau.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Relative errors
# ══════════════════════════════════════════════════════════════════════════════

grid$V_rel_err <- (grid$V_seasonal - grid$V_homogeneous) /
                   abs(grid$V_homogeneous) * 100
grid$Vtilde_rel_err <- (grid$Vtilde_seasonal - grid$Vtilde_homogeneous) /
                        abs(grid$Vtilde_homogeneous) * 100
grid$tau_rel_err <- (grid$tau_seasonal - grid$tau_homogeneous) /
                     grid$tau_homogeneous * 100

err_long <- data.frame(
  t = rep(grid$t, 3),
  rel_error = c(grid$V_rel_err, grid$Vtilde_rel_err, grid$tau_rel_err),
  quantity = factor(rep(
    c("V(t)", "\u1e7c(t\u2080)", "\u03c4*(t\u2080)"),
    each = nrow(grid)),
    levels = c("V(t)", "\u1e7c(t\u2080)", "\u03c4*(t\u2080)"))
)

p4 <- ggplot(err_long, aes(x = t, y = rel_error, color = quantity)) +
  geom_line(linewidth = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c(
    "V(t)" = "steelblue",
    "\u1e7c(t\u2080)" = "darkorange",
    "\u03c4*(t\u2080)" = "forestgreen")) +
  labs(x = "Calendar date (day of year)",
       y = "Relative error (%)",
       color = NULL,
       title = "Seasonal solver error vs homogeneous benchmark") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "homogeneous_validation_errors.png"), p4,
       width = 10, height = 6, dpi = 400)
cat("Saved homogeneous_validation_errors.png\n")
