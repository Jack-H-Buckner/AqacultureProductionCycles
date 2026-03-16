library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
grid <- read.csv("results/simulations/fixed_point_validation_grid.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ── Compute relative errors ──────────────────────────────────────────────────
grid$V_rel_error <- (grid$V_bellman - grid$V_linear) /
                     abs(grid$V_linear) * 100
grid$Vtilde_rel_error <- (grid$Vtilde_bellman - grid$Vtilde_linear) /
                          abs(grid$Vtilde_linear) * 100

# ── Row labels ───────────────────────────────────────────────────────────────
grid$N_label <- factor(
  paste0("N = ", grid$N, "  (", grid$nodes, " nodes)"),
  levels = c("N = 10  (21 nodes)", "N = 20  (41 nodes)",
             "N = 40  (81 nodes)", "N = 60  (121 nodes)")
)

# ── Fallow labels ────────────────────────────────────────────────────────────
grid$fallow_label <- factor(
  ifelse(grid$fallow == "optimal_fallow", "Optimal fallow", "No fallow (d* = 0)"),
  levels = c("Optimal fallow", "No fallow (d* = 0)")
)

# ══════════════════════════════════════════════════════════════════════════════
# Panel figure: V(t) and Ṽ(t₀) — overlaid fallow conditions
# ══════════════════════════════════════════════════════════════════════════════

V_long <- data.frame(
  t        = rep(grid$t, 2),
  value    = c(grid$V_linear, grid$V_bellman),
  method   = rep(c("Direct linear solve", "Full Bellman iteration"),
                 each = nrow(grid)),
  N_label  = rep(grid$N_label, 2),
  fallow   = rep(grid$fallow_label, 2),
  panel    = "V(t): Continuation value"
)

Vt_long <- data.frame(
  t        = rep(grid$t, 2),
  value    = c(grid$Vtilde_linear, grid$Vtilde_bellman),
  method   = rep(c("Direct linear solve", "Full Bellman iteration"),
                 each = nrow(grid)),
  N_label  = rep(grid$N_label, 2),
  fallow   = rep(grid$fallow_label, 2),
  panel    = "\u1e7c(t\u2080): Cycle value"
)

vals_long <- rbind(V_long, Vt_long)
vals_long$panel <- factor(vals_long$panel,
  levels = c("V(t): Continuation value", "\u1e7c(t\u2080): Cycle value"))

# Interaction for color + linetype
vals_long$group <- interaction(vals_long$method, vals_long$fallow, sep = " \u2014 ")

p1 <- ggplot(vals_long, aes(x = t, y = value, color = group, linetype = group)) +
  geom_line(linewidth = 1.5) +
  facet_grid(N_label ~ panel, scales = "free_y") +
  scale_color_manual(values = c(
    "Direct linear solve \u2014 Optimal fallow" = "steelblue",
    "Full Bellman iteration \u2014 Optimal fallow" = "darkorange",
    "Direct linear solve \u2014 No fallow (d* = 0)" = "steelblue4",
    "Full Bellman iteration \u2014 No fallow (d* = 0)" = "firebrick")) +
  scale_linetype_manual(values = c(
    "Direct linear solve \u2014 Optimal fallow" = "solid",
    "Full Bellman iteration \u2014 Optimal fallow" = "dashed",
    "Direct linear solve \u2014 No fallow (d* = 0)" = "solid",
    "Full Bellman iteration \u2014 No fallow (d* = 0)" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = NULL,
       color = NULL, linetype = NULL) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.direction = "vertical",
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank()
  )

ggsave(file.path(outdir, "fixed_point_validation_panel.png"), p1,
       width = 16, height = 18, dpi = 400)
cat("Saved fixed_point_validation_panel.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Error panel: overlaid fallow conditions
# ══════════════════════════════════════════════════════════════════════════════

err_V <- data.frame(
  t = grid$t,
  rel_error = grid$V_rel_error,
  quantity = "V(t)",
  N_label = grid$N_label,
  fallow = grid$fallow_label
)

err_Vt <- data.frame(
  t = grid$t,
  rel_error = grid$Vtilde_rel_error,
  quantity = "\u1e7c(t\u2080)",
  N_label = grid$N_label,
  fallow = grid$fallow_label
)

err_long <- rbind(err_V, err_Vt)
err_long$quantity <- factor(err_long$quantity, levels = c("V(t)", "\u1e7c(t\u2080)"))
err_long$group <- interaction(err_long$quantity, err_long$fallow, sep = " \u2014 ")

p2 <- ggplot(err_long, aes(x = t, y = rel_error, color = group, linetype = group)) +
  geom_line(linewidth = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  facet_wrap(~ N_label, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(
    "V(t) \u2014 Optimal fallow" = "steelblue",
    "\u1e7c(t\u2080) \u2014 Optimal fallow" = "darkorange",
    "V(t) \u2014 No fallow (d* = 0)" = "steelblue4",
    "\u1e7c(t\u2080) \u2014 No fallow (d* = 0)" = "firebrick")) +
  scale_linetype_manual(values = c(
    "V(t) \u2014 Optimal fallow" = "solid",
    "\u1e7c(t\u2080) \u2014 Optimal fallow" = "solid",
    "V(t) \u2014 No fallow (d* = 0)" = "dashed",
    "\u1e7c(t\u2080) \u2014 No fallow (d* = 0)" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = "Relative error (%)",
       color = NULL, linetype = NULL,
       title = "Approximation error: f/g decomposition vs full Bellman") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.direction = "vertical",
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fixed_point_validation_errors.png"), p2,
       width = 12, height = 18, dpi = 400)
cat("Saved fixed_point_validation_errors.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Fallow duration d*(t) panel: rows = N, overlaid fallow conditions
# ══════════════════════════════════════════════════════════════════════════════

p3 <- ggplot(grid, aes(x = t, y = d_star, color = fallow_label, linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ N_label, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(
    "Optimal fallow" = "steelblue",
    "No fallow (d* = 0)" = "firebrick")) +
  scale_linetype_manual(values = c(
    "Optimal fallow" = "solid",
    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = "Fallow duration d*(t) (days)",
       color = NULL, linetype = NULL,
       title = "Optimal stocking delay by number of spline nodes") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fixed_point_validation_fallow.png"), p3,
       width = 12, height = 18, dpi = 400)
cat("Saved fixed_point_validation_fallow.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Harvest time τ*(t₀) panel: rows = N, overlaid fallow conditions
# ══════════════════════════════════════════════════════════════════════════════

p4 <- ggplot(grid, aes(x = t, y = tau_star, color = fallow_label, linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ N_label, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(
    "Optimal fallow" = "steelblue",
    "No fallow (d* = 0)" = "firebrick")) +
  scale_linetype_manual(values = c(
    "Optimal fallow" = "solid",
    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = expression(paste(tau, "*(t"[0], ") (days)")),
       color = NULL, linetype = NULL,
       title = "Optimal cycle duration by number of spline nodes") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fixed_point_validation_tau.png"), p4,
       width = 12, height = 18, dpi = 400)
cat("Saved fixed_point_validation_tau.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# V(t) comparison: optimal fallow vs no fallow (direct solver only)
# ══════════════════════════════════════════════════════════════════════════════

p5 <- ggplot(grid, aes(x = t, y = V_linear, color = fallow_label, linetype = fallow_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ N_label, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(
    "Optimal fallow" = "steelblue",
    "No fallow (d* = 0)" = "firebrick")) +
  scale_linetype_manual(values = c(
    "Optimal fallow" = "solid",
    "No fallow (d* = 0)" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = "V(t)",
       color = NULL, linetype = NULL,
       title = "Continuation value: optimal fallow vs no fallow") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fixed_point_validation_V.png"), p5,
       width = 12, height = 18, dpi = 400)
cat("Saved fixed_point_validation_V.png\n")
