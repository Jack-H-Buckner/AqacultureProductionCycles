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
  levels = c("N = 10  (21 nodes)", "N = 20  (41 nodes)", "N = 40  (81 nodes)")
)

# ══════════════════════════════════════════════════════════════════════════════
# Panel figure: columns = V(t), Ṽ(t₀); rows = N
# ══════════════════════════════════════════════════════════════════════════════

# Reshape to long format for V and Vtilde
V_long <- data.frame(
  t        = rep(grid$t, 2),
  value    = c(grid$V_linear, grid$V_bellman),
  method   = rep(c("Direct linear solve", "Full Bellman iteration"),
                 each = nrow(grid)),
  N_label  = rep(grid$N_label, 2),
  panel    = "V(t): Continuation value"
)

Vt_long <- data.frame(
  t        = rep(grid$t, 2),
  value    = c(grid$Vtilde_linear, grid$Vtilde_bellman),
  method   = rep(c("Direct linear solve", "Full Bellman iteration"),
                 each = nrow(grid)),
  N_label  = rep(grid$N_label, 2),
  panel    = "\u1e7c(t\u2080): Cycle value"
)

vals_long <- rbind(V_long, Vt_long)
vals_long$panel <- factor(vals_long$panel,
  levels = c("V(t): Continuation value", "\u1e7c(t\u2080): Cycle value"))

p1 <- ggplot(vals_long, aes(x = t, y = value, color = method, linetype = method)) +
  geom_line(linewidth = 1.5) +
  facet_grid(N_label ~ panel, scales = "free_y") +
  scale_color_manual(values = c(
    "Direct linear solve" = "steelblue",
    "Full Bellman iteration" = "darkorange")) +
  scale_linetype_manual(values = c(
    "Direct linear solve" = "solid",
    "Full Bellman iteration" = "dashed")) +
  labs(x = "Calendar date (day of year)",
       y = NULL,
       color = NULL, linetype = NULL) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank()
  )

ggsave(file.path(outdir, "fixed_point_validation_panel.png"), p1,
       width = 16, height = 14, dpi = 400)
cat("Saved fixed_point_validation_panel.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Error panel: columns = V(t), Ṽ(t₀); rows = N
# ══════════════════════════════════════════════════════════════════════════════

err_long <- data.frame(
  t = rep(grid$t, 2),
  rel_error = c(grid$V_rel_error, grid$Vtilde_rel_error),
  quantity = factor(rep(
    c("V(t)", "\u1e7c(t\u2080)"),
    each = nrow(grid)),
    levels = c("V(t)", "\u1e7c(t\u2080)")),
  N_label = rep(grid$N_label, 2)
)

p2 <- ggplot(err_long, aes(x = t, y = rel_error, color = quantity)) +
  geom_line(linewidth = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  facet_wrap(~ N_label, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(
    "V(t)" = "steelblue",
    "\u1e7c(t\u2080)" = "darkorange")) +
  labs(x = "Calendar date (day of year)",
       y = "Relative error (%)",
       color = NULL,
       title = "Approximation error: f/g decomposition vs full Bellman") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fixed_point_validation_errors.png"), p2,
       width = 12, height = 14, dpi = 400)
cat("Saved fixed_point_validation_errors.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Fallow duration d*(t) panel: rows = N
# ══════════════════════════════════════════════════════════════════════════════

p3 <- ggplot(grid, aes(x = t, y = d_star)) +
  geom_line(linewidth = 1.5, color = "steelblue") +
  facet_wrap(~ N_label, ncol = 1, scales = "free_y") +
  labs(x = "Calendar date (day of year)",
       y = "Fallow duration d*(t) (days)",
       title = "Optimal stocking delay by number of Fourier harmonics") +
  theme_classic(base_size = 20) +
  theme(
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fixed_point_validation_fallow.png"), p3,
       width = 12, height = 14, dpi = 400)
cat("Saved fixed_point_validation_fallow.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Harvest time τ*(t₀) panel: rows = N
# ══════════════════════════════════════════════════════════════════════════════

p4 <- ggplot(grid, aes(x = t, y = tau_star)) +
  geom_line(linewidth = 1.5, color = "steelblue") +
  facet_wrap(~ N_label, ncol = 1, scales = "free_y") +
  labs(x = "Calendar date (day of year)",
       y = expression(paste(tau, "*(t"[0], ") (days)")),
       title = "Optimal cycle duration by number of Fourier harmonics") +
  theme_classic(base_size = 20) +
  theme(
    strip.text = element_text(face = "bold", size = 16),
    strip.background = element_blank(),
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "fixed_point_validation_tau.png"), p4,
       width = 12, height = 14, dpi = 400)
cat("Saved fixed_point_validation_tau.png\n")
