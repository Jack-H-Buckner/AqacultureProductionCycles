library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
df <- read.csv("results/simulations/Q_xi_grid.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ── Label formatting ──────────────────────────────────────────────────────────
df$Q_label <- paste0("Q = ", df$Q)
df$xi_label <- paste0("\u03be = ", df$xi)

# Order factor levels
df$Q_label <- factor(df$Q_label,
                     levels = paste0("Q = ", sort(unique(df$Q))))
df$xi_label <- factor(df$xi_label,
                      levels = paste0("\u03be = ", sort(unique(df$xi))))

# Color palette for ξ levels
xi_cols <- c(
  "\u03be = 0"    = "steelblue",
  "\u03be = 0.25" = "darkorange",
  "\u03be = 0.5"  = "forestgreen",
  "\u03be = 1"    = "firebrick",
  "\u03be = 0.75" = "purple"
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Continuation value V(t), faceted by Q, colored by ξ
# ══════════════════════════════════════════════════════════════════════════════

p1 <- ggplot(df, aes(x = t, y = V, color = xi_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ Q_label, ncol = 2, scales = "free_y") +
  scale_color_manual(values = xi_cols) +
  labs(x = "Calendar day t",
       y = "Continuation value V(t)",
       color = NULL,
       title = "Continuation value across Q and \u03be (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 16)
  )

ggsave(file.path(outdir, "Q_xi_grid_V.png"), p1,
       width = 14, height = 10, dpi = 400)
cat("Saved Q_xi_grid_V.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Optimal rotation τ*(t₀), faceted by Q, colored by ξ
# ══════════════════════════════════════════════════════════════════════════════

p2 <- ggplot(df, aes(x = t, y = tau_star, color = xi_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ Q_label, ncol = 2, scales = "free_y") +
  scale_color_manual(values = xi_cols) +
  labs(x = "Calendar day t\u2080 (stocking date)",
       y = "Optimal rotation \u03c4*(t\u2080) (days)",
       color = NULL,
       title = "Optimal rotation across Q and \u03be (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 16)
  )

ggsave(file.path(outdir, "Q_xi_grid_tau.png"), p2,
       width = 14, height = 10, dpi = 400)
cat("Saved Q_xi_grid_tau.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Fallow duration d*(t), faceted by Q, colored by ξ
# ══════════════════════════════════════════════════════════════════════════════

p3 <- ggplot(df, aes(x = t, y = d_star, color = xi_label)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ Q_label, ncol = 2, scales = "free_y") +
  scale_color_manual(values = xi_cols) +
  labs(x = "Calendar day t",
       y = "Fallow duration d*(t) (days)",
       color = NULL,
       title = "Optimal fallow across Q and \u03be (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 16)
  )

ggsave(file.path(outdir, "Q_xi_grid_d.png"), p3,
       width = 14, height = 10, dpi = 400)
cat("Saved Q_xi_grid_d.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Combined panel — V(t), τ*(t₀), d*(t) faceted by Q
# ══════════════════════════════════════════════════════════════════════════════

df_panel <- pivot_longer(df,
                         cols = c("V", "tau_star", "d_star", "fallow_foc"),
                         names_to = "variable",
                         values_to = "value")

df_panel$variable <- factor(
  ifelse(df_panel$variable == "V", "V(t)",
  ifelse(df_panel$variable == "tau_star", "\u03c4*(t\u2080)",
  ifelse(df_panel$variable == "d_star", "d*(t)",
         "Fallow FOC"))),
  levels = c("V(t)", "\u03c4*(t\u2080)", "d*(t)", "Fallow FOC"))

p4 <- ggplot(df_panel, aes(x = t, y = value, color = xi_label)) +
  geom_line(linewidth = 1.5) +
  facet_grid(variable ~ Q_label, scales = "free_y") +
  scale_color_manual(values = xi_cols) +
  labs(x = "Calendar day",
       y = NULL,
       color = NULL,
       title = "Policy comparison across Q and \u03be (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 14)
  )

ggsave(file.path(outdir, "Q_xi_grid_panel.png"), p4,
       width = 18, height = 14, dpi = 400)
cat("Saved Q_xi_grid_panel.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Fallow FOC residual alone, faceted by Q, colored by ξ
# ══════════════════════════════════════════════════════════════════════════════

p5 <- ggplot(df, aes(x = t, y = fallow_foc, color = xi_label)) +
  geom_line(linewidth = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  facet_wrap(~ Q_label, ncol = 2, scales = "free_y") +
  scale_color_manual(values = xi_cols) +
  labs(x = "Calendar day t",
       y = expression(tilde(V)*"'(t) - "*delta*tilde(V)*"(t)"),
       color = NULL,
       title = "Fallow FOC residual across Q and \u03be (medium risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 16)
  )

ggsave(file.path(outdir, "Q_xi_grid_fallow_foc.png"), p5,
       width = 14, height = 10, dpi = 400)
cat("Saved Q_xi_grid_fallow_foc.png\n")
