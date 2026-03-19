library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
df <- read.csv("results/simulations/profit_coverage_comparison.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Color palette for three ξ levels
col_xi0   <- "steelblue"
col_small <- "darkorange"
col_mid   <- "firebrick"

lab_xi0   <- "Breakeven (\u03be = 0)"
lab_small <- "\u03be = 0.001"
lab_mid   <- "\u03be = 0.25"
lvls      <- c(lab_xi0, lab_small, lab_mid)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Continuation value V(t)
# ══════════════════════════════════════════════════════════════════════════════

df_V <- pivot_longer(df,
                     cols = c("V_xi0", "V_xi_small", "V_xi_mid"),
                     names_to = "coverage",
                     values_to = "V")

df_V$coverage <- factor(
  ifelse(df_V$coverage == "V_xi0", lab_xi0,
  ifelse(df_V$coverage == "V_xi_small", lab_small, lab_mid)),
  levels = lvls)

p1 <- ggplot(df_V, aes(x = t, y = V, color = coverage)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = setNames(c(col_xi0, col_small, col_mid), lvls)) +
  labs(x = "Calendar day t",
       y = "Continuation value V(t)",
       color = NULL,
       title = "Continuation value across coverage levels") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "profit_coverage_V_comparison.png"), p1,
       width = 12, height = 7, dpi = 400)
cat("Saved profit_coverage_V_comparison.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Optimal rotation τ*(t₀)
# ══════════════════════════════════════════════════════════════════════════════

df_tau <- pivot_longer(df,
                       cols = c("tau_star_xi0", "tau_star_xi_small", "tau_star_xi_mid"),
                       names_to = "coverage",
                       values_to = "tau_star")

df_tau$coverage <- factor(
  ifelse(df_tau$coverage == "tau_star_xi0", lab_xi0,
  ifelse(df_tau$coverage == "tau_star_xi_small", lab_small, lab_mid)),
  levels = lvls)

p2 <- ggplot(df_tau, aes(x = t, y = tau_star, color = coverage)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = setNames(c(col_xi0, col_small, col_mid), lvls)) +
  labs(x = "Calendar day t\u2080 (stocking date)",
       y = "Optimal rotation \u03c4*(t\u2080) (days)",
       color = NULL,
       title = "Optimal rotation length across coverage levels") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "profit_coverage_tau_comparison.png"), p2,
       width = 12, height = 7, dpi = 400)
cat("Saved profit_coverage_tau_comparison.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Fallow duration d*(t)
# ══════════════════════════════════════════════════════════════════════════════

df_d <- pivot_longer(df,
                     cols = c("d_star_xi0", "d_star_xi_small", "d_star_xi_mid"),
                     names_to = "coverage",
                     values_to = "d_star")

df_d$coverage <- factor(
  ifelse(df_d$coverage == "d_star_xi0", lab_xi0,
  ifelse(df_d$coverage == "d_star_xi_small", lab_small, lab_mid)),
  levels = lvls)

p3 <- ggplot(df_d, aes(x = t, y = d_star, color = coverage)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = setNames(c(col_xi0, col_small, col_mid), lvls)) +
  labs(x = "Calendar day t (end of previous cycle)",
       y = "Fallow duration d*(t) (days)",
       color = NULL,
       title = "Optimal fallow across coverage levels") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

ggsave(file.path(outdir, "profit_coverage_d_comparison.png"), p3,
       width = 12, height = 7, dpi = 400)
cat("Saved profit_coverage_d_comparison.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Combined panel — V(t), τ*(t₀), d*(t), and W(t)
# ══════════════════════════════════════════════════════════════════════════════

df_panel <- data.frame(
  t = rep(df$t, 4),
  variable = rep(c("V(t)", "\u03c4*(t\u2080)", "d*(t)", "W(t)"),
                 each = nrow(df)),
  xi0   = c(df$V_xi0,         df$tau_star_xi0,       df$d_star_xi0,       df$W),
  small = c(df$V_xi_small,    df$tau_star_xi_small,  df$d_star_xi_small,  df$W),
  mid   = c(df$V_xi_mid,      df$tau_star_xi_mid,    df$d_star_xi_mid,    df$W)
)

df_panel$variable <- factor(df_panel$variable,
                            levels = c("V(t)", "\u03c4*(t\u2080)", "d*(t)", "W(t)"))

df_panel_long <- pivot_longer(df_panel,
                              cols = c("xi0", "small", "mid"),
                              names_to = "coverage",
                              values_to = "value")

df_panel_long$coverage <- factor(
  ifelse(df_panel_long$coverage == "xi0", lab_xi0,
  ifelse(df_panel_long$coverage == "small", lab_small, lab_mid)),
  levels = lvls)

p4 <- ggplot(df_panel_long, aes(x = t, y = value, color = coverage)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ variable, scales = "free_y", ncol = 2) +
  scale_color_manual(values = setNames(c(col_xi0, col_small, col_mid), lvls)) +
  labs(x = "Calendar day",
       y = NULL,
       color = NULL,
       title = "Policy comparison across coverage levels (baseline risk)") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 16)
  )

ggsave(file.path(outdir, "profit_coverage_panel_comparison.png"), p4,
       width = 14, height = 10, dpi = 400)
cat("Saved profit_coverage_panel_comparison.png\n")
