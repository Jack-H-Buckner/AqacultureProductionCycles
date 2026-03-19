library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
df <- read.csv("results/simulations/indemnity_xi_comparison.csv")

outdir <- "results/figures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Month labels for faceting
df$month_label <- paste0("t\u2080 = ", (df$t0_month - 1) * 30, " days")
df$month_label <- factor(df$month_label,
                         levels = paste0("t\u2080 = ", seq(0, 330, by = 30), " days"))

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: I(τ) comparison — baseline (ξ=0) vs ξ=0.001 vs ξ=1
# ══════════════════════════════════════════════════════════════════════════════

df_long <- pivot_longer(df,
                        cols = c("I_baseline", "I_xi_small", "I_xi_full"),
                        names_to = "formulation",
                        values_to = "indemnity")

df_long$formulation <- factor(
  ifelse(df_long$formulation == "I_baseline", "Baseline (\u03be = 0)",
  ifelse(df_long$formulation == "I_xi_small", "\u03be = 0.001",
         "\u03be = 1")),
  levels = c("Baseline (\u03be = 0)", "\u03be = 0.001", "\u03be = 1"))

p1 <- ggplot(df_long, aes(x = tau, y = indemnity, color = formulation)) +
  geom_line(linewidth = 1.5) +
  facet_wrap(~ month_label, scales = "free_y", ncol = 4) +
  scale_color_manual(values = c(
    "Baseline (\u03be = 0)" = "steelblue",
    "\u03be = 0.001" = "darkorange",
    "\u03be = 1" = "firebrick")) +
  labs(x = "Cycle age \u03c4 (days since stocking)",
       y = "Indemnity I(\u03c4)",
       color = NULL,
       title = "Indemnity comparison across coverage levels") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 14)
  )

ggsave(file.path(outdir, "indemnity_xi_comparison.png"), p1,
       width = 16, height = 12, dpi = 400)
cat("Saved indemnity_xi_comparison.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Opportunity cost OC(τ) across the cycle
# ══════════════════════════════════════════════════════════════════════════════

p2 <- ggplot(df, aes(x = tau, y = OC)) +
  geom_line(linewidth = 1.5, color = "darkgreen") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  facet_wrap(~ month_label, scales = "free_y", ncol = 4) +
  labs(x = "Cycle age \u03c4 (days since stocking)",
       y = "Opportunity cost OC(\u03c4)",
       title = "Opportunity cost of cycle loss: medium-risk seasonal case") +
  theme_classic(base_size = 20) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(size = 14)
  )

ggsave(file.path(outdir, "opportunity_cost_profile.png"), p2,
       width = 16, height = 12, dpi = 400)
cat("Saved opportunity_cost_profile.png\n")
