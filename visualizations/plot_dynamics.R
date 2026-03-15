library(ggplot2)
library(tidyr)

# ── Read data ─────────────────────────────────────────────────────────────────
growth <- read.csv("results/simulations/growth_dynamics.csv")
costs  <- read.csv("results/simulations/cost_dynamics.csv")

T_star <- 297.2  # from solver output

# ── Figure 1: Growth dynamics (4-panel) ──────────────────────────────────────

# Reshape for faceted plot
panels <- data.frame(
  day      = rep(growth$day, 4),
  value    = c(growth$survival, growth$weight_g, growth$f_value, growth$v_stock),
  variable = rep(c("Survival probability", "Weight (g)", "Value per fish (f)", "Total stock value (v)"),
                 each = nrow(growth))
)
panels$variable <- factor(panels$variable,
  levels = c("Survival probability", "Weight (g)", "Value per fish (f)", "Total stock value (v)"))

p1 <- ggplot(panels, aes(x = day, y = value)) +
  geom_line(linewidth = 2, color = "steelblue") +
  geom_vline(xintercept = T_star, linetype = "dashed", color = "firebrick", linewidth = 0.5) +
  facet_wrap(~ variable, scales = "free_y", ncol = 2) +
  labs(x = "Days since stocking", y = NULL,
       title = "Homogeneous model: growth and survival dynamics",
       subtitle = paste0("Dashed line = optimal harvest T* = ", round(T_star, 1), " days")) +
  theme_classic(base_size = 20) +
  theme(
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/growth_dynamics.png", p1,
       width = 10, height = 7, dpi = 400)

cat("Saved results/figures/growth_dynamics.png\n")

# ── Figure 2: Accumulated costs ──────────────────────────────────────────────

cost_long <- costs |>
  pivot_longer(cols = c(feed_accumulated, premium_accumulated, stocking_compounded, indemnity),
               names_to = "cost_type", values_to = "value") |>
  transform(cost_type = factor(cost_type,
    levels = c("feed_accumulated", "premium_accumulated", "stocking_compounded", "indemnity"),
    labels = c("Feed (accumulated)", "Insurance premiums (accumulated)",
               "Stocking cost (compounded)", "Indemnity I(t)")))

p2 <- ggplot(cost_long, aes(x = day, y = value, color = cost_type)) +
  geom_line(linewidth = 2) +
  geom_vline(xintercept = T_star, linetype = "dashed", color = "firebrick", linewidth = 0.5) +
  facet_wrap(~ cost_type, ncol = 2) +
  scale_color_manual(values = c("Feed (accumulated)" = "darkorange",
                                "Insurance premiums (accumulated)" = "purple",
                                "Stocking cost (compounded)" = "darkgreen",
                                "Indemnity I(t)" = "steelblue")) +
  labs(x = "Days since stocking", y = NULL,
       title = "Homogeneous model: accumulated costs and indemnity",
       subtitle = paste0("Dashed line = optimal harvest T* = ", round(T_star, 1), " days")) +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/cost_dynamics.png", p2,
       width = 10, height = 7, dpi = 400)

cat("Saved results/figures/cost_dynamics.png\n")

# ── Figure 3: FOC for all four cases ────────────────────────────────────────

foc <- read.csv("results/simulations/foc_all_cases.csv")

# Pivot to long format for color mapping
foc_long <- foc |>
  pivot_longer(cols = c(LHS, RHS), names_to = "side", values_to = "value") |>
  transform(side = factor(side,
    levels = c("LHS", "RHS"),
    labels = c("Marginal benefit (LHS)", "Opportunity cost (RHS)")))

# Facet ordering
foc_long$case <- factor(foc_long$case, levels = c(
  "Case 1: Classical Reed", "Case 2: Risk Aversion",
  "Case 3: Feed Costs", "Case 4: Insurance"))

# T* markers (one per case)
t_stars <- unique(foc[, c("case", "T_star")])
t_stars$case <- factor(t_stars$case, levels = levels(foc_long$case))

p3 <- ggplot(foc_long, aes(x = day, y = value, color = side)) +
  geom_line(linewidth = 2, na.rm = TRUE) +
  geom_vline(data = t_stars, aes(xintercept = T_star),
             linetype = "dashed", color = "firebrick", linewidth = 0.5) +
  facet_wrap(~ case, scales = "free_y", ncol = 2) +
  scale_color_manual(values = c(
    "Marginal benefit (LHS)" = "steelblue",
    "Opportunity cost (RHS)" = "darkorange")) +
  labs(x = "Rotation length T (days)", y = NULL,
       color = NULL,
       title = "First-order conditions: homogeneous model cases") +
  theme_classic(base_size = 20) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold")
  )

ggsave("results/figures/foc_all_cases.png", p3,
       width = 12, height = 9, dpi = 400)

cat("Saved results/figures/foc_all_cases.png\n")
