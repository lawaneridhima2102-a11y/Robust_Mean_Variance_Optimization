# Robust Mean-Variance Optimization
# Quantitative Trading with R
# Universität Trier

library(CVXR)
library(quantmod)
library(ggplot2)
library(PerformanceAnalytics)
library(reshape2)

# --------------------------------------------------
# 1. Data
# --------------------------------------------------

tickers <- c("AAPL", "MSFT", "JPM", "BAC", "JNJ",
             "PFE", "XOM", "CVX", "AMZN", "WMT")

getSymbols(tickers,
           from = "2019-01-01",
           to = "2024-01-01",
           src = "yahoo")

prices <- do.call(merge, lapply(tickers, function(x) Ad(get(x))))
colnames(prices) <- tickers

returns <- na.omit(Return.calculate(prices))

mu <- colMeans(returns)
Sigma <- cov(returns)
n <- length(tickers)
volas <- sqrt(diag(Sigma))

# same uncertainty parameter as in the original project
rho <- rep(quantile(abs(mu), 0.20), n)

# strength of the additional risk-uncertainty term
gamma_risk <- 0.02

# Cholesky factor used for portfolio risk
R_chol <- chol(Sigma)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

annual_return <- function(w) {
  as.numeric(t(mu) %*% w) * 252
}

annual_risk <- function(w) {
  as.numeric(sqrt(t(w) %*% Sigma %*% w)) * sqrt(252)
}

robust_risk <- function(w) {
  base_risk <- as.numeric(sqrt(t(w) %*% Sigma %*% w))
  penalty <- sqrt(gamma_risk) * sum(volas * abs(w))

  sqrt(base_risk^2 + penalty^2) * sqrt(252)
}


# --------------------------------------------------
# 2. Unconstrained MVO
# --------------------------------------------------

w1 <- Variable(n)

model1 <- Problem(
  Minimize(quad_form(w1, Sigma)),
  list(sum(w1) == 1)
)

solve(model1)
w1_sol <- value(w1)

cat("\nUNCONSTRAINED MVO\n")
print(data.frame(
  Stock = tickers,
  Weight = round(w1_sol * 100, 2)
))

cat("Return:", round(annual_return(w1_sol) * 100, 2), "%\n")
cat("Risk:", round(annual_risk(w1_sol) * 100, 2), "%\n")


# --------------------------------------------------
# 3. Constrained MVO
# --------------------------------------------------

w2 <- Variable(n)

model2 <- Problem(
  Minimize(quad_form(w2, Sigma)),
  list(
    sum(w2) == 1,
    w2 >= 0,
    w2 <= 0.20
  )
)

solve(model2)
w2_sol <- value(w2)

cat("\nCONSTRAINED MVO\n")
print(data.frame(
  Stock = tickers,
  Weight = round(w2_sol * 100, 2)
))

cat("Return:", round(annual_return(w2_sol) * 100, 2), "%\n")
cat("Risk:", round(annual_risk(w2_sol) * 100, 2), "%\n")


# --------------------------------------------------
# 4. Robust MVO
# --------------------------------------------------

# First check the minimum feasible robust risk
w_min <- Variable(n)

base_min <- p_norm(R_chol %*% w_min, 2)
pen_min <- sqrt(gamma_risk) * (t(volas) %*% abs(w_min))
risk_min <- p_norm(vstack(base_min, pen_min), 2)

model_min <- Problem(
  Minimize(risk_min),
  list(
    sum(w_min) == 1,
    w_min >= 0,
    w_min <= 0.20
  )
)

solve(model_min)
w_min_sol <- value(w_min)

min_robust_risk <- robust_risk(w_min_sol)

# use 20% unless it is infeasible
risk_limit <- 0.20

if (risk_limit <= min_robust_risk) {
  risk_limit <- min_robust_risk + 0.01
}

w3 <- Variable(n)

worst_return <- t(mu) %*% w3 - t(rho) %*% abs(w3)

base_risk <- p_norm(R_chol %*% w3, 2)
risk_penalty <- sqrt(gamma_risk) * (t(volas) %*% abs(w3))
worst_risk <- p_norm(vstack(base_risk, risk_penalty), 2)

model3 <- Problem(
  Maximize(worst_return),
  list(
    sum(w3) == 1,
    w3 >= 0,
    w3 <= 0.20,
    worst_risk <= risk_limit / sqrt(252)
  )
)

solve(model3)
w3_sol <- value(w3)

cat("\nROBUST MVO\n")
print(data.frame(
  Stock = tickers,
  Weight = round(w3_sol * 100, 2)
))

cat("Return:", round(annual_return(w3_sol) * 100, 2), "%\n")
cat("Risk:", round(annual_risk(w3_sol) * 100, 2), "%\n")
cat("Robust risk limit:", round(risk_limit * 100, 2), "%\n")


# --------------------------------------------------
# 5. Comparison table
# --------------------------------------------------

get_stats <- function(w, name) {
  ret <- annual_return(w) * 100
  risk <- annual_risk(w) * 100

  data.frame(
    Portfolio = name,
    Return = round(ret, 2),
    Risk = round(risk, 2),
    Sharpe = round(ret / risk, 3),
    Max_Weight = round(max(abs(w)) * 100, 2),
    Has_Shorts = ifelse(any(w < -0.001), "YES", "NO")
  )
}

comparison <- rbind(
  get_stats(w1_sol, "Unconstrained"),
  get_stats(w2_sol, "Constrained"),
  get_stats(w3_sol, "Robust")
)

cat("\nCOMPARISON TABLE\n")
print(comparison)


# --------------------------------------------------
# 6. Portfolio weights
# --------------------------------------------------

weight_df <- data.frame(
  Stock = tickers,
  Unconstrained = as.numeric(w1_sol) * 100,
  Constrained = as.numeric(w2_sol) * 100,
  Robust = as.numeric(w3_sol) * 100
)

weight_long <- melt(
  weight_df,
  id.vars = "Stock",
  variable.name = "Method",
  value.name = "Weight"
)

p_weights <- ggplot(
  weight_long,
  aes(x = Stock, y = Weight, fill = Method)
) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  geom_hline(yintercept = 20,
             linetype = "dashed",
             linewidth = 0.7) +
  scale_fill_manual(values = c(
    "Unconstrained" = "#E74C3C",
    "Constrained" = "#3498DB",
    "Robust" = "#2ECC71"
  )) +
  labs(
    title = "Portfolio Weights: Three Methods",
    subtitle = "Dashed line = 20% maximum weight for Constrained and Robust MVO",
    x = "Stock",
    y = "Weight (%)",
    fill = "Method"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p_weights)

ggsave("weights_chart_FINAL.png",
       p_weights,
       width = 10,
       height = 6,
       dpi = 300)


# --------------------------------------------------
# 7. Sensitivity analysis
# --------------------------------------------------

# maximum-return robust portfolio without a volatility limit
w_max <- Variable(n)

ret_max <- t(mu) %*% w_max - t(rho) %*% abs(w_max)

model_max <- Problem(
  Maximize(ret_max),
  list(
    sum(w_max) == 1,
    w_max >= 0,
    w_max <= 0.20
  )
)

solve(model_max)
w_max_sol <- value(w_max)

max_robust_risk <- robust_risk(w_max_sol)

risk_low <- min_robust_risk + 0.001
risk_high <- max_robust_risk

if (risk_high <= risk_low) {
  risk_high <- risk_low + 0.05
}

risk_limits <- seq(risk_low, risk_high, length.out = 6)

sens_return <- c()
sens_risk <- c()
sens_robust_risk <- c()
sens_sharpe <- c()
sens_limit <- c()

for (limit in risk_limits) {

  ws <- Variable(n)

  ret_wc <- t(mu) %*% ws - t(rho) %*% abs(ws)

  base <- p_norm(R_chol %*% ws, 2)
  pen <- sqrt(gamma_risk) * (t(volas) %*% abs(ws))
  risk_wc <- p_norm(vstack(base, pen), 2)

  model_s <- Problem(
    Maximize(ret_wc),
    list(
      sum(ws) == 1,
      ws >= 0,
      ws <= 0.20,
      risk_wc <= limit / sqrt(252)
    )
  )

  solve(model_s)
  w_s <- value(ws)

  if (!is.null(w_s) && !any(is.na(w_s))) {
    sens_limit <- c(sens_limit, limit * 100)
    sens_return <- c(sens_return, annual_return(w_s) * 100)
    sens_risk <- c(sens_risk, annual_risk(w_s) * 100)
    sens_robust_risk <- c(sens_robust_risk, robust_risk(w_s) * 100)
    sens_sharpe <- c(
      sens_sharpe,
      (annual_return(w_s) / annual_risk(w_s))
    )
  }
}

sens_df <- data.frame(
  Volatility_Limit = round(sens_limit, 2),
  Actual_Risk = round(sens_risk, 2),
  Robust_Risk = round(sens_robust_risk, 2),
  Return = round(sens_return, 2),
  Sharpe = round(sens_sharpe, 3)
)

cat("\nSENSITIVITY TABLE\n")
print(sens_df)

p_sens <- ggplot(
  sens_df,
  aes(x = Volatility_Limit, y = Return)
) +
  geom_line(color = "#E74C3C", linewidth = 1.5) +
  geom_point(color = "#E74C3C", size = 4) +
  geom_label(
    aes(label = paste0(Return, "%")),
    vjust = -0.9,
    size = 3.5
  ) +
  labs(
    title = "Sensitivity Analysis: Changing Volatility Constraint",
    subtitle = "Maximum robust volatility is varied; other constraints remain fixed",
    x = "Maximum Robust Volatility (%)",
    y = "Portfolio Return (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold")
  )

print(p_sens)

ggsave("sensitivity_chart_FINAL.png",
       p_sens,
       width = 10,
       height = 6,
       dpi = 300)


# --------------------------------------------------
# 8. Efficient frontier
# --------------------------------------------------

min_constrained_risk <- annual_risk(w2_sol)

frontier_low <- max(min_constrained_risk, min_robust_risk) + 0.001
frontier_high <- max_robust_risk

if (frontier_high <= frontier_low) {
  frontier_high <- frontier_low + 0.05
}

frontier_limits <- seq(frontier_low,
                       frontier_high,
                       length.out = 20)

con_ret <- c()
con_risk <- c()

rob_ret <- c()
rob_risk <- c()

for (limit in frontier_limits) {

  # Constrained frontier
  wc <- Variable(n)

  risk_c <- p_norm(R_chol %*% wc, 2)

  model_c <- Problem(
    Maximize(t(mu) %*% wc),
    list(
      sum(wc) == 1,
      wc >= 0,
      wc <= 0.20,
      risk_c <= limit / sqrt(252)
    )
  )

  solve(model_c)
  w_c <- value(wc)

  if (!is.null(w_c) && !any(is.na(w_c))) {
    con_ret <- c(con_ret, annual_return(w_c) * 100)
    con_risk <- c(con_risk, annual_risk(w_c) * 100)
  }

  # Robust frontier
  wr <- Variable(n)

  ret_r <- t(mu) %*% wr - t(rho) %*% abs(wr)

  base_r <- p_norm(R_chol %*% wr, 2)
  pen_r <- sqrt(gamma_risk) * (t(volas) %*% abs(wr))
  risk_r <- p_norm(vstack(base_r, pen_r), 2)

  model_r <- Problem(
    Maximize(ret_r),
    list(
      sum(wr) == 1,
      wr >= 0,
      wr <= 0.20,
      risk_r <= limit / sqrt(252)
    )
  )

  solve(model_r)
  w_r <- value(wr)

  if (!is.null(w_r) && !any(is.na(w_r))) {
    rob_ret <- c(rob_ret, annual_return(w_r) * 100)
    rob_risk <- c(rob_risk, annual_risk(w_r) * 100)
  }
}

front_con <- data.frame(
  Return = con_ret,
  Risk = con_risk,
  Method = "Constrained MVO"
)

front_rob <- data.frame(
  Return = rob_ret,
  Risk = rob_risk,
  Method = "Robust MVO"
)

frontier_df <- rbind(front_con, front_rob)

stock_points <- data.frame(
  Stock = tickers,
  Return = mu * 252 * 100,
  Risk = volas * sqrt(252) * 100
)

p_frontier <- ggplot() +
  geom_line(
    data = frontier_df,
    aes(x = Risk, y = Return, color = Method),
    linewidth = 1.5
  ) +
  geom_point(
    data = stock_points,
    aes(x = Risk, y = Return),
    color = "gray30",
    size = 3
  ) +
  geom_text(
    data = stock_points,
    aes(x = Risk, y = Return, label = Stock),
    vjust = -0.8,
    size = 3.2
  ) +
  scale_color_manual(values = c(
    "Constrained MVO" = "#3498DB",
    "Robust MVO" = "#E74C3C"
  )) +
  labs(
    title = "Efficient Frontier: Constrained vs Robust MVO",
    subtitle = "Same risk-limit range for both approaches",
    x = "Risk - Annualized Volatility (%)",
    y = "Return - Annualized (%)",
    color = "Method"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "top"
  )

print(p_frontier)

ggsave("frontier_chart_FINAL.png",
       p_frontier,
       width = 10,
       height = 6,
       dpi = 300)

cat("\nProject complete.\n")
