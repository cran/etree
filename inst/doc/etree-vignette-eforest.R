## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(etree)

## -----------------------------------------------------------------------------
# Set seed 
set.seed(123)

# Number of observation
n_obs <- 100

# Response variable for regression
resp_reg <- c(rnorm(n_obs / 4, mean = 0, sd = 1),
              rnorm(n_obs / 4, mean = 2, sd = 1),
              rnorm(n_obs / 4, mean = 4, sd = 1),
              rnorm(n_obs / 4, mean = 6, sd = 1))

## -----------------------------------------------------------------------------
# Response variable for classification
resp_cls <- factor(c(sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                            prob = c(0.85, 0.05, 0.05, 0.05)),
                     sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                            prob = c(0.05, 0.85, 0.05, 0.05)),
                     sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                            prob = c(0.05, 0.05, 0.85, 0.05)),
                     sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                            prob = c(0.05, 0.05, 0.05, 0.85))))

## -----------------------------------------------------------------------------
# Numeric
x1 <- c(runif(n_obs / 4, min = 0, max = 0.55),
        runif(n_obs / 4 * 3, min = 0.45, max = 1))

# Nominal
x2 <- factor(c(rbinom(n_obs / 4, 1, 0.1),
               rbinom(n_obs / 4, 1, 0.9),
               rbinom(n_obs / 2, 1, 0.1)))

# Functions
x3 <- c(fda.usc::rproc2fdata(n_obs / 2, seq(0, 1, len = 100), sigma = 1),
        fda.usc::rproc2fdata(n_obs / 4, seq(0, 1, len = 100), sigma = 1,
                             mu = rep(1, 100)),
        fda.usc::rproc2fdata(n_obs / 4, seq(0, 1, len = 100), sigma = 1))

# Graphs
x4 <- c(lapply(1:(n_obs / 4 * 3), function(j) igraph::sample_gnp(100, 0.1)),
        lapply(1:(n_obs / 4), function(j) igraph::sample_gnp(100, 0.9)))

# Covariates list
cov_list <- list(X1 = x1, X2 = x2, X3 = x3, X4 = x4)


## -----------------------------------------------------------------------------
# Quick fit
ef_fit <- eforest(response = resp_reg,
                  covariates = cov_list,
                  ntrees = 12,
                  perf_metric = 'MSE')

## ---- fig.dim = c(25, 11)-----------------------------------------------------
# Plot of a tree from the ensemble
plot(ef_fit$ensemble[[4]])

## -----------------------------------------------------------------------------
# Retrieve performance metric
ef_fit$perf_metric

# OOB MSE for this ensemble
ef_fit$oob_score

## -----------------------------------------------------------------------------
# Predictions from the fitted object
pred <- predict(ef_fit)
print(pred)

## -----------------------------------------------------------------------------
# New set of covariates
n_obs <- 40
x1n <- c(runif(n_obs / 4, min = 0, max = 0.55),
         runif(n_obs / 4 * 3, min = 0.45, max = 1))
x2n <- factor(c(rbinom(n_obs / 4, 1, 0.1),
                rbinom(n_obs / 4, 1, 0.9),
                rbinom(n_obs / 2, 1, 0.1)))
x3n <- c(fda.usc::rproc2fdata(n_obs / 2, seq(0, 1, len = 100), sigma = 1),
         fda.usc::rproc2fdata(n_obs / 4, seq(0, 1, len = 100), sigma = 1,
                              mu = rep(1, 100)),
         fda.usc::rproc2fdata(n_obs / 4, seq(0, 1, len = 100), sigma = 1))
x4n <- c(lapply(1:(n_obs / 4 * 3), function(j) igraph::sample_gnp(100, 0.1)),
         lapply(1:(n_obs / 4), function(j) igraph::sample_gnp(100, 0.9)))
new_cov_list <- list(X1 = x1n, X2 = x2n, X3 = x3n, X4 = x4n)

# New response 
new_resp_reg <- c(rnorm(n_obs / 4, mean = 0, sd = 1),
                  rnorm(n_obs / 4, mean = 2, sd = 1),
                  rnorm(n_obs / 4, mean = 4, sd = 1),
                  rnorm(n_obs / 4, mean = 6, sd = 1))

# Predictions
new_pred <- predict(ef_fit,
                    newdata = new_cov_list)
print(new_pred)

## -----------------------------------------------------------------------------
# MSE between the new response and its average
mean((new_resp_reg - mean(new_resp_reg)) ^ 2)

# MSE between the new response and predictions with the new set of covariates
mean((new_resp_reg - new_pred) ^ 2)

## ---- fig.dim = c(7, 6)-------------------------------------------------------
# Quick fit
ef_fit <- eforest(response = resp_cls,
                  covariates = cov_list,
                  ntrees = 12,
                  split_type = 'coeff')

## -----------------------------------------------------------------------------
# Predictions from the fitted object
pred <- predict(ef_fit)
print(pred)

## -----------------------------------------------------------------------------
# New response 
new_resp_cls <- factor(c(sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                                prob = c(0.85, 0.05, 0.05, 0.05)),
                         sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                                prob = c(0.05, 0.85, 0.05, 0.05)),
                         sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                                prob = c(0.05, 0.05, 0.85, 0.05)),
                         sample(x = 1:4, size = n_obs / 4, replace = TRUE, 
                                prob = c(0.05, 0.05, 0.05, 0.85))))

# Predictions
new_pred <- predict(ef_fit,
                    newdata = new_cov_list)
print(new_pred)

# Confusion matrix between the new response and predictions from the fitted tree
table(new_pred, new_resp_cls, dnn = c('Predicted', 'True'))

# Misclassification error for predictions on the new set of covariates
sum(new_pred != new_resp_cls) / length(new_resp_cls)

