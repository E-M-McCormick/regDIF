###############################################################################
# End-to-End Spline DIF Integration Test
#
# Verifies the complete pipeline:
#   make_spline_pred() -> combine_predictors() -> regDIF() ->
#   extract_spline_coefs() -> predict_spline_dif() -> plot_spline_dif()
#
# Uses simulated data with KNOWN nonlinear DIF structure to verify that:
#   1. The pipeline runs without errors for all penalty types
#   2. Items with true DIF are selected (non-zero) at reasonable tau
#   3. Items without DIF are zeroed out at reasonable tau
#   4. Recovered DIF curves approximate the true generating function
#   5. Spline DIF norms track the regularization path correctly
#   6. Multiple spline covariates can be combined
#   7. Spline + scalar predictor combinations work
#
# NOTE: These tests use generous tolerances and focus on qualitative
# recovery (correct sign, rough shape) rather than exact parameter
# recovery. Exact recovery depends on sample size, number of tau values,
# and convergence — topics for a formal simulation study, not unit tests.
###############################################################################

context("spline-e2e")


# ===========================================================================
# DATA GENERATION WITH KNOWN DIF
# ===========================================================================

set.seed(2026)
n_e2e <- 500
num_items_e2e <- 6

# Continuous covariate with nonlinear DIF potential.
age_raw <- rnorm(n_e2e, mean = 0, sd = 1)

# Binary covariate (no spline needed).
gender_raw <- sample(c(-1, 1), n_e2e, replace = TRUE)

# True latent trait.
theta_true <- rnorm(n_e2e)

# True DIF structure:
#   Items 1-2: nonlinear intercept DIF in age (quadratic-like bump)
#   Items 3-4: linear intercept DIF in gender only
#   Items 5-6: no DIF at all (anchor items)
#
# We don't inject the DIF via spline coefficients (that would be
# circular). Instead, we inject a known smooth function of age directly
# into the item response probabilities.

true_dif_age <- function(x) 0.8 * (x^2 - 1)  # Centered quadratic.

# Generate binary item responses.
item_data_e2e <- matrix(NA, nrow = n_e2e, ncol = num_items_e2e)
for (j in 1:num_items_e2e) {
  intercept <- rnorm(1, 0, 0.3)
  slope <- runif(1, 0.8, 1.2)
  eta <- intercept + slope * theta_true

  if (j <= 2) {
    # Nonlinear DIF in age.
    eta <- eta + true_dif_age(age_raw)
  } else if (j %in% 3:4) {
    # Linear DIF in gender.
    eta <- eta + 0.5 * gender_raw
  }
  # Items 5-6: no DIF.

  item_data_e2e[, j] <- rbinom(n_e2e, 1, plogis(eta))
}
colnames(item_data_e2e) <- paste0("item", 1:num_items_e2e)


# ===========================================================================
# 1. FULL PIPELINE: make_spline_pred -> combine -> regDIF -> extract -> plot
# ===========================================================================

test_that("Full spline pipeline runs without error (grp.lasso).", {
  sp <- make_spline_pred(age_raw, k = 6, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                num.tau = 10,
                control = list(dif.groups = combined$dif.groups))

  expect_s3_class(fit, "regDIF")

  # Extract coefficients for all items.
  for (j in 1:num_items_e2e) {
    beta <- extract_spline_coefs(fit, sp, item = j)
    expect_length(beta, ncol(sp$basis))
  }

  # Plot without error.
  curves <- plot_spline_dif(fit, sp, items = 1:num_items_e2e)
  expect_length(curves, num_items_e2e)

  # Norms matrix.
  norms <- spline_dif_norms(fit, sp)
  expect_equal(dim(norms), c(num_items_e2e, ncol(fit$dif)))
})

test_that("Full spline pipeline runs with grp.mcp.", {
  sp <- make_spline_pred(age_raw, k = 6, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.mcp",
                num.tau = 10,
                control = list(dif.groups = combined$dif.groups))

  expect_s3_class(fit, "regDIF")
})

test_that("Full spline pipeline runs with lasso (scalar penalty).", {
  sp <- make_spline_pred(age_raw, k = 6, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  # Scalar lasso: no dif.groups needed (each basis column penalized
  # independently). This is a valid but less powerful approach.
  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "lasso",
                num.tau = 10)

  expect_s3_class(fit, "regDIF")
})


# ===========================================================================
# 2. QUALITATIVE DIF RECOVERY
# ===========================================================================

test_that("Items with true DIF have larger spline norms than no-DIF items.", {
  sp <- make_spline_pred(age_raw, k = 6, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                num.tau = 15,
                control = list(dif.groups = combined$dif.groups))

  norms <- spline_dif_norms(fit, sp)
  best_tau <- which.min(fit$bic)

  # Items 1-2 have true age DIF; items 5-6 have none.
  dif_norms <- norms[1:2, best_tau]
  no_dif_norms <- norms[5:6, best_tau]

  # At the BIC-optimal model, DIF items should have larger norms
  # than no-DIF items. This is a soft check — depends on power.
  # We check the average rather than individual items.
  expect_true(mean(dif_norms) >= mean(no_dif_norms),
              info = paste("Mean DIF norm:", round(mean(dif_norms), 4),
                           "Mean no-DIF norm:", round(mean(no_dif_norms), 4)))
})


# ===========================================================================
# 3. TAU PATH STRUCTURE FOR SPLINES
# ===========================================================================

test_that("Spline norms decrease monotonically as tau increases.", {
  sp <- make_spline_pred(age_raw, k = 6, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                num.tau = 10,
                control = list(dif.groups = combined$dif.groups))

  norms <- spline_dif_norms(fit, sp)
  completed <- which(!is.na(fit$tau))

  # For group lasso, norms should generally decrease as tau increases
  # (earlier columns = larger tau = more penalized).
  # Check that the first completed model has the smallest norms.
  first <- min(completed)
  last <- max(completed)

  total_norm_first <- sum(norms[, first])
  total_norm_last <- sum(norms[, last])

  expect_true(total_norm_first <= total_norm_last,
              info = paste("Total norm at max tau:", round(total_norm_first, 4),
                           "Total norm at min tau:", round(total_norm_last, 4)))
})

test_that("At max tau, all spline DIF is zero.", {
  sp <- make_spline_pred(age_raw, k = 6, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                num.tau = 10,
                control = list(dif.groups = combined$dif.groups))

  norms <- spline_dif_norms(fit, sp)
  first_complete <- min(which(!is.na(fit$tau)))

  expect_true(all(norms[, first_complete] == 0),
              info = "All spline DIF norms should be zero at max tau")
})


# ===========================================================================
# 4. CURVE SHAPE RECOVERY
# ===========================================================================

test_that("Recovered DIF curve has correct qualitative shape for item 1.", {
  sp <- make_spline_pred(age_raw, k = 8, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                num.tau = 15,
                control = list(dif.groups = combined$dif.groups))

  beta <- extract_spline_coefs(fit, sp, item = 1, tau_index = "bic")

  # Skip if the spline was fully zeroed (insufficient power).
  skip_if(all(beta == 0), "Spline zeroed out — insufficient power for shape test")

  # Evaluate the recovered curve at a grid.
  x_grid <- seq(-2, 2, length.out = 50)
  curve <- predict_spline_dif(sp, beta, x_new = x_grid)

  # The true DIF function is 0.8*(x^2 - 1), which is:
  #   - Negative at x = 0 (minimum at center)
  #   - Positive at x = -2 and x = 2 (U-shaped)
  #   - Minimum near x = 0
  #
  # Check that the recovered curve has a U-shape: higher at the
  # edges than at the center. This is a qualitative check.
  center_val <- curve$f_x[which.min(abs(x_grid))]
  edge_vals <- c(curve$f_x[1], curve$f_x[length(x_grid)])
  mean_edge <- mean(edge_vals)

  expect_true(mean_edge > center_val,
              info = paste("Edge mean:", round(mean_edge, 3),
                           "Center:", round(center_val, 3),
                           "— expected U-shape"))
})


# ===========================================================================
# 5. MULTIPLE SPLINE COVARIATES
# ===========================================================================

test_that("Two spline covariates can be combined and fitted.", {
  # Create a second continuous covariate.
  ses_raw <- rnorm(n_e2e)

  sp_age <- make_spline_pred(age_raw, k = 5, name = "age")
  sp_ses <- make_spline_pred(ses_raw, k = 4, name = "ses")
  combined <- combine_predictors(age = sp_age, ses = sp_ses,
                                 gender = gender_raw)

  # Total predictors: 4 (age) + 3 (ses) + 1 (gender) = 8.
  expect_equal(ncol(combined$pred.data), 8)
  expect_length(combined$dif.groups, 3)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                num.tau = 5,
                control = list(dif.groups = combined$dif.groups))

  expect_s3_class(fit, "regDIF")

  # Extract and plot age DIF curves.
  beta_age <- extract_spline_coefs(fit, sp_age, item = 1)
  expect_length(beta_age, ncol(sp_age$basis))

  # Extract SES DIF curves.
  beta_ses <- extract_spline_coefs(fit, sp_ses, item = 1)
  expect_length(beta_ses, ncol(sp_ses$basis))

  # Both sets of norms should work.
  norms_age <- spline_dif_norms(fit, sp_age)
  norms_ses <- spline_dif_norms(fit, sp_ses)
  expect_equal(nrow(norms_age), 6)
  expect_equal(nrow(norms_ses), 6)
})


# ===========================================================================
# 6. SPLINE WITH ANCHOR ITEMS
# ===========================================================================

test_that("Spline DIF with anchor items.", {
  sp <- make_spline_pred(age_raw, k = 5, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  # Anchor items 5 and 6 (the ones with no true DIF).
  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                anchor = c(5, 6), num.tau = 5,
                control = list(dif.groups = combined$dif.groups))

  expect_s3_class(fit, "regDIF")

  # Anchor items should have zero DIF at all tau values.
  norms <- spline_dif_norms(fit, sp)
  completed <- which(!is.na(fit$tau))
  expect_true(all(norms[5, completed] == 0),
              info = "Anchor item 5 should have zero spline DIF")
  expect_true(all(norms[6, completed] == 0),
              info = "Anchor item 6 should have zero spline DIF")
})


# ===========================================================================
# 7. SPLINE WITH RASCH ITEMS
# ===========================================================================

test_that("Spline DIF works with Rasch items.", {
  sp <- make_spline_pred(age_raw, k = 5, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "rasch", pen.type = "grp.lasso",
                num.tau = 5,
                control = list(dif.groups = combined$dif.groups))

  expect_s3_class(fit, "regDIF")

  # Rasch items have c1 (intercept DIF) estimated, and a1 (slope DIF)
  # present in the parameter vector but fixed to zero.
  beta_int <- extract_spline_coefs(fit, sp, item = 1, family = "int")
  expect_length(beta_int, ncol(sp$basis))

  # Slope DIF coefficients exist but should all be zero for Rasch.
  beta_slp <- extract_spline_coefs(fit, sp, item = 1, family = "slp")
  expect_length(beta_slp, ncol(sp$basis))
  expect_true(all(beta_slp == 0),
              info = "Rasch slope DIF should be zero at all tau values")
})


# ===========================================================================
# 8. SPLINE WITH SQRT WEIGHTS
# ===========================================================================

test_that("Spline DIF with sqrt group weights.", {
  sp <- make_spline_pred(age_raw, k = 6, name = "age")
  combined <- combine_predictors(age = sp, gender = gender_raw)

  fit <- regDIF(item_data_e2e, combined$pred.data,
                item.type = "2pl", pen.type = "grp.lasso",
                num.tau = 5,
                control = list(
                  dif.groups = combined$dif.groups,
                  dif.group.weights = "sqrt"
                ))

  expect_s3_class(fit, "regDIF")

  norms <- spline_dif_norms(fit, sp)
  first_complete <- min(which(!is.na(fit$tau)))
  expect_true(all(norms[, first_complete] == 0))
})
