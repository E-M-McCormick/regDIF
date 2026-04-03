###############################################################################
# Unit Tests for Spline Helper Functions (R/spline_helpers.R)
#
# Tests cover:
#   1. make_spline_pred: basis dimensions, sum-to-zero constraint, column
#      naming, group definition, penalty matrix, input validation
#   2. predict_spline_dif: curve reconstruction, known-coefficient recovery,
#      default grid, custom grid, input validation
#   3. combine_predictors: merging spline + scalar, column ordering,
#      group definitions, input validation
###############################################################################

context("spline-helpers")

# Shared test data.
set.seed(456)
n_test <- 100
x_age <- rnorm(n_test, mean = 40, sd = 10)
x_ses <- runif(n_test, 0, 100)
gender <- sample(c(-1, 1), n_test, replace = TRUE)


# ===========================================================================
# 1. make_spline_pred
# ===========================================================================

test_that("make_spline_pred returns correct structure.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")

  expect_true(is.list(sp))
  expect_true(all(c("basis", "dif.groups", "penalty", "smooth_obj") %in%
                    names(sp)))
  expect_true(is.matrix(sp$basis))
  expect_true(is.list(sp$dif.groups))
  expect_false(is.null(sp$smooth_obj))
})

test_that("Basis has k-1 columns after sum-to-zero constraint.", {
  for (k_val in c(4, 6, 10)) {
    sp <- make_spline_pred(x_age, k = k_val, name = "age")
    expect_equal(ncol(sp$basis), k_val - 1,
                 info = paste("k =", k_val, "should produce", k_val - 1,
                              "columns"))
    expect_equal(nrow(sp$basis), n_test)
  }
})

test_that("Basis columns are named correctly.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")
  expect_equal(colnames(sp$basis),
               paste0("age_s", 1:5))

  sp2 <- make_spline_pred(x_ses, k = 4, name = "ses_score")
  expect_equal(colnames(sp2$basis),
               paste0("ses_score_s", 1:3))
})

test_that("Sum-to-zero constraint is satisfied.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")

  # For any coefficient vector beta, sum_i f(x_i) should be ~0.
  # This means colSums(B) %*% beta ≈ 0 for all beta,
  # which requires colSums(B) ≈ 0 (the constraint).
  col_sums <- colSums(sp$basis)

  # With the absorb.cons constraint, colSums should be near zero
  # (not exactly zero due to floating point, but very close).
  expect_true(all(abs(col_sums) < 1e-8),
              info = paste("Column sums should be ~0, got max:",
                           max(abs(col_sums))))
})

test_that("dif.groups has correct structure.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")

  expect_length(sp$dif.groups, 1)
  expect_equal(names(sp$dif.groups), "age")
  expect_equal(sp$dif.groups$age, paste0("age_s", 1:5))
})

test_that("Penalty matrix is returned and correctly sized.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")

  expect_false(is.null(sp$penalty))
  expect_equal(dim(sp$penalty), c(5, 5))
  # Penalty matrix should be symmetric and positive semi-definite.
  expect_equal(sp$penalty, t(sp$penalty), tolerance = 1e-10)
  eigenvals <- eigen(sp$penalty, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(eigenvals >= -1e-10),
              info = "Penalty matrix should be positive semi-definite")
})

test_that("Different basis types work.", {
  for (bs_type in c("cr", "tp", "ps", "bs")) {
    sp <- make_spline_pred(x_age, k = 6, bs = bs_type, name = "age")
    expect_equal(ncol(sp$basis), 5,
                 info = paste("Basis type", bs_type))
    expect_equal(nrow(sp$basis), n_test)
  }
})

test_that("Default name is derived from variable.", {
  my_var <- x_age
  sp <- make_spline_pred(my_var, k = 4)
  # Column names should start with "my_var_s".
  expect_true(all(grepl("^my_var_s", colnames(sp$basis))))
})


# --- Input validation ---

test_that("NA values in x are rejected.", {
  x_with_na <- x_age
  x_with_na[5] <- NA
  expect_error(make_spline_pred(x_with_na, k = 6, name = "age"),
               "NA values")
})

test_that("Non-numeric x is rejected.", {
  expect_error(make_spline_pred(c("a", "b", "c"), k = 3, name = "x"),
               "numeric vector")
})

test_that("Too few unique values for k is rejected.", {
  x_few <- rep(c(1, 2, 3), length.out = 50)
  expect_error(make_spline_pred(x_few, k = 6, name = "x"),
               "unique values")
})

test_that("k < 3 is rejected.", {
  expect_error(make_spline_pred(x_age, k = 2, name = "age"),
               "k must be >= 3")
})

test_that("Invalid basis type is rejected.", {
  expect_error(make_spline_pred(x_age, k = 6, bs = "invalid", name = "age"),
               "Unsupported basis type")
})


# ===========================================================================
# 2. predict_spline_dif
# ===========================================================================

test_that("predict_spline_dif returns correct structure.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")
  coefs <- rep(0.1, 5)
  result <- predict_spline_dif(sp, coefs)

  expect_true(is.data.frame(result))
  expect_equal(names(result), c("x", "f_x"))
  # Default grid: 200 points.
  expect_equal(nrow(result), 200)
})

test_that("Zero coefficients produce zero DIF curve.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")
  coefs <- rep(0, 5)
  result <- predict_spline_dif(sp, coefs)

  expect_true(all(abs(result$f_x) < 1e-10),
              info = "Zero coefficients should produce zero DIF")
})

test_that("Prediction at training points recovers fitted values.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")
  coefs <- c(0.5, -0.3, 0.2, 0.1, -0.4)

  # Evaluate at the original x values.
  result <- predict_spline_dif(sp, coefs, x_new = x_age)

  # Should match B %*% coefs.
  expected <- as.numeric(sp$basis %*% coefs)
  expect_equal(result$f_x, expected, tolerance = 1e-10)
})

test_that("Custom x_new grid works.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")
  coefs <- rep(0.1, 5)
  x_grid <- seq(20, 60, by = 5)
  result <- predict_spline_dif(sp, coefs, x_new = x_grid)

  expect_equal(nrow(result), length(x_grid))
  expect_equal(result$x, x_grid)
})

test_that("Sum-to-zero holds at training points.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")
  coefs <- c(1.0, -0.5, 0.3, -0.2, 0.8)

  result <- predict_spline_dif(sp, coefs, x_new = x_age)
  # The sum of f(x_i) across training points should be ~0.
  expect_true(abs(sum(result$f_x)) < 1e-8,
              info = paste("Sum of DIF at training points should be ~0, got:",
                           sum(result$f_x)))
})


# --- Input validation ---

test_that("Wrong coefficient length is rejected.", {
  sp <- make_spline_pred(x_age, k = 6, name = "age")
  expect_error(predict_spline_dif(sp, rep(0.1, 3)),
               "does not match")
})

test_that("Missing smooth_obj is rejected.", {
  fake_spec <- list(basis = matrix(0, 10, 5), smooth_obj = NULL)
  expect_error(predict_spline_dif(fake_spec, rep(0.1, 5)),
               "smooth_obj")
})


# ===========================================================================
# 3. combine_predictors
# ===========================================================================

test_that("combine_predictors merges spline + scalar correctly.", {
  sp_age <- make_spline_pred(x_age, k = 6, name = "age")
  combined <- combine_predictors(age = sp_age, gender = gender)

  expect_true(is.matrix(combined$pred.data))
  # 5 spline columns + 1 scalar = 6 total.
  expect_equal(ncol(combined$pred.data), 6)
  expect_equal(nrow(combined$pred.data), n_test)

  # Column names.
  expect_equal(colnames(combined$pred.data),
               c(paste0("age_s", 1:5), "gender"))
})

test_that("combine_predictors group definitions are correct.", {
  sp_age <- make_spline_pred(x_age, k = 6, name = "age")
  combined <- combine_predictors(age = sp_age, gender = gender)

  expect_length(combined$dif.groups, 2)
  expect_equal(combined$dif.groups$age, paste0("age_s", 1:5))
  expect_equal(combined$dif.groups$gender, "gender")
})

test_that("Multiple spline specs are combined correctly.", {
  sp_age <- make_spline_pred(x_age, k = 5, name = "age")
  sp_ses <- make_spline_pred(x_ses, k = 4, name = "ses")
  combined <- combine_predictors(age = sp_age, ses = sp_ses,
                                 gender = gender)

  # 4 + 3 + 1 = 8 columns.
  expect_equal(ncol(combined$pred.data), 8)
  expect_length(combined$dif.groups, 3)
  expect_equal(combined$dif.groups$age, paste0("age_s", 1:4))
  expect_equal(combined$dif.groups$ses, paste0("ses_s", 1:3))
  expect_equal(combined$dif.groups$gender, "gender")
})

test_that("Matrix predictors are handled.", {
  sp_age <- make_spline_pred(x_age, k = 4, name = "age")
  custom_mat <- matrix(rnorm(n_test * 2), ncol = 2,
                       dimnames = list(NULL, c("z1", "z2")))
  combined <- combine_predictors(age = sp_age, custom = custom_mat)

  expect_equal(ncol(combined$pred.data), 3 + 2)
  expect_equal(combined$dif.groups$custom, c("z1", "z2"))
})


# --- Input validation ---

test_that("Unnamed arguments are rejected.", {
  sp <- make_spline_pred(x_age, k = 4, name = "age")
  expect_error(combine_predictors(sp, gender),
               "must be named")
})

test_that("Non-numeric, non-spline arguments are rejected.", {
  expect_error(combine_predictors(bad = "not_numeric"),
               "numeric vector")
})


# ===========================================================================
# 4. INTEGRATION: Spline predictors work with parse_dif_groups
# ===========================================================================

# ===========================================================================
# 5. EXTRACTION + PLOTTING (requires a fitted model)
# ===========================================================================

# Fit a small model once for all extraction/plotting tests.
# Use binary items from ida with a spline basis for age.
sp_fit_age <- make_spline_pred(as.numeric(ida[, 7]), k = 5, name = "age")
comb_fit <- combine_predictors(age = sp_fit_age,
                               gender = as.numeric(ida[, 8]))
fit_for_tests <- regDIF(ida[, 1:6], comb_fit$pred.data,
                        item.type = "2pl", pen.type = "grp.lasso",
                        num.tau = 5,
                        control = list(dif.groups = comb_fit$dif.groups))


test_that("extract_spline_coefs returns correct length.", {
  beta <- extract_spline_coefs(fit_for_tests, sp_fit_age, item = 1)
  expect_length(beta, ncol(sp_fit_age$basis))
  expect_equal(names(beta), colnames(sp_fit_age$basis))
})

test_that("extract_spline_coefs works with item name.", {
  beta <- extract_spline_coefs(fit_for_tests, sp_fit_age, item = "item1")
  expect_length(beta, ncol(sp_fit_age$basis))
})

test_that("extract_spline_coefs works with different families.", {
  beta_int <- extract_spline_coefs(fit_for_tests, sp_fit_age,
                                   item = 1, family = "int")
  beta_slp <- extract_spline_coefs(fit_for_tests, sp_fit_age,
                                   item = 1, family = "slp")
  expect_length(beta_int, ncol(sp_fit_age$basis))
  expect_length(beta_slp, ncol(sp_fit_age$basis))
})

test_that("extract_spline_coefs with aic tau selection.", {
  beta <- extract_spline_coefs(fit_for_tests, sp_fit_age,
                               item = 1, tau_index = "aic")
  expect_length(beta, ncol(sp_fit_age$basis))
})

test_that("extract_spline_coefs with numeric tau index.", {
  beta <- extract_spline_coefs(fit_for_tests, sp_fit_age,
                               item = 1, tau_index = 1)
  expect_length(beta, ncol(sp_fit_age$basis))
  # First tau (highest penalty) should be all zeros.
  expect_true(all(beta == 0),
              info = "At max tau, all spline DIF should be zero")
})

test_that("extract_spline_coefs errors for invalid item index.", {
  expect_error(
    extract_spline_coefs(fit_for_tests, sp_fit_age, item = 99),
    "out of range"
  )
})

test_that("Extracted coefficients produce valid DIF curves.", {
  beta <- extract_spline_coefs(fit_for_tests, sp_fit_age, item = 1)
  curve_df <- predict_spline_dif(sp_fit_age, beta)
  expect_true(is.data.frame(curve_df))
  expect_equal(nrow(curve_df), 200)
  expect_false(any(is.nan(curve_df$f_x)))
})

test_that("plot_spline_dif runs without error.", {
  # Just check it doesn't error; don't verify graphical output.
  expect_silent(
    curves <- plot_spline_dif(fit_for_tests, sp_fit_age, items = c(1, 2))
  )
  expect_true(is.list(curves))
  expect_length(curves, 2)
  expect_true(all(c("x", "f_x") %in% names(curves[[1]])))
})

test_that("plot_spline_dif returns curves invisibly.", {
  curves <- plot_spline_dif(fit_for_tests, sp_fit_age,
                            items = 1, legend = FALSE)
  expect_length(curves, 1)
})

test_that("spline_dif_norms returns correct dimensions.", {
  norms <- spline_dif_norms(fit_for_tests, sp_fit_age)
  expect_equal(nrow(norms), 6)  # 6 items.
  expect_equal(ncol(norms), ncol(fit_for_tests$dif))
})

test_that("spline_dif_norms are zero at max tau.", {
  norms <- spline_dif_norms(fit_for_tests, sp_fit_age)
  first_complete <- min(which(!is.na(fit_for_tests$tau)))
  expect_true(all(norms[, first_complete] == 0),
              info = "All DIF norms should be zero at max tau")
})

test_that("spline_dif_norms are non-negative.", {
  norms <- spline_dif_norms(fit_for_tests, sp_fit_age)
  completed <- which(!is.na(fit_for_tests$tau))
  expect_true(all(norms[, completed] >= 0, na.rm = TRUE))
})

test_that("spline_dif_norms for subset of items.", {
  norms <- spline_dif_norms(fit_for_tests, sp_fit_age, items = c(1, 3))
  expect_equal(nrow(norms), 2)
})


# ===========================================================================
# 6. INTEGRATION: Spline predictors work with parse_dif_groups
# ===========================================================================

test_that("Spline dif.groups are parsed correctly by parse_dif_groups.", {
  sp_age <- make_spline_pred(x_age, k = 6, name = "age")
  combined <- combine_predictors(age = sp_age, gender = gender)

  result <- parse_dif_groups(
    dif_groups        = combined$dif.groups,
    dif_group_mode    = "paired",
    dif_group_weights = "sqrt",
    dif_families      = NULL,
    pred_data         = combined$pred.data,
    pen_type          = "grp.lasso",
    num_predictors    = ncol(combined$pred.data),
    item_type         = rep("2pl", 4),
    num_items         = 4,
    num_responses     = rep(2, 4)
  )

  # Two groups: age (5 covariates) and gender (1 covariate).
  expect_length(result$groups_idx, 2)
  expect_equal(length(result$groups_idx[["age"]]), 5)
  expect_equal(length(result$groups_idx[["gender"]]), 1)

  # Weights for 2PL (2 families): age = sqrt(5*2), gender = sqrt(1*2).
  expect_equal(result$item_group_weights[1, 1], sqrt(10))
  expect_equal(result$item_group_weights[1, 2], sqrt(2))
})
