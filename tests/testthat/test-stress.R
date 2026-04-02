###############################################################################
# Phase D: Stress Tests for Group Penalty Generalization
#
# These tests exercise edge cases, mixed configurations, and new group
# penalty features introduced in Phases A-C. They verify that the package
# runs without errors and produces numerically sensible results across
# the full range of supported configurations.
#
# All tests use small N and small num.tau for speed. The goal is not to
# validate statistical properties (that requires simulation studies) but
# to catch runtime errors, NaN propagation, and integration bugs.
###############################################################################

context("stress-tests")


# ===========================================================================
# SYNTHETIC DATA GENERATION
# ===========================================================================

# --- Shared predictors (used by all synthetic datasets) ---
set.seed(123)
n_synth <- 200
pred_synth <- data.frame(
  x1 = rnorm(n_synth),
  x2 = rbinom(n_synth, 1, 0.5) * 2 - 1,   # {-1, 1}
  x3 = rnorm(n_synth)
)

# --- Graded items: 4-category ordinal responses ---
# Simulate from a simple latent variable model, then discretize.
theta_synth <- rnorm(n_synth)
graded_items <- matrix(NA, nrow = n_synth, ncol = 4)
for (j in 1:4) {
  latent <- theta_synth + rnorm(n_synth, sd = 0.8)
  graded_items[, j] <- cut(latent,
                            breaks = c(-Inf, -0.5, 0.3, 1.0, Inf),
                            labels = FALSE)
}
colnames(graded_items) <- paste0("item", 1:4)
graded_data <- as.data.frame(graded_items)

# --- CFA items: continuous Gaussian responses ---
cfa_items <- matrix(NA, nrow = n_synth, ncol = 4)
for (j in 1:4) {
  cfa_items[, j] <- theta_synth * runif(1, 0.5, 1.5) +
    rnorm(1, 0, 1) + rnorm(n_synth, sd = 1)
}
colnames(cfa_items) <- paste0("item", 1:4)
cfa_data <- as.data.frame(cfa_items)

# --- Binary items for Rasch/mixed tests ---
binary_items <- matrix(NA, nrow = n_synth, ncol = 6)
for (j in 1:6) {
  p_resp <- plogis(theta_synth * runif(1, 0.7, 1.3) + rnorm(1, 0, 0.5))
  binary_items[, j] <- rbinom(n_synth, 1, p_resp)
}
colnames(binary_items) <- paste0("item", 1:6)
binary_data <- as.data.frame(binary_items)

# --- Helper: check a regDIF result for numerical sanity ---
check_sanity <- function(res, label) {
  # DIF coefficients should have no NaN/NA for completed models.
  completed <- which(!is.na(res$tau))
  if (length(completed) > 0) {
    expect_false(any(is.nan(res$dif[, completed])),
                 info = paste(label, "- NaN in DIF coefficients"))
  }
  # AIC/BIC should be finite for completed models.
  if (!is.null(res$aic)) {
    aic_vals <- res$aic[completed]
    expect_true(all(is.finite(aic_vals) | is.na(aic_vals)),
                info = paste(label, "- non-finite AIC"))
  }
}


# ===========================================================================
# 1. ITEM TYPE COVERAGE
# ===========================================================================

test_that("Rasch items run with lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "rasch", pen.type = "lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "rasch+lasso")
})

test_that("Rasch items run with grp.lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "rasch", pen.type = "grp.lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "rasch+grp.lasso")
})

test_that("Graded items run with lasso.", {
  res <- regDIF(graded_data, pred_synth,
                item.type = "graded", pen.type = "lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "graded+lasso")
})

test_that("Graded items run with grp.lasso (NEW: group penalty for graded).", {
  res <- regDIF(graded_data, pred_synth,
                item.type = "graded", pen.type = "grp.lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "graded+grp.lasso")
})

test_that("Graded items run with grp.mcp.", {
  res <- regDIF(graded_data, pred_synth,
                item.type = "graded", pen.type = "grp.mcp", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "graded+grp.mcp")
})

test_that("CFA items run with lasso.", {
  res <- regDIF(cfa_data, pred_synth,
                item.type = "cfa", pen.type = "lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "cfa+lasso")
})

test_that("CFA items run with grp.lasso.", {
  res <- regDIF(cfa_data, pred_synth,
                item.type = "cfa", pen.type = "grp.lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "cfa+grp.lasso")
})

test_that("Mixed item types (2pl + rasch) run with lasso.", {
  # Items 1-3 as 2PL, items 4-6 as Rasch.
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = c("2pl", "2pl", "2pl", "rasch", "rasch", "rasch"),
                pen.type = "lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "mixed+lasso")
})

test_that("Mixed item types (2pl + rasch) run with grp.lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = c("2pl", "2pl", "2pl", "rasch", "rasch", "rasch"),
                pen.type = "grp.lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "mixed+grp.lasso")
})


# ===========================================================================
# 2. ANCHOR ITEMS
# ===========================================================================

test_that("Single anchor item with lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "lasso",
                anchor = 1, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "anchor=1+lasso")
})

test_that("Multiple anchors with grp.lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "grp.lasso",
                anchor = c(1, 3), num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "anchor=1,3+grp.lasso")
})

test_that("tau = 0 with anchor (unpenalized model).", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "lasso",
                tau = 0, anchor = 1)
  expect_s3_class(res, "regDIF")
  # At tau = 0, DIF should generally be non-zero.
  expect_true(any(res$dif[, 1] != 0),
              info = "Expected non-zero DIF at tau = 0")
})


# ===========================================================================
# 3. PEN.DERIV = TRUE
# ===========================================================================

test_that("pen.deriv with lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "lasso",
                pen.deriv = TRUE, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "pen.deriv+lasso")
})

test_that("pen.deriv with mcp.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "mcp",
                pen.deriv = TRUE, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "pen.deriv+mcp")
})


# ===========================================================================
# 4. ELASTIC NET (alpha < 1)
# ===========================================================================

test_that("Elastic net (alpha = 0.5) with lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "lasso",
                alpha = 0.5, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "alpha=0.5+lasso")
})


# ===========================================================================
# 5. GROUP PENALTY FEATURES (new infrastructure)
# ===========================================================================

test_that("User-defined multi-covariate groups with grp.lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.groups = list(pair = c(1, 2), single = 3)
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "user-groups+grp.lasso")
})

test_that("User-defined groups by name with grp.mcp.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "grp.mcp", num.tau = 5,
                control = list(
                  dif.groups = list(demo = c("age", "study"), other = "gender")
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "named-groups+grp.mcp")
})

test_that("Sqrt weights with user-defined groups.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.groups = list(pair = c(1, 2), single = 3),
                  dif.group.weights = "sqrt"
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "sqrt-weights+grp.lasso")
})

test_that("Separate mode with grp.lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.group.mode = "separate"
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "separate+grp.lasso")
})

test_that("Family override: a1 excluded from group.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.families = list(a1 = list(in_group = FALSE))
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "a1-excluded+grp.lasso")
})

test_that("Multi-covariate groups with Rasch items.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "rasch", pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.groups = list(pair = c(1, 2), single = 3)
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "rasch+user-groups+grp.lasso")
})

test_that("Multi-covariate groups with graded items.", {
  res <- regDIF(graded_data, pred_synth,
                item.type = "graded", pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.groups = list(pair = c("x1", "x3"), single = "x2")
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "graded+user-groups+grp.lasso")
})


# ===========================================================================
# 6. PROXY DATA
# ===========================================================================

test_that("Proxy scores with lasso.", {
  proxy <- rowMeans(ida[, 1:6])
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                prox.data = proxy,
                item.type = "2pl", pen.type = "lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "proxy+lasso")
})

test_that("Proxy scores with grp.lasso.", {
  proxy <- rowMeans(ida[, 1:6])
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                prox.data = proxy,
                item.type = "2pl", pen.type = "grp.lasso", num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "proxy+grp.lasso")
})


# ===========================================================================
# 7. NUMERICAL SANITY: TAU PATH STRUCTURE
# ===========================================================================

test_that("DIF is all zero at max_tau (fully penalized model).", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "lasso", num.tau = 10)
  # First completed model (highest tau) should have all DIF = 0.
  first_complete <- min(which(!is.na(res$tau)))
  expect_true(all(res$dif[, first_complete] == 0),
              info = "DIF should be all zero at max_tau")
})

test_that("DIF is all zero at max_tau for grp.lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "grp.lasso", num.tau = 10)
  first_complete <- min(which(!is.na(res$tau)))
  expect_true(all(res$dif[, first_complete] == 0),
              info = "Group LASSO DIF should be all zero at max_tau")
})

test_that("DIF is generally non-zero at smallest tau.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = "2pl", pen.type = "lasso", num.tau = 10)
  last_complete <- max(which(!is.na(res$tau)))
  # At least some DIF parameters should be non-zero at the smallest tau.
  expect_true(any(res$dif[, last_complete] != 0),
              info = "Expected some non-zero DIF at smallest tau")
})
