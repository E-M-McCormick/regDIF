###############################################################################
# Extended Stress Tests: Additional Combinations + Random Seed Variations
#
# Supplements test-stress.R with:
# 1. Different random seeds for synthetic data (checks robustness to data)
# 2. New argument combinations not covered in the original stress tests
# 3. Gamma parameter variations for MCP
# 4. Custom tau sequences
# 5. Cross-combinations of features (e.g., anchor + pen.deriv + group)
###############################################################################

context("stress-tests-extended")


# ===========================================================================
# HELPER: Generate synthetic data with a given seed
# ===========================================================================

make_synth_data <- function(seed, n = 150) {
  set.seed(seed)
  theta <- rnorm(n)
  preds <- data.frame(
    x1 = rnorm(n),
    x2 = rbinom(n, 1, 0.5) * 2 - 1,
    x3 = rnorm(n)
  )

  # Binary items
  bin <- matrix(NA, n, 6)
  for (j in 1:6) {
    bin[, j] <- rbinom(n, 1, plogis(theta * runif(1, 0.6, 1.4) + rnorm(1)))
  }
  colnames(bin) <- paste0("item", 1:6)

  # Graded items (4 categories)
  grd <- matrix(NA, n, 4)
  for (j in 1:4) {
    lat <- theta + rnorm(n, sd = 0.8)
    grd[, j] <- cut(lat, breaks = c(-Inf, -0.5, 0.3, 1.0, Inf), labels = FALSE)
  }
  colnames(grd) <- paste0("item", 1:4)

  # CFA items (continuous)
  cfa <- matrix(NA, n, 4)
  for (j in 1:4) {
    cfa[, j] <- theta * runif(1, 0.5, 1.5) + rnorm(1) + rnorm(n)
  }
  colnames(cfa) <- paste0("item", 1:4)

  list(bin = as.data.frame(bin),
       grd = as.data.frame(grd),
       cfa = as.data.frame(cfa),
       pred = preds,
       proxy = scale(rowMeans(bin))[, 1])
}

check_sanity <- function(res, label) {
  completed <- which(!is.na(res$tau))
  if (length(completed) > 0) {
    expect_false(any(is.nan(res$dif[, completed])),
                 info = paste(label, "- NaN in DIF"))
  }
}


# ===========================================================================
# 1. SEED VARIATION: Rerun core configs with different random data
# ===========================================================================

for (seed in c(42, 999, 2026)) {

  d <- make_synth_data(seed)

  test_that(paste0("2PL + lasso (seed=", seed, ")."), {
    res <- regDIF(d$bin, d$pred, item.type = "2pl",
                  pen.type = "lasso", num.tau = 5)
    expect_s3_class(res, "regDIF")
    check_sanity(res, paste0("2pl+lasso+seed", seed))
  })

  test_that(paste0("2PL + grp.lasso (seed=", seed, ")."), {
    res <- regDIF(d$bin, d$pred, item.type = "2pl",
                  pen.type = "grp.lasso", num.tau = 5)
    expect_s3_class(res, "regDIF")
    check_sanity(res, paste0("2pl+grp.lasso+seed", seed))
  })

  test_that(paste0("Rasch + grp.mcp (seed=", seed, ")."), {
    res <- regDIF(d$bin, d$pred, item.type = "rasch",
                  pen.type = "grp.mcp", num.tau = 5)
    expect_s3_class(res, "regDIF")
    check_sanity(res, paste0("rasch+grp.mcp+seed", seed))
  })

  test_that(paste0("Graded + lasso (seed=", seed, ")."), {
    res <- regDIF(d$grd, d$pred, item.type = "graded",
                  pen.type = "lasso", num.tau = 5)
    expect_s3_class(res, "regDIF")
    check_sanity(res, paste0("graded+lasso+seed", seed))
  })

  test_that(paste0("CFA + grp.lasso (seed=", seed, ")."), {
    res <- regDIF(d$cfa, d$pred, item.type = "cfa",
                  pen.type = "grp.lasso", num.tau = 5)
    expect_s3_class(res, "regDIF")
    check_sanity(res, paste0("cfa+grp.lasso+seed", seed))
  })
}


# ===========================================================================
# 2. GAMMA PARAMETER VARIATIONS (MCP)
# ===========================================================================

test_that("MCP with gamma = 2 (fast taper).", {
  d <- make_synth_data(100)
  res <- regDIF(d$bin, d$pred, item.type = "2pl",
                pen.type = "mcp", gamma = 2, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "mcp+gamma2")
})

test_that("MCP with gamma = 10 (slow taper).", {
  d <- make_synth_data(100)
  res <- regDIF(d$bin, d$pred, item.type = "2pl",
                pen.type = "mcp", gamma = 10, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "mcp+gamma10")
})

test_that("Group MCP with gamma = 2.", {
  d <- make_synth_data(100)
  res <- regDIF(d$bin, d$pred, item.type = "2pl",
                pen.type = "grp.mcp", gamma = 2, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "grp.mcp+gamma2")
})


# ===========================================================================
# 3. CUSTOM TAU SEQUENCES
# ===========================================================================

test_that("User-supplied tau sequence with lasso.", {
  d <- make_synth_data(200)
  res <- regDIF(d$bin, d$pred, item.type = "2pl",
                pen.type = "lasso", tau = seq(1, 0, -0.2))
  expect_s3_class(res, "regDIF")
  expect_equal(length(res$tau), 6)
  check_sanity(res, "custom-tau+lasso")
})

test_that("User-supplied tau sequence with grp.lasso.", {
  d <- make_synth_data(200)
  res <- regDIF(d$bin, d$pred, item.type = "2pl",
                pen.type = "grp.lasso", tau = seq(1, 0, -0.25))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "custom-tau+grp.lasso")
})

test_that("Single large tau (all DIF should be zero).", {
  d <- make_synth_data(200)
  res <- regDIF(d$bin, d$pred, item.type = "2pl",
                pen.type = "lasso", tau = 100)
  expect_s3_class(res, "regDIF")
  expect_true(all(res$dif[, 1] == 0),
              info = "Expected all-zero DIF at tau=100")
})


# ===========================================================================
# 4. CROSS-COMBINATIONS (anchor + pen.deriv + group features)
# ===========================================================================

test_that("Anchor + pen.deriv + lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9], item.type = "2pl",
                pen.type = "lasso", pen.deriv = TRUE,
                anchor = 1, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "anchor+pen.deriv+lasso")
})

test_that("Anchor + user groups + grp.lasso.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9], item.type = "2pl",
                pen.type = "grp.lasso", anchor = c(1, 2), num.tau = 5,
                control = list(
                  dif.groups = list(pair = c("age", "study"), other = "gender")
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "anchor+usergroups+grp.lasso")
})

test_that("Anchor + separate mode + grp.mcp.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9], item.type = "2pl",
                pen.type = "grp.mcp", anchor = 3, num.tau = 5,
                control = list(dif.group.mode = "separate"))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "anchor+separate+grp.mcp")
})

test_that("Elastic net + pen.deriv.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9], item.type = "2pl",
                pen.type = "lasso", alpha = 0.5, pen.deriv = TRUE,
                num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "alpha0.5+pen.deriv")
})

test_that("Proxy + grp.mcp + anchor.", {
  proxy <- rowMeans(ida[, 1:6])
  res <- regDIF(ida[, 1:6], ida[, 7:9], prox.data = proxy,
                item.type = "2pl", pen.type = "grp.mcp",
                anchor = 1, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "proxy+grp.mcp+anchor")
})


# ===========================================================================
# 5. GRADED + GROUP FEATURES (new capability from Phase B)
# ===========================================================================

test_that("Graded + separate mode + grp.lasso.", {
  d <- make_synth_data(300)
  res <- regDIF(d$grd, d$pred, item.type = "graded",
                pen.type = "grp.lasso", num.tau = 5,
                control = list(dif.group.mode = "separate"))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "graded+separate+grp.lasso")
})

test_that("Graded + user groups + grp.mcp.", {
  d <- make_synth_data(300)
  res <- regDIF(d$grd, d$pred, item.type = "graded",
                pen.type = "grp.mcp", num.tau = 5,
                control = list(
                  dif.groups = list(pair = c("x1", "x3"), single = "x2")
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "graded+usergroups+grp.mcp")
})

test_that("Graded + anchor + grp.lasso.", {
  d <- make_synth_data(300)
  res <- regDIF(d$grd, d$pred, item.type = "graded",
                pen.type = "grp.lasso", anchor = 1, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "graded+anchor+grp.lasso")
})


# ===========================================================================
# 6. CFA + GROUP FEATURES
# ===========================================================================

test_that("CFA + separate mode.", {
  d <- make_synth_data(400)
  res <- regDIF(d$cfa, d$pred, item.type = "cfa",
                pen.type = "grp.lasso", num.tau = 5,
                control = list(dif.group.mode = "separate"))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "cfa+separate+grp.lasso")
})

test_that("CFA + s1 excluded from group.", {
  d <- make_synth_data(400)
  res <- regDIF(d$cfa, d$pred, item.type = "cfa",
                pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.families = list(s1 = list(in_group = FALSE))
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "cfa+s1excluded+grp.lasso")
})

test_that("CFA + s1 not penalized (free estimation).", {
  d <- make_synth_data(400)
  res <- regDIF(d$cfa, d$pred, item.type = "cfa",
                pen.type = "lasso", num.tau = 5,
                control = list(
                  dif.families = list(s1 = list(penalized = FALSE))
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "cfa+s1free+lasso")
})


# ===========================================================================
# 7. MIXED ITEM TYPES + GROUP FEATURES
# ===========================================================================

test_that("Mixed (2pl + rasch) + user groups.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = c("2pl", "2pl", "2pl", "rasch", "rasch", "rasch"),
                pen.type = "grp.lasso", num.tau = 5,
                control = list(
                  dif.groups = list(pair = c(1, 2), single = 3)
                ))
  expect_s3_class(res, "regDIF")
  check_sanity(res, "mixed+usergroups+grp.lasso")
})

test_that("Mixed (2pl + rasch) + anchor + mcp.", {
  res <- regDIF(ida[, 1:6], ida[, 7:9],
                item.type = c("2pl", "2pl", "2pl", "rasch", "rasch", "rasch"),
                pen.type = "mcp", anchor = 1, num.tau = 5)
  expect_s3_class(res, "regDIF")
  check_sanity(res, "mixed+anchor+mcp")
})


# ===========================================================================
# 8. NUMERICAL CONSISTENCY ACROSS SEEDS
# ===========================================================================

test_that("Same seed produces identical results (determinism check).", {
  d1 <- make_synth_data(777)
  d2 <- make_synth_data(777)
  res1 <- regDIF(d1$bin, d1$pred, item.type = "2pl",
                 pen.type = "lasso", num.tau = 5)
  res2 <- regDIF(d2$bin, d2$pred, item.type = "2pl",
                 pen.type = "lasso", num.tau = 5)
  expect_equal(res1$dif, res2$dif, tolerance = 0,
               info = "Same data should produce identical DIF coefficients")
})
