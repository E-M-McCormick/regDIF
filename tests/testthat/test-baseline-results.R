###############################################################################
# Baseline Results: Generate and Save Reference Output for Regression Tests
#
# PURPOSE:
# Run regDIF with all 4 penalty types on the ida dataset and save the
# results as .rds files. These baselines serve as ground truth for verifying
# that future code changes (Phase B: M-step rewrite, Phase C: max_tau fix)
# do NOT alter numerical results for existing functionality.
#
# USAGE:
# Run this file once to generate baselines:
#   source("tests/testthat/test-baseline-results.R")
#
# The baselines are saved to tests/testthat/baselines/ and checked into
# version control so that testthat can compare against them.
#
# After running Phase B/C changes, the test-no-behavioral-change.R file
# will load these baselines and compare against fresh runs.
###############################################################################

context("baseline-results")

# ---- Setup ----
baseline_dir <- file.path(testthat::test_path(), "baselines")

# ---- Helper: run regDIF with a specific penalty type ----
run_baseline <- function(pen_type, num_tau = 10, ...) {
  regDIF(
    item.data = ida[, 1:6],
    pred.data = ida[, 7:9],
    item.type = "2pl",
    pen.type  = pen_type,
    num.tau   = num_tau,
    ...
  )
}

# ---- Generate and save baselines (only if directory doesn't exist) ----
test_that("Baseline results can be generated and saved.", {

  if (!dir.exists(baseline_dir)) {
    dir.create(baseline_dir, recursive = TRUE)
  }

  # Generate any missing baselines. Only regenerates what's missing,
  # so existing baselines are preserved.
  types <- list(
    list(pen = "lasso",     file = "lasso_baseline.rds"),
    list(pen = "mcp",       file = "mcp_baseline.rds"),
    list(pen = "grp.lasso", file = "grp_lasso_baseline.rds"),
    list(pen = "grp.mcp",   file = "grp_mcp_baseline.rds")
  )

  any_generated <- FALSE
  for (tp in types) {
    f <- file.path(baseline_dir, tp$file)
    if (!file.exists(f)) {
      cat("Generating", tp$pen, "baseline...\n")
      res <- run_baseline(tp$pen, num_tau = 10)
      saveRDS(res, f)
      any_generated <- TRUE
    }
  }
  if (!any_generated) skip("All baselines already exist.")

  for (tp in types) {
    expect_true(file.exists(file.path(baseline_dir, tp$file)))
  }
})


# ---- Verify current results match baselines ----
test_that("LASSO results match baseline.", {
  baseline_file <- file.path(baseline_dir, "lasso_baseline.rds")
  skip_if_not(file.exists(baseline_file), "Baseline not yet generated.")

  baseline <- readRDS(baseline_file)
  current  <- run_baseline("lasso", num_tau = 10)

  expect_equal(current$dif, baseline$dif, tolerance = 1e-8,
               info = "LASSO DIF coefficients changed")
  expect_equal(current$base, baseline$base, tolerance = 1e-8,
               info = "LASSO base parameters changed")
  expect_equal(current$impact, baseline$impact, tolerance = 1e-8,
               info = "LASSO impact parameters changed")
})

test_that("MCP results match baseline.", {
  baseline_file <- file.path(baseline_dir, "mcp_baseline.rds")
  skip_if_not(file.exists(baseline_file), "Baseline not yet generated.")

  baseline <- readRDS(baseline_file)
  current  <- run_baseline("mcp", num_tau = 10)

  expect_equal(current$dif, baseline$dif, tolerance = 1e-8,
               info = "MCP DIF coefficients changed")
  expect_equal(current$base, baseline$base, tolerance = 1e-8,
               info = "MCP base parameters changed")
  expect_equal(current$impact, baseline$impact, tolerance = 1e-8,
               info = "MCP impact parameters changed")
})

test_that("Group LASSO results match baseline.", {
  baseline_file <- file.path(baseline_dir, "grp_lasso_baseline.rds")
  skip_if_not(file.exists(baseline_file), "Baseline not yet generated.")

  baseline <- readRDS(baseline_file)
  current  <- run_baseline("grp.lasso", num_tau = 10)

  expect_equal(current$dif, baseline$dif, tolerance = 1e-8,
               info = "Group LASSO DIF coefficients changed")
  expect_equal(current$base, baseline$base, tolerance = 1e-8,
               info = "Group LASSO base parameters changed")
  expect_equal(current$impact, baseline$impact, tolerance = 1e-8,
               info = "Group LASSO impact parameters changed")
})

test_that("Group MCP results match baseline.", {
  baseline_file <- file.path(baseline_dir, "grp_mcp_baseline.rds")
  skip_if_not(file.exists(baseline_file), "Baseline not yet generated.")

  baseline <- readRDS(baseline_file)
  current  <- run_baseline("grp.mcp", num_tau = 10)

  expect_equal(current$dif, baseline$dif, tolerance = 1e-8,
               info = "Group MCP DIF coefficients changed")
  expect_equal(current$base, baseline$base, tolerance = 1e-8,
               info = "Group MCP base parameters changed")
  expect_equal(current$impact, baseline$impact, tolerance = 1e-8,
               info = "Group MCP impact parameters changed")
})
