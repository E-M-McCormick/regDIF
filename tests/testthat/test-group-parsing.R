###############################################################################
# Unit Tests for Group Parsing Infrastructure (R/group_parsing.R)
#
# These tests verify that the DIF parameter registry, group parsing,
# weight computation, and index building all produce correct results
# WITHOUT changing any estimation behavior. This is Phase A testing.
###############################################################################

context("group-parsing")

# ===========================================================================
# DIF PARAMETER REGISTRY
# ===========================================================================

test_that("get_dif_param_families returns correct families per item type.", {

  rasch_fams <- get_dif_param_families("rasch")
  expect_length(rasch_fams, 1)
  expect_equal(rasch_fams[[1]]$name, "c1")

  twopl_fams <- get_dif_param_families("2pl")
  expect_length(twopl_fams, 2)
  expect_equal(twopl_fams[[1]]$name, "c1")
  expect_equal(twopl_fams[[2]]$name, "a1")

  graded_fams <- get_dif_param_families("graded")
  expect_length(graded_fams, 2)
  expect_equal(graded_fams[[1]]$name, "c1")
  expect_equal(graded_fams[[2]]$name, "a1")

  cfa_fams <- get_dif_param_families("cfa")
  expect_length(cfa_fams, 3)
  expect_equal(cfa_fams[[1]]$name, "c1")
  expect_equal(cfa_fams[[2]]$name, "a1")
  expect_equal(cfa_fams[[3]]$name, "s1")
})

test_that("Unknown item type warns and defaults to 2PL.", {
  expect_warning(
    fams <- get_dif_param_families("unknown_type"),
    "Unknown item type"
  )
  expect_length(fams, 2)
})


# ===========================================================================
# FAMILY OVERRIDES
# ===========================================================================

test_that("apply_family_overrides correctly disables estimation.", {
  fams <- get_dif_param_families("2pl")
  overridden <- apply_family_overrides(
    fams,
    list(a1 = list(estimated = FALSE)),
    "2pl"
  )

  # a1 should now be not estimated, not penalized, not in group.
  a1 <- overridden[[2]]
  expect_false(a1$estimated)
  expect_false(a1$penalized)
  expect_false(a1$in_group)

  # c1 should be unchanged.
  c1 <- overridden[[1]]
  expect_true(c1$estimated)
  expect_true(c1$penalized)
  expect_true(c1$in_group)
})

test_that("apply_family_overrides enforces hierarchy.", {
  fams <- get_dif_param_families("cfa")

  # Set penalized = FALSE but in_group = TRUE (contradictory).
  # Hierarchy should force in_group = FALSE.
  overridden <- apply_family_overrides(
    fams,
    list(s1 = list(penalized = FALSE, in_group = TRUE)),
    "cfa"
  )
  s1 <- overridden[[3]]
  expect_true(s1$estimated)
  expect_false(s1$penalized)
  expect_false(s1$in_group)  # Forced FALSE by hierarchy.
})

test_that("Overrides for non-existent families are silently skipped.", {
  fams <- get_dif_param_families("2pl")
  # g1 doesn't exist in 2PL; should not error or warn.
  overridden <- apply_family_overrides(
    fams,
    list(g1 = list(estimated = FALSE)),
    "2pl"
  )
  # Result should be identical to input.
  expect_equal(length(overridden), length(fams))
})

test_that("Invalid override fields produce errors.", {
  fams <- get_dif_param_families("2pl")
  expect_error(
    apply_family_overrides(fams, list(c1 = list(bogus = TRUE)), "2pl"),
    "Unknown field"
  )
})


# ===========================================================================
# COUNTING AND QUERYING HELPERS
# ===========================================================================

test_that("count_dif_families returns correct counts.", {
  # Paired mode: count of in_group families.
  expect_equal(count_dif_families("rasch", "paired"), 1)
  expect_equal(count_dif_families("2pl",   "paired"), 2)
  expect_equal(count_dif_families("cfa",   "paired"), 3)

  # Separate mode: always 1 (each family is its own sub-group).
  expect_equal(count_dif_families("2pl", "separate"), 1)
  expect_equal(count_dif_families("cfa", "separate"), 1)
})

test_that("count_dif_families respects overrides.", {
  # CFA with s1 not in group -> only 2 families in group.
  expect_equal(
    count_dif_families("cfa", "paired",
                       user_overrides = list(s1 = list(in_group = FALSE))),
    2
  )
})

test_that("get_grouped_family_names returns correct names.", {
  expect_equal(get_grouped_family_names("rasch"), "c1")
  expect_equal(get_grouped_family_names("2pl"), c("c1", "a1"))
  expect_equal(get_grouped_family_names("cfa"), c("c1", "a1", "s1"))
})

test_that("get_non_grouped_families categorizes correctly.", {
  # CFA with s1 set to penalized=FALSE.
  result <- get_non_grouped_families(
    "cfa",
    user_overrides = list(s1 = list(penalized = FALSE))
  )
  expect_length(result$penalized_scalar, 0)
  expect_equal(result$free, "s1")

  # CFA with s1 set to in_group=FALSE (but still penalized).
  result2 <- get_non_grouped_families(
    "cfa",
    user_overrides = list(s1 = list(in_group = FALSE))
  )
  expect_equal(result2$penalized_scalar, "s1")
  expect_length(result2$free, 0)
})


# ===========================================================================
# GROUP PARSING (parse_dif_groups)
# ===========================================================================

test_that("parse_dif_groups returns NULL groups_idx for scalar penalties.", {
  result <- parse_dif_groups(
    dif_groups        = NULL,
    dif_group_mode    = NULL,
    dif_group_weights = NULL,
    dif_families      = NULL,
    pred_data         = matrix(rnorm(30), ncol = 3),
    pen_type          = "lasso",
    num_predictors    = 3,
    item_type         = rep("2pl", 5),
    num_items         = 5,
    num_responses     = rep(2, 5)
  )
  expect_null(result$groups_idx)
})

test_that("Default singleton groups are created for group penalties.", {
  pred <- matrix(rnorm(30), ncol = 3)
  colnames(pred) <- c("age", "sex", "ses")

  result <- parse_dif_groups(
    dif_groups        = NULL,
    dif_group_mode    = NULL,
    dif_group_weights = NULL,
    dif_families      = NULL,
    pred_data         = pred,
    pen_type          = "grp.lasso",
    num_predictors    = 3,
    item_type         = rep("2pl", 5),
    num_items         = 5,
    num_responses     = rep(2, 5)
  )

  expect_length(result$groups_idx, 3)
  expect_equal(result$groups_idx[["age"]], 1)
  expect_equal(result$groups_idx[["sex"]], 2)
  expect_equal(result$groups_idx[["ses"]], 3)
  expect_equal(result$group_mode, "paired")
})

test_that("Per-item sqrt weights are correct for mixed item types.", {
  pred <- matrix(rnorm(30), ncol = 3)
  colnames(pred) <- c("x1", "x2", "x3")

  # Mix of Rasch (1 family) and 2PL (2 families) items.
  # Explicitly request sqrt weights (not the default for singletons).
  result <- parse_dif_groups(
    dif_groups        = NULL,
    dif_group_mode    = "paired",
    dif_group_weights = "sqrt",
    dif_families      = NULL,
    pred_data         = pred,
    pen_type          = "grp.lasso",
    num_predictors    = 3,
    item_type         = c("rasch", "2pl", "rasch"),
    num_items         = 3,
    num_responses     = rep(2, 3)
  )

  # Singleton groups with 1 covariate each.
  # Rasch: sqrt(1 * 1) = 1.0
  # 2PL:   sqrt(1 * 2) = sqrt(2)
  expect_equal(result$item_group_weights[1, 1], 1.0)       # Rasch
  expect_equal(result$item_group_weights[2, 1], sqrt(2))    # 2PL
  expect_equal(result$item_group_weights[3, 1], 1.0)       # Rasch
})

test_that("Default singleton groups use weight = 1 (backward compat).", {
  pred <- matrix(rnorm(30), ncol = 3)
  colnames(pred) <- c("x1", "x2", "x3")

  result <- parse_dif_groups(
    dif_groups        = NULL,
    dif_group_mode    = NULL,
    dif_group_weights = NULL,  # Default.
    dif_families      = NULL,
    pred_data         = pred,
    pen_type          = "grp.lasso",
    num_predictors    = 3,
    item_type         = rep("2pl", 3),
    num_items         = 3,
    num_responses     = rep(2, 3)
  )

  # Default singletons should use weight = 1, not sqrt(2).
  expect_equal(result$item_group_weights[1, 1], 1)
  expect_equal(result$item_group_weights[2, 2], 1)
})

test_that("Multi-covariate groups produce correct weights.", {
  pred <- matrix(rnorm(60), ncol = 6)
  colnames(pred) <- paste0("x", 1:6)

  result <- parse_dif_groups(
    dif_groups        = list(spline = c("x1", "x2", "x3"),
                             other  = c("x4", "x5", "x6")),
    dif_group_mode    = "paired",
    dif_group_weights = "sqrt",
    dif_families      = NULL,
    pred_data         = pred,
    pen_type          = "grp.lasso",
    num_predictors    = 6,
    item_type         = rep("2pl", 4),
    num_items         = 4,
    num_responses     = rep(2, 4)
  )

  # Group "spline" has 3 covariates, 2PL has 2 families: sqrt(3*2) = sqrt(6).
  expect_equal(result$item_group_weights[1, 1], sqrt(6))
  # Group "other" has 3 covariates: same weight.
  expect_equal(result$item_group_weights[1, 2], sqrt(6))
})

test_that("User-defined groups with name resolution work.", {
  pred <- matrix(rnorm(30), ncol = 3)
  colnames(pred) <- c("age", "sex", "ses")

  result <- parse_dif_groups(
    dif_groups        = list(demo = c("age", "ses"), gender = "sex"),
    dif_group_mode    = "paired",
    dif_group_weights = "sqrt",
    dif_families      = NULL,
    pred_data         = pred,
    pen_type          = "grp.lasso",
    num_predictors    = 3,
    item_type         = rep("2pl", 4),
    num_items         = 4,
    num_responses     = rep(2, 4)
  )

  expect_length(result$groups_idx, 2)
  expect_equal(result$groups_idx[["demo"]], c(1, 3))
  expect_equal(result$groups_idx[["gender"]], 2)
})

test_that("Overlapping groups are rejected.", {
  pred <- matrix(rnorm(30), ncol = 3)
  colnames(pred) <- c("x1", "x2", "x3")

  expect_error(
    parse_dif_groups(
      dif_groups        = list(g1 = c("x1", "x2"), g2 = c("x2", "x3")),
      dif_group_mode    = "paired",
      dif_group_weights = "sqrt",
      dif_families      = NULL,
      pred_data         = pred,
      pen_type          = "grp.lasso",
      num_predictors    = 3,
      item_type         = rep("2pl", 3),
      num_items         = 3,
      num_responses     = rep(2, 3)
    ),
    "Overlapping"
  )
})

test_that("Uncovered covariates get singleton groups automatically.", {
  pred <- matrix(rnorm(30), ncol = 3)
  colnames(pred) <- c("x1", "x2", "x3")

  result <- parse_dif_groups(
    dif_groups        = list(pair = c("x1", "x2")),
    dif_group_mode    = "paired",
    dif_group_weights = "sqrt",
    dif_families      = NULL,
    pred_data         = pred,
    pen_type          = "grp.lasso",
    num_predictors    = 3,
    item_type         = rep("2pl", 3),
    num_items         = 3,
    num_responses     = rep(2, 3)
  )

  # x3 should get its own singleton group.
  expect_length(result$groups_idx, 2)
  expect_equal(result$groups_idx[["x3"]], 3)
})


# ===========================================================================
# INDEX BUILDING (build_group_param_indices)
# ===========================================================================

test_that("Index positions are correct for 2PL binary items.", {
  # 2PL with 3 predictors: c0 at [1], a0 at [2],
  # c1 at [3,4,5], a1 at [6,7,8].
  idx <- build_group_param_indices(
    cov_indices        = c(1, 3),  # Group with covariates 1 and 3.
    num_predictors     = 3,
    item_type          = "2pl",
    num_responses_item = 2,
    group_mode         = "paired"
  )

  # c1 for cov 1 -> position 3, c1 for cov 3 -> position 5.
  # a1 for cov 1 -> position 6, a1 for cov 3 -> position 8.
  expect_equal(idx$paired_idx, c(3, 5, 6, 8))
  expect_equal(idx$family_ranges$c1, 1:2)
  expect_equal(idx$family_ranges$a1, 3:4)
})

test_that("Index positions are correct for Rasch items.", {
  idx <- build_group_param_indices(
    cov_indices        = c(1, 2),
    num_predictors     = 3,
    item_type          = "rasch",
    num_responses_item = 2,
    group_mode         = "paired"
  )

  # Rasch: only c1 in the group (no a1).
  # c1 for cov 1 -> position 3, c1 for cov 2 -> position 4.
  expect_equal(idx$paired_idx, c(3, 4))
  expect_equal(idx$family_ranges$c1, 1:2)
  expect_null(idx$family_ranges$a1)
})

test_that("Index positions are correct for graded items.", {
  # Graded with 4 response categories, 3 predictors:
  # [1:3] thresholds, [4] a0, [5:7] c1, [8:10] a1.
  idx <- build_group_param_indices(
    cov_indices        = c(2),
    num_predictors     = 3,
    item_type          = "graded",
    num_responses_item = 4,
    group_mode         = "paired"
  )

  # c1 for cov 2 -> base_offset(4) + 2 = position 6.
  # a1 for cov 2 -> base_offset(4) + 3 + 2 = position 9.
  expect_equal(idx$paired_idx, c(6, 9))
})

test_that("Index positions are correct for CFA items.", {
  # CFA with 3 predictors:
  # [1] c0, [2] a0, [3:5] c1, [6:8] a1, [9] s0, [10:12] s1.
  idx <- build_group_param_indices(
    cov_indices        = c(1, 2, 3),
    num_predictors     = 3,
    item_type          = "cfa",
    num_responses_item = 2,  # Not used for CFA offset.
    group_mode         = "paired"
  )

  # All 3 families in the group:
  # c1: positions 3, 4, 5
  # a1: positions 6, 7, 8
  # s1: positions 10, 11, 12  (s0 at 9, s1 starts at 10)
  expect_equal(idx$paired_idx, c(3, 4, 5, 6, 7, 8, 10, 11, 12))
  expect_equal(idx$family_ranges$c1, 1:3)
  expect_equal(idx$family_ranges$a1, 4:6)
  expect_equal(idx$family_ranges$s1, 7:9)
})

test_that("Separate mode returns per-family index lists.", {
  idx <- build_group_param_indices(
    cov_indices        = c(1, 3),
    num_predictors     = 3,
    item_type          = "2pl",
    num_responses_item = 2,
    group_mode         = "separate"
  )

  expect_equal(idx$c1, c(3, 5))
  expect_equal(idx$a1, c(6, 8))
})

test_that("CFA with s1 excluded from group via overrides.", {
  idx <- build_group_param_indices(
    cov_indices        = c(1, 2),
    num_predictors     = 3,
    item_type          = "cfa",
    num_responses_item = 2,
    group_mode         = "paired",
    user_overrides     = list(s1 = list(in_group = FALSE))
  )

  # Only c1 and a1 in the group; s1 excluded.
  # c1 for cov 1,2 -> positions 3, 4.
  # a1 for cov 1,2 -> positions 6, 7.
  expect_equal(idx$paired_idx, c(3, 4, 6, 7))
  expect_equal(idx$family_ranges$c1, 1:2)
  expect_equal(idx$family_ranges$a1, 3:4)
  expect_null(idx$family_ranges$s1)
})


# ===========================================================================
# PRE-COMPUTED INDEX TABLE (precompute_group_indices)
# ===========================================================================

test_that("precompute_group_indices returns correct structure.", {
  pred <- matrix(rnorm(30), ncol = 3)
  colnames(pred) <- c("x1", "x2", "x3")

  group_spec <- parse_dif_groups(
    dif_groups        = NULL,
    dif_group_mode    = "paired",
    dif_group_weights = "sqrt",
    dif_families      = NULL,
    pred_data         = pred,
    pen_type          = "grp.lasso",
    num_predictors    = 3,
    item_type         = c("2pl", "rasch", "2pl"),
    num_items         = 3,
    num_responses     = rep(2, 3)
  )

  idx_table <- precompute_group_indices(
    group_spec     = group_spec,
    num_items      = 3,
    item_type      = c("2pl", "rasch", "2pl"),
    num_responses  = rep(2, 3),
    num_predictors = 3
  )

  # Should be a list of 3 items, each with 3 groups.
  expect_length(idx_table, 3)
  expect_length(idx_table[[1]], 3)
  expect_length(idx_table[[2]], 3)

  # Item 1 (2PL), group 1 (cov 1): paired_idx should have c1 and a1 for cov 1.
  expect_equal(idx_table[[1]][[1]]$paired_idx, c(3, 6))

  # Item 2 (Rasch), group 1 (cov 1): paired_idx should have only c1 for cov 1.
  expect_equal(idx_table[[2]][[1]]$paired_idx, c(3))
})

test_that("precompute_group_indices returns NULL for scalar penalties.", {
  result <- precompute_group_indices(
    group_spec     = list(dif_families_override = NULL),  # From scalar penalty.
    num_items      = 3,
    item_type      = rep("2pl", 3),
    num_responses  = rep(2, 3),
    num_predictors = 3
  )
  expect_null(result)
})


# ===========================================================================
# INTEGRATION: Verify preprocess() populates group infrastructure
# ===========================================================================

test_that("preprocess() populates group_spec for group penalties.", {
  preprocessed <- preprocess(
    item.data     = ida[, 1:6],
    pred.data     = ida[, 7:9],
    prox.data     = NULL,
    item.type     = "2pl",
    pen.type      = "grp.lasso",
    tau           = NULL,
    num.tau       = 10,
    anchor        = NULL,
    stdz          = TRUE,
    free.theta.var = FALSE,
    control       = list(),
    call          = call("regDIF")
  )

  gs <- preprocessed$final_control$group_spec
  expect_false(is.null(gs))
  expect_false(is.null(gs$groups_idx))
  expect_equal(gs$group_mode, "paired")

  # 3 predictors -> 3 singleton groups.
  expect_length(gs$groups_idx, 3)

  # item_group_weights: 6 items x 3 groups.
  expect_equal(dim(gs$item_group_weights), c(6, 3))

  # Default singletons use weight = 1 (backward compatible with original
  # code that called grp_soft_threshold without weights).
  expect_equal(gs$item_group_weights[1, 1], 1)

  # Pre-computed indices should also exist.
  gpi <- preprocessed$final_control$group_param_idx
  expect_false(is.null(gpi))
  expect_length(gpi, 6)       # 6 items.
  expect_length(gpi[[1]], 3)  # 3 groups per item.
})

test_that("preprocess() populates group_spec for scalar penalties.", {
  preprocessed <- preprocess(
    item.data     = ida[, 1:6],
    pred.data     = ida[, 7:9],
    prox.data     = NULL,
    item.type     = "2pl",
    pen.type      = "lasso",
    tau           = NULL,
    num.tau       = 10,
    anchor        = NULL,
    stdz          = TRUE,
    free.theta.var = FALSE,
    control       = list(),
    call          = call("regDIF")
  )

  gs <- preprocessed$final_control$group_spec
  # For scalar penalties, groups_idx should be NULL.
  expect_null(gs$groups_idx)

  # group_param_idx should be NULL.
  expect_null(preprocessed$final_control$group_param_idx)
})
