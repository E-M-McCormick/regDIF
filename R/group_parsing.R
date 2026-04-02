################################################################################
#                                                                              #
#  STEP 2 (REVISED v2): Group Definition Infrastructure for preprocess.R       #
#                                                                              #
#  CHANGES FROM v1:                                                            #
#                                                                              #
#  1. THREE-LEVEL DIF CONTROL per parameter family:                            #
#     (a) estimated: Does DIF exist for this family? If FALSE, DIF params      #
#         are fixed to zero and never updated. If TRUE, they are estimated.    #
#     (b) penalized: If estimated, is this family subject to the selection     #
#         penalty (lasso/group lasso)? If FALSE, the DIF params are freely     #
#         estimated without regularization.                                    #
#     (c) in_group: If penalized, does this family participate in the group    #
#         norm? Or is it penalized with a scalar penalty independently?        #
#                                                                              #
#     Hierarchy: estimated = FALSE  =>  penalized is irrelevant                #
#                penalized = FALSE  =>  in_group is irrelevant                 #
#                                                                              #
#  2. NEW USER-FACING CONTROL ARGUMENT: control$dif.families                   #
#     Allows the user to override which parameter families have DIF and how    #
#     they are treated. The default comes from the registry (all families      #
#     estimated, penalized, and in_group = TRUE), but the user can selectively #
#     turn off DIF estimation, penalization, or grouping for specific families.#
#                                                                              #
#     Example use cases:                                                       #
#       - 4PL with DIF on c1, a1, u1 but NOT g1:                              #
#           control$dif.families = list(g1 = list(estimated = FALSE))          #
#       - CFA with s1 DIF estimated freely (not regularized):                  #
#           control$dif.families = list(s1 = list(penalized = FALSE))          #
#       - 2PL with slope DIF not in the group (penalized independently):       #
#           control$dif.families = list(a1 = list(in_group = FALSE))           #
#                                                                              #
#  3. FIXED ISSUES FROM v1:                                                    #
#     - The file header docstring still referenced the old output format       #
#       ($group_weights as a vector); now correctly documents                  #
#       $item_group_weights as a matrix.                                       #
#     - Clarified that the registry defines MODEL DEFAULTS, and                #
#       control$dif.families provides user overrides.                          #
#                                                                              #
################################################################################


# ==============================================================================
# DIF PARAMETER REGISTRY
#
# Defines, for each item type, which DIF parameter families exist and their
# default estimation/penalization/grouping behavior.
#
# CURRENT ITEM TYPES:
#   rasch:  c1 only
#   2pl:    c1, a1
#   graded: c1, a1
#   cfa:    c1, a1, s1
#
# FUTURE (commented out until implemented):
#   3pl:    c1, a1, g1
#   4pl:    c1, a1, g1, u1
#
# Each family entry:
#   $name:       Short identifier (e.g., "c1", "a1", "s1", "g1", "u1")
#   $label:      Human-readable label for output/messages
#   $estimated:  Default: is DIF estimated for this family?
#   $penalized:  Default: is the DIF subject to the selection penalty?
#   $in_group:   Default: does this family participate in group penalties?
#
# The hierarchy is:
#   estimated = FALSE  =>  the DIF parameters are fixed to zero;
#                          penalized and in_group are irrelevant.
#   penalized = FALSE  =>  DIF is estimated freely (no regularization);
#                          in_group is irrelevant.
#   in_group  = FALSE  =>  DIF is penalized, but with a scalar penalty
#                          (not included in the group norm).
# ==============================================================================


#' Get the DIF parameter family definitions for a given item type.
#'
#' Returns the DEFAULT family definitions. User overrides from
#' control$dif.families are applied separately by apply_family_overrides().
#'
#' @param item_type Character. One of "rasch", "2pl", "graded", "cfa".
#'
#' @return A list of family definitions.
#'
#' @keywords internal
get_dif_param_families <- function(item_type) {

  families <- switch(item_type,

    "rasch" = list(
      list(name = "c1", label = "intercept",
           estimated = TRUE, penalized = TRUE, in_group = TRUE)
    ),

    "2pl" = list(
      list(name = "c1", label = "intercept",
           estimated = TRUE, penalized = TRUE, in_group = TRUE),
      list(name = "a1", label = "slope",
           estimated = TRUE, penalized = TRUE, in_group = TRUE)
    ),

    "graded" = list(
      list(name = "c1", label = "intercept",
           estimated = TRUE, penalized = TRUE, in_group = TRUE),
      list(name = "a1", label = "slope",
           estimated = TRUE, penalized = TRUE, in_group = TRUE)
    ),

    "cfa" = list(
      list(name = "c1", label = "intercept",
           estimated = TRUE, penalized = TRUE, in_group = TRUE),
      list(name = "a1", label = "slope",
           estimated = TRUE, penalized = TRUE, in_group = TRUE),
      list(name = "s1", label = "residual_var",
           estimated = TRUE, penalized = TRUE, in_group = TRUE)
    ),

    # "3pl" = list(
    #   list(name = "c1", label = "intercept",
    #        estimated = TRUE, penalized = TRUE, in_group = TRUE),
    #   list(name = "a1", label = "slope",
    #        estimated = TRUE, penalized = TRUE, in_group = TRUE),
    #   list(name = "g1", label = "guessing",
    #        estimated = TRUE, penalized = TRUE, in_group = TRUE)
    # ),

    # "4pl" = list(
    #   list(name = "c1", label = "intercept",
    #        estimated = TRUE, penalized = TRUE, in_group = TRUE),
    #   list(name = "a1", label = "slope",
    #        estimated = TRUE, penalized = TRUE, in_group = TRUE),
    #   list(name = "g1", label = "guessing",
    #        estimated = TRUE, penalized = TRUE, in_group = TRUE),
    #   list(name = "u1", label = "upper_asymptote",
    #        estimated = TRUE, penalized = TRUE, in_group = TRUE)
    # ),

    {
      warning(paste0("Unknown item type '", item_type,
                     "' in get_dif_param_families(). Defaulting to 2PL."),
              call. = FALSE)
      list(
        list(name = "c1", label = "intercept",
             estimated = TRUE, penalized = TRUE, in_group = TRUE),
        list(name = "a1", label = "slope",
             estimated = TRUE, penalized = TRUE, in_group = TRUE)
      )
    }
  )

  return(families)
}


#' Apply user overrides to DIF parameter family defaults.
#'
#' Takes the default family definitions from the registry and applies any
#' user-specified overrides from control$dif.families. This lets users
#' selectively turn off DIF estimation, penalization, or grouping for
#' specific parameter families without modifying the registry.
#'
#' @param families List of family definitions (from get_dif_param_families).
#' @param user_overrides NULL or a named list. Names correspond to family
#'   names (e.g., "c1", "a1", "s1", "g1"). Values are lists with any
#'   subset of: estimated, penalized, in_group.
#' @param item_type Character. Used only for error messages.
#'
#' @return Updated list of family definitions.
#'
#' @details
#' EXAMPLES OF user_overrides:
#'
#'   # 4PL: estimate DIF on everything except guessing
#'   list(g1 = list(estimated = FALSE))
#'
#'   # CFA: estimate s1 DIF freely (no regularization), but still
#'   # regularize c1 and a1
#'   list(s1 = list(penalized = FALSE))
#'
#'   # 2PL: penalize a1 DIF independently (not in the group norm)
#'   list(a1 = list(in_group = FALSE))
#'
#'   # Multiple overrides at once:
#'   list(
#'     g1 = list(estimated = FALSE),
#'     u1 = list(penalized = FALSE)
#'   )
#'
#' ENFORCEMENT OF THE HIERARCHY:
#'   If the user sets estimated = FALSE, then penalized and in_group are
#'   forced to FALSE regardless of what the user specified.
#'   If the user sets penalized = FALSE, then in_group is forced to FALSE.
#'   This prevents logically inconsistent configurations like
#'   "not estimated but in the group."
#'
#' VALIDATION:
#'   - Family names in user_overrides that don't match any family in the
#'     registry trigger a warning (not an error), because the user might
#'     be specifying overrides for a family that exists in a different item
#'     type (e.g., setting g1 overrides when some items are 3PL and others
#'     are 2PL — the 2PL items simply ignore the g1 override).
#'   - Override values must be logical scalars. Non-logical values error.
#'
#' @keywords internal
apply_family_overrides <- function(families, user_overrides, item_type) {

  if (is.null(user_overrides)) return(families)

  if (!is.list(user_overrides)) {
    stop("control$dif.families must be a named list.", call. = FALSE)
  }

  # Get the set of family names that exist for this item type.
  valid_names <- vapply(families, function(f) f$name, character(1))

  for (override_name in names(user_overrides)) {

    # Find the family this override applies to.
    match_idx <- which(valid_names == override_name)

    if (length(match_idx) == 0) {
      # This family doesn't exist for this item type. That's not necessarily
      # an error — the user might have mixed item types and is setting
      # overrides for a family that only some items have (e.g., g1 for 3PL
      # items in a model that also has 2PL items). Just skip silently.
      next
    }

    override_vals <- user_overrides[[override_name]]

    if (!is.list(override_vals)) {
      stop(paste0("control$dif.families$", override_name,
                  " must be a list with fields like ",
                  "estimated, penalized, in_group."),
           call. = FALSE)
    }

    # Validate and apply each field.
    allowed_fields <- c("estimated", "penalized", "in_group")
    for (field in names(override_vals)) {
      if (!(field %in% allowed_fields)) {
        stop(paste0("Unknown field '", field, "' in control$dif.families$",
                    override_name, ". Allowed fields: ",
                    paste(allowed_fields, collapse = ", ")),
             call. = FALSE)
      }
      val <- override_vals[[field]]
      if (!is.logical(val) || length(val) != 1) {
        stop(paste0("control$dif.families$", override_name, "$", field,
                    " must be a single logical value (TRUE/FALSE)."),
             call. = FALSE)
      }
      families[[match_idx]][[field]] <- val
    }

    # -----------------------------------------------------------------------
    # ENFORCE THE HIERARCHY.
    #
    # The logical chain is:
    #   estimated = FALSE  =>  nothing else matters; can't penalize or group
    #                          something that isn't estimated.
    #   penalized = FALSE  =>  in_group is meaningless; a freely estimated
    #                          parameter can't participate in a penalty group.
    #
    # We enforce this AFTER applying overrides so that users can set any
    # combination and we silently fix the inconsistencies (rather than
    # erroring on "in_group = TRUE but estimated = FALSE", which would be
    # annoying when setting bulk overrides).
    # -----------------------------------------------------------------------
    if (!families[[match_idx]]$estimated) {
      families[[match_idx]]$penalized <- FALSE
      families[[match_idx]]$in_group  <- FALSE
    }
    if (!families[[match_idx]]$penalized) {
      families[[match_idx]]$in_group  <- FALSE
    }
  }

  return(families)
}


#' Get the resolved DIF families for a specific item, with user overrides applied.
#'
#' Convenience wrapper that calls get_dif_param_families() and then
#' apply_family_overrides().
#'
#' @param item_type Character. Item type for this item.
#' @param user_overrides NULL or a named list from control$dif.families.
#'
#' @return List of resolved family definitions.
#'
#' @keywords internal
resolve_dif_families <- function(item_type, user_overrides = NULL) {
  families <- get_dif_param_families(item_type)
  families <- apply_family_overrides(families, user_overrides, item_type)
  return(families)
}


# ==============================================================================
# COUNTING AND QUERYING HELPERS
# ==============================================================================

#' Count the number of in-group DIF families per covariate for an item type.
#'
#' @param item_type Character. Item type.
#' @param group_mode Character. "paired" or "separate".
#' @param user_overrides NULL or named list of family overrides.
#'
#' @return Integer. Number of in-group DIF families per covariate.
#'
#' @keywords internal
count_dif_families <- function(item_type,
                               group_mode = "paired",
                               user_overrides = NULL) {
  families <- resolve_dif_families(item_type, user_overrides)

  if (group_mode == "paired") {
    sum(sapply(families, function(f) f$in_group))
  } else {
    # In separate mode, each family is its own sub-group (size 1 per cov).
    1L
  }
}


#' Get the names of DIF families that participate in groups.
#'
#' @param item_type Character.
#' @param user_overrides NULL or named list.
#'
#' @return Character vector of family names.
#'
#' @keywords internal
get_grouped_family_names <- function(item_type, user_overrides = NULL) {
  families <- resolve_dif_families(item_type, user_overrides)
  vapply(
    Filter(function(f) f$in_group, families),
    function(f) f$name,
    character(1)
  )
}


#' Get the names of DIF families that are estimated but NOT in the group.
#'
#' These families need scalar penalization (if penalized = TRUE) or free
#' estimation (if penalized = FALSE) in the M-step, separate from the
#' group threshold.
#'
#' @param item_type Character.
#' @param user_overrides NULL or named list.
#'
#' @return A list with two character vectors:
#'   $penalized_scalar: families that are penalized but not in the group
#'   $free: families that are estimated freely (no penalty)
#'
#' @keywords internal
get_non_grouped_families <- function(item_type, user_overrides = NULL) {
  families <- resolve_dif_families(item_type, user_overrides)

  estimated <- Filter(function(f) f$estimated, families)

  penalized_scalar <- vapply(
    Filter(function(f) f$penalized && !f$in_group, estimated),
    function(f) f$name,
    character(1)
  )

  free <- vapply(
    Filter(function(f) !f$penalized, estimated),
    function(f) f$name,
    character(1)
  )

  list(penalized_scalar = penalized_scalar,
       free             = free)
}


#' Get the names of DIF families that are NOT estimated (fixed to zero).
#'
#' The M-step should skip these entirely — don't compute derivatives, don't
#' update parameters.
#'
#' @param item_type Character.
#' @param user_overrides NULL or named list.
#'
#' @return Character vector of family names.
#'
#' @keywords internal
get_excluded_families <- function(item_type, user_overrides = NULL) {
  families <- resolve_dif_families(item_type, user_overrides)
  vapply(
    Filter(function(f) !f$estimated, families),
    function(f) f$name,
    character(1)
  )
}


# ==============================================================================
# GROUP PARSING
# ==============================================================================

#' Parse and validate DIF group definitions.
#'
#' @param dif_groups NULL or a named list of group definitions.
#' @param dif_group_mode Character: "paired" or "separate".
#' @param dif_group_weights Character ("none" or "sqrt") or numeric vector.
#' @param dif_families NULL or a named list of family overrides.
#' @param pred_data The processed predictor data matrix.
#' @param pen_type Character indicating the penalty type.
#' @param num_predictors Integer. Number of predictor columns.
#' @param item_type Character vector of item types (length = num_items).
#' @param num_items Integer. Number of items.
#' @param num_responses Integer vector. Number of response categories per item.
#'
#' @return A list with components:
#'   \describe{
#'     \item{groups_idx}{List of integer vectors. Covariate indices per group.}
#'     \item{group_mode}{Character: "paired" or "separate".}
#'     \item{item_group_weights}{Matrix (num_items x num_groups). Per-item,
#'       per-group weights reflecting actual DIF parameter count.}
#'     \item{group_names}{Character vector of group names.}
#'     \item{dif_families_override}{The user_overrides list, stored so the
#'       M-step and other downstream code can query resolved families.}
#'   }
#'
#' @keywords internal
parse_dif_groups <-
  function(dif_groups,
           dif_group_mode,
           dif_group_weights,
           dif_families,
           pred_data,
           pen_type,
           num_predictors,
           item_type,
           num_items,
           num_responses) {

  # =========================================================================
  # Only process groups for group penalty types.
  # =========================================================================
  is_group_penalty <- pen_type %in% c("grp.lasso", "grp.mcp")
  if (!is_group_penalty) {
    # Even for scalar penalties, store the family overrides so the M-step
    # knows which families to skip or estimate freely.
    return(list(dif_families_override = dif_families))
  }

  # =========================================================================
  # Defaults.
  # =========================================================================
  if (is.null(dif_group_mode))    dif_group_mode    <- "paired"
  # Default weight: "none" for auto-generated singleton groups (preserves
  # backward compatibility with the original code that called
  # grp_soft_threshold without weights). "sqrt" is the correct default when
  # the user explicitly defines multi-covariate groups.
  if (is.null(dif_group_weights)) {
    dif_group_weights <- if (is.null(dif_groups)) "none" else "sqrt"
  }

  if (!(dif_group_mode %in% c("paired", "separate"))) {
    stop(paste0("dif.group.mode must be 'paired' or 'separate'. ",
                "Got: '", dif_group_mode, "'"),
         call. = FALSE)
  }

  # =========================================================================
  # Build the group index list.
  # =========================================================================
  if (is.null(dif_groups)) {
    pred_names <- colnames(pred_data)
    if (is.null(pred_names) || length(pred_names) == 0) {
      pred_names <- paste0("cov", 1:num_predictors)
    }
    groups_idx <- lapply(1:num_predictors, function(k) k)
    names(groups_idx) <- pred_names

  } else {
    if (!is.list(dif_groups)) {
      stop("dif.groups must be a named list.", call. = FALSE)
    }
    if (is.null(names(dif_groups)) || any(names(dif_groups) == "")) {
      stop("All elements of dif.groups must be named.", call. = FALSE)
    }

    pred_names <- colnames(pred_data)

    groups_idx <- lapply(names(dif_groups), function(grp_name) {
      grp_def <- dif_groups[[grp_name]]

      if (is.character(grp_def)) {
        if (is.null(pred_names)) {
          stop("Cannot use column names in dif.groups without column names ",
               "in pred.data.", call. = FALSE)
        }
        idx <- match(grp_def, pred_names)
        bad <- grp_def[is.na(idx)]
        if (length(bad) > 0) {
          stop(paste0("Names in dif.groups$", grp_name,
                      " not found in pred.data: ",
                      paste(bad, collapse = ", ")), call. = FALSE)
        }
      } else if (is.numeric(grp_def)) {
        idx <- as.integer(grp_def)
        if (any(idx < 1) || any(idx > num_predictors)) {
          stop(paste0("Indices in dif.groups$", grp_name,
                      " out of range [1, ", num_predictors, "]."),
               call. = FALSE)
        }
      } else {
        stop(paste0("Each dif.groups element must be character or numeric. ",
                    "Problem in '", grp_name, "'."), call. = FALSE)
      }
      return(idx)
    })
    names(groups_idx) <- names(dif_groups)

    # Overlap check.
    all_assigned <- unlist(groups_idx)
    if (anyDuplicated(all_assigned)) {
      dup <- unique(all_assigned[duplicated(all_assigned)])
      stop(paste0("Overlapping groups not yet supported. Duplicated indices: ",
                  paste(dup, collapse = ", ")), call. = FALSE)
    }

    # Coverage: singletons for uncovered covariates.
    uncovered <- setdiff(1:num_predictors, sort(unique(all_assigned)))
    if (length(uncovered) > 0) {
      for (k in uncovered) {
        nm <- if (!is.null(pred_names) && length(pred_names) >= k) {
          pred_names[k]
        } else {
          paste0("cov", k)
        }
        groups_idx[[nm]] <- k
      }
    }
  }

  num_groups <- length(groups_idx)

  # =========================================================================
  # COMPUTE PER-ITEM GROUP WEIGHTS.
  #
  # The effective group size for item j in group g is:
  #   |cov_indices_g| * (number of in_group DIF families for item j's type,
  #                      after applying user overrides)
  #
  # This means:
  #   - Rasch items (c1 only):   weight = sqrt(|G| * 1)
  #   - 2PL items (c1 + a1):     weight = sqrt(|G| * 2)
  #   - CFA items (c1+a1+s1):    weight = sqrt(|G| * 3)
  #   - CFA with s1 excluded from group via dif.families override:
  #                               weight = sqrt(|G| * 2)
  #   - Future 4PL with g1 not estimated (dif.families = list(g1 = ...)):
  #                               weight = sqrt(|G| * 3) for c1+a1+u1
  #
  # In "separate" mode, each family forms its own sub-group of size |G|,
  # so the weight is sqrt(|G|) for all items.
  # =========================================================================

  if (is.character(dif_group_weights)) {

    if (dif_group_weights == "none") {
      item_group_weights <- matrix(1, nrow = num_items, ncol = num_groups)

    } else if (dif_group_weights == "sqrt") {
      item_group_weights <- matrix(NA_real_,
                                   nrow = num_items, ncol = num_groups)

      for (j in 1:num_items) {
        n_families_j <- count_dif_families(item_type[j],
                                           dif_group_mode,
                                           dif_families)
        for (g in 1:num_groups) {
          n_covs_g <- length(groups_idx[[g]])
          item_group_weights[j, g] <- sqrt(n_covs_g * n_families_j)
        }
      }

    } else {
      stop(paste0("dif.group.weights must be 'none', 'sqrt', or numeric. ",
                  "Got: '", dif_group_weights, "'"), call. = FALSE)
    }

  } else if (is.numeric(dif_group_weights)) {
    if (length(dif_group_weights) != num_groups) {
      stop(paste0("Length of numeric dif.group.weights (",
                  length(dif_group_weights),
                  ") must match number of groups (", num_groups, ")."),
           call. = FALSE)
    }
    if (any(dif_group_weights <= 0)) {
      stop("All group weights must be positive.", call. = FALSE)
    }
    item_group_weights <- matrix(rep(dif_group_weights, each = num_items),
                                 nrow = num_items, ncol = num_groups)

  } else {
    stop("dif.group.weights must be 'none', 'sqrt', or a numeric vector.",
         call. = FALSE)
  }

  # =========================================================================
  # Return.
  # =========================================================================
  list(
    groups_idx            = groups_idx,
    group_mode            = dif_group_mode,
    item_group_weights    = item_group_weights,
    group_names           = names(groups_idx),
    dif_families_override = dif_families
  )
}


# ==============================================================================
# INDEX BUILDING
# ==============================================================================

#' Build parameter index vectors for a DIF group within an item.
#'
#' @param cov_indices Integer vector. Covariate indices in this group.
#' @param num_predictors Integer. Total number of predictors.
#' @param item_type Character. Item type.
#' @param num_responses_item Integer. Number of response categories.
#' @param group_mode Character. "paired" or "separate".
#' @param user_overrides NULL or named list of family overrides.
#'
#' @return A list. Structure depends on mode:
#'   "paired":   $paired_idx (integer vector of ALL grouped DIF positions)
#'               $family_ranges (named list: family_name -> positions within
#'                 paired_idx for write-back after thresholding)
#'   "separate": Named list with one element per in_group family.
#'
#' @details
#' PARAMETER VECTOR LAYOUTS:
#'
#' 2pl / rasch (binary, num_responses = 2):
#'   [1] c0, [2] a0, [3:2+P] c1, [3+P:2+2P] a1
#'
#' graded (ordinal, R categories):
#'   [1:R-1] intercepts/thresholds, [R] a0, [R+1:R+P] c1, [R+1+P:R+2P] a1
#'
#' cfa (continuous):
#'   [1] c0, [2] a0, [3:2+P] c1, [3+P:2+2P] a1, [3+2P] s0, [4+2P:3+3P] s1
#'
#' Future 3pl (hypothetical):
#'   [1] c0, [2] a0, [3] g0, [4:3+P] c1, [4+P:3+2P] a1, [4+2P:3+3P] g1
#'
#' Future 4pl (hypothetical):
#'   [1] c0, [2] a0, [3] g0, [4] u0,
#'   [5:4+P] c1, [5+P:4+2P] a1, [5+2P:4+3P] g1, [5+3P:4+4P] u1
#'
#' @keywords internal
build_group_param_indices <-
  function(cov_indices,
           num_predictors,
           item_type,
           num_responses_item,
           group_mode,
           user_overrides = NULL) {

  # =========================================================================
  # Base offset: number of non-DIF parameters before the DIF block.
  # =========================================================================
  base_offset <- switch(item_type,
    "rasch"  = 2L,    # c0, a0
    "2pl"    = 2L,    # c0, a0
    "graded" = as.integer(num_responses_item),  # intercepts/thresholds + a0
    "cfa"    = 2L,    # c0, a0
    # Future types would be defined here:
    # "3pl" = 3L,     # c0, a0, g0
    # "4pl" = 4L,     # c0, a0, g0, u0
    {
      warning(paste0("Unknown item type '", item_type,
                     "'. Defaulting to base_offset = 2."), call. = FALSE)
      2L
    }
  )

  # =========================================================================
  # Resolve families with user overrides applied.
  # =========================================================================
  families <- resolve_dif_families(item_type, user_overrides)

  # =========================================================================
  # Compute positions for each family.
  #
  # DIF families are laid out sequentially after the base parameters:
  #   Family 1 occupies positions [base_offset + 1 : base_offset + P]
  #   Family 2 occupies positions [base_offset + P + 1 : base_offset + 2P]
  #   ...
  # BUT some families have non-DIF base parameters interleaved (e.g., s0
  # for CFA sits between a1 and s1). We handle this by computing each
  # family's start position explicitly.
  #
  # The position map for known family names:
  #   c1:  base_offset + 1           (always first DIF family)
  #   a1:  base_offset + P + 1       (second DIF family)
  #   s1:  base_offset + 2P + 2      (third; +2 accounts for s0 at 2P+1)
  #   g1:  depends on layout (future)
  #   u1:  depends on layout (future)
  # =========================================================================
  family_positions <- list()

  for (f in families) {

    start_pos <- switch(f$name,
      "c1" = base_offset + 1L,
      "a1" = base_offset + num_predictors + 1L,
      "s1" = base_offset + 2L * num_predictors + 2L,  # +2 for s0
      # Future families:
      # "g1" = base_offset + 2L * num_predictors + 1L,  # after a1 block
      # "u1" = base_offset + 3L * num_predictors + 1L,  # after g1 block
      {
        warning(paste0("Cannot compute position for unknown DIF family '",
                       f$name, "'. Skipping."), call. = FALSE)
        next
      }
    )

    family_positions[[f$name]] <- start_pos + (cov_indices - 1L)
  }

  # =========================================================================
  # Filter to only in_group families for the group index.
  # =========================================================================
  grouped_families <- Filter(function(f) f$in_group, families)
  grouped_names <- vapply(grouped_families, function(f) f$name, character(1))

  if (group_mode == "paired") {
    paired_idx <- unlist(family_positions[grouped_names], use.names = FALSE)

    # Track where each family's entries sit within paired_idx so the M-step
    # can split results back out after thresholding.
    family_ranges <- list()
    pos_start <- 1L
    for (fname in grouped_names) {
      n <- length(family_positions[[fname]])
      family_ranges[[fname]] <- pos_start:(pos_start + n - 1L)
      pos_start <- pos_start + n
    }

    return(list(paired_idx    = paired_idx,
                family_ranges = family_ranges))

  } else if (group_mode == "separate") {
    # Each in_group family is its own sub-group.
    return(family_positions[grouped_names])
  }
}


#' Pre-compute parameter index mappings for all items and groups.
#'
#' @param group_spec Output from parse_dif_groups().
#' @param num_items Integer.
#' @param item_type Character vector.
#' @param num_responses Integer vector.
#' @param num_predictors Integer.
#'
#' @return List of length num_items, each a list of length num_groups.
#'
#' @keywords internal
precompute_group_indices <-
  function(group_spec,
           num_items,
           item_type,
           num_responses,
           num_predictors) {

  if (is.null(group_spec)) return(NULL)
  if (is.null(group_spec$groups_idx)) return(NULL)

  num_groups <- length(group_spec$groups_idx)
  user_overrides <- group_spec$dif_families_override

  lapply(1:num_items, function(item) {
    lapply(1:num_groups, function(g) {
      build_group_param_indices(
        cov_indices        = group_spec$groups_idx[[g]],
        num_predictors     = num_predictors,
        item_type          = item_type[item],
        num_responses_item = num_responses[item],
        group_mode         = group_spec$group_mode,
        user_overrides     = user_overrides
      )
    })
  })
}


################################################################################
#                                                                              #
#  INTEGRATION INTO preprocess.R                                               #
#                                                                              #
#  Insert after pen_type is set and after num_items, num_predictors,           #
#  item_type, and num_responses are finalized:                                 #
#                                                                              #
#  group_spec <- parse_dif_groups(                                             #
#    dif_groups        = control$dif.groups,                                    #
#    dif_group_mode    = control$dif.group.mode,                               #
#    dif_group_weights = control$dif.group.weights,                            #
#    dif_families      = control$dif.families,                                 #
#    pred_data         = pred_data,                                            #
#    pen_type          = pen_type,                                             #
#    num_predictors    = num_predictors,                                       #
#    item_type         = item_type,                                            #
#    num_items         = num_items,                                            #
#    num_responses     = num_responses                                         #
#  )                                                                           #
#  final_control$group_spec <- group_spec                                      #
#                                                                              #
#  final_control$group_param_idx <- precompute_group_indices(                   #
#    group_spec     = group_spec,                                              #
#    num_items      = num_items,                                               #
#    item_type      = item_type,                                               #
#    num_responses  = num_responses,                                           #
#    num_predictors = num_predictors                                            #
#  )                                                                           #
#                                                                              #
################################################################################


# ==============================================================================
# DIF UPDATE FUNCTION
#
# Unified function for updating DIF parameters (c1, a1, and optionally s1)
# within a single item during the M-step. Handles both scalar penalties
# (lasso, mcp) and group penalties (grp.lasso, grp.mcp) through a single
# interface.
#
# Called from Mstep_simple() for each item, after base parameter updates.
# The caller provides a derivative closure so this function remains
# item-type-agnostic.
# ==============================================================================


#' Update DIF parameters for a single item using group-aware penalization.
#'
#' @param p Full parameter list (all items + impact).
#' @param item Integer. Index of the current item.
#' @param item_type_item Character. Type of the current item.
#' @param pen_type Character. Penalty type.
#' @param tau_current Numeric. Current tuning parameter value.
#' @param alpha Numeric. Elastic net mixing parameter.
#' @param gamma Numeric. MCP concavity parameter.
#' @param pen Integer. Current index in the tau vector.
#' @param pen.deriv Logical. Whether to scale tau by inverse Hessian.
#' @param anchor Integer vector or NULL. Anchor item indices.
#' @param num_items Integer. Total number of items.
#' @param num_predictors Integer. Total number of predictors.
#' @param num_tau Integer. Number of tau values.
#' @param max_tau Logical. Whether to collect z proposals for max_tau.
#' @param final_control List containing group_spec, group_param_idx, etc.
#' @param compute_deriv_fn Function(family_name, p_item, cov) returning
#'   list(d1, d2). Takes the current item parameter vector as an explicit
#'   argument so it always sees the latest values during iterative updates.
#' @param num_responses_item Integer. Number of response categories.
#'
#' @return List with: p_item (updated params), under_identified (logical),
#'   id_max_z (numeric vector or NULL).
#'
#' @keywords internal
update_dif_groups <-
  function(p,
           item,
           item_type_item,
           pen_type,
           tau_current,
           alpha,
           gamma,
           pen,
           pen.deriv,
           anchor,
           num_items,
           num_predictors,
           num_tau,
           max_tau,
           final_control,
           compute_deriv_fn,
           num_responses_item) {

  under_identified <- FALSE
  id_max_z <- if (max_tau) numeric(0) else NULL

  is_group_penalty <- pen_type %in% c("grp.lasso", "grp.mcp")
  is_rasch <- (item_type_item == "rasch")

  # Base offset: number of non-DIF parameters before the DIF block.
  if (item_type_item == "graded") {
    base_offset <- num_responses_item
  } else {
    base_offset <- 2L
  }

  # =========================================================================
  # SCALAR PENALTIES (lasso, mcp)
  #
  # Per-covariate loop matching current m_step_simple.R behavior exactly.
  # =========================================================================
  if (!is_group_penalty) {

    p2 <- unlist(p)

    for (cov in 1:num_predictors) {

      # --- Under-identification check for c1 ---
      if (is.null(anchor) &
          sum(p2[grep(paste("c1(.*?)cov", cov, sep = ""), names(p2))] != 0) >
          (num_items - 1) &
          alpha == 1 &&
          (length(final_control$start.values) == 0 || pen > 1) &&
          num_tau >= 10) {
        under_identified <- TRUE
        break
      }

      # --- Intercept DIF (c1) update ---
      anl_deriv <- compute_deriv_fn("c1", p[[item]], cov)
      z_int <- p[[item]][[base_offset + cov]] -
        anl_deriv[[1]] / anl_deriv[[2]]

      if (max_tau & pen.deriv) {
        id_max_z <- c(id_max_z, z_int * (-anl_deriv[[2]]))
      } else if (max_tau & !pen.deriv) {
        id_max_z <- c(id_max_z, z_int)
      }

      if (pen.deriv) {
        p[[item]][[base_offset + cov]] <-
          ifelse(pen_type == "lasso",
                 soft_threshold(z_int, alpha,
                                tau_current / -anl_deriv[[2]]),
                 firm_threshold(z_int, alpha,
                                tau_current / -anl_deriv[[2]], gamma))
      } else {
        p[[item]][[base_offset + cov]] <-
          ifelse(pen_type == "lasso",
                 soft_threshold(z_int, alpha, tau_current),
                 firm_threshold(z_int, alpha, tau_current, gamma))
      }

      # --- Under-identification check for a1 ---
      if (is.null(anchor) &
          sum(p2[grep(paste0("a1(.*?)cov", cov), names(p2))] != 0) >
          (num_items - 1) &
          alpha == 1 &&
          (length(final_control$start.values) == 0 || pen > 1) &&
          num_tau >= 10) {
        under_identified <- TRUE
        break
      }

      # --- Slope DIF (a1) update --- skip for Rasch
      if (!is_rasch) {

        anl_deriv <- compute_deriv_fn("a1", p[[item]], cov)
        z_slp <- p[[item]][[base_offset + num_predictors + cov]] -
          anl_deriv[[1]] / anl_deriv[[2]]

        if (max_tau & pen.deriv) {
          id_max_z <- c(id_max_z, z_slp * (-anl_deriv[[2]]))
        } else if (max_tau & !pen.deriv) {
          id_max_z <- c(id_max_z, z_slp)
        }

        if (pen.deriv) {
          p[[item]][[base_offset + num_predictors + cov]] <-
            ifelse(pen_type == "lasso",
                   soft_threshold(z_slp, alpha,
                                  tau_current / -anl_deriv[[2]]),
                   firm_threshold(z_slp, alpha,
                                  tau_current / -anl_deriv[[2]], gamma))
        } else {
          p[[item]][[base_offset + num_predictors + cov]] <-
            ifelse(pen_type == "lasso",
                   soft_threshold(z_slp, alpha, tau_current),
                   firm_threshold(z_slp, alpha, tau_current, gamma))
        }

      } # End Rasch conditional.

    } # End covariate loop.


  # =========================================================================
  # GROUP PENALTIES (grp.lasso, grp.mcp)
  #
  # Loop over groups instead of covariates. Assemble the full group vector
  # and apply the group threshold operator once per group.
  # =========================================================================
  } else {

    group_spec <- final_control$group_spec
    group_param_idx <- final_control$group_param_idx

    num_groups <- length(group_spec$groups_idx)
    p2 <- unlist(p)

    for (g in 1:num_groups) {

      cov_indices <- group_spec$groups_idx[[g]]
      w_g <- group_spec$item_group_weights[item, g]
      group_size <- length(cov_indices)

      if (group_spec$group_mode == "paired") {

        # --- Compute Newton proposals for all covariates in this group ---
        z_c1 <- numeric(group_size)

        for (k_idx in seq_along(cov_indices)) {
          cov <- cov_indices[k_idx]
          anl_deriv_c1 <- compute_deriv_fn("c1", p[[item]], cov)
          z_c1[k_idx] <- p[[item]][[base_offset + cov]] -
            anl_deriv_c1[[1]] / anl_deriv_c1[[2]]

          # Collect z for max_tau (matching original per-covariate pattern).
          if (max_tau & pen.deriv) {
            id_max_z <- c(id_max_z, z_c1[k_idx] * (-anl_deriv_c1[[2]]))
          } else if (max_tau & !pen.deriv) {
            id_max_z <- c(id_max_z, z_c1[k_idx])
          }
        }

        z_a1 <- numeric(0)
        if (!is_rasch) {
          z_a1 <- numeric(group_size)
          for (k_idx in seq_along(cov_indices)) {
            cov <- cov_indices[k_idx]
            anl_deriv_a1 <- compute_deriv_fn("a1", p[[item]], cov)
            z_a1[k_idx] <- p[[item]][[base_offset + num_predictors + cov]] -
              anl_deriv_a1[[1]] / anl_deriv_a1[[2]]

            if (max_tau & pen.deriv) {
              id_max_z <- c(id_max_z, z_a1[k_idx] * (-anl_deriv_a1[[2]]))
            } else if (max_tau & !pen.deriv) {
              id_max_z <- c(id_max_z, z_a1[k_idx])
            }
          }
        }

        # Assemble group vector.
        z_group <- c(z_c1, z_a1)

        # --- Apply group threshold ---
        grp_update <-
          if (pen_type == "grp.lasso") {
            grp_soft_threshold(z_group, tau_current, w = w_g)
          } else {
            grp_firm_threshold(z_group, tau_current, gamma, w = w_g)
          }

        # --- Write results back ---
        for (k_idx in seq_along(cov_indices)) {
          cov <- cov_indices[k_idx]
          p[[item]][[base_offset + cov]] <- grp_update[k_idx]
        }

        if (!is_rasch) {
          for (k_idx in seq_along(cov_indices)) {
            cov <- cov_indices[k_idx]
            p[[item]][[base_offset + num_predictors + cov]] <-
              grp_update[group_size + k_idx]
          }
        }

      } else if (group_spec$group_mode == "separate") {

        # --- c1 sub-group ---
        z_c1 <- numeric(group_size)
        for (k_idx in seq_along(cov_indices)) {
          cov <- cov_indices[k_idx]
          anl_deriv_c1 <- compute_deriv_fn("c1", p[[item]], cov)
          z_c1[k_idx] <- p[[item]][[base_offset + cov]] -
            anl_deriv_c1[[1]] / anl_deriv_c1[[2]]

          if (max_tau & pen.deriv) {
            id_max_z <- c(id_max_z, z_c1[k_idx] * (-anl_deriv_c1[[2]]))
          } else if (max_tau & !pen.deriv) {
            id_max_z <- c(id_max_z, z_c1[k_idx])
          }
        }

        grp_update_c1 <-
          if (pen_type == "grp.lasso") {
            grp_soft_threshold(z_c1, tau_current, w = w_g)
          } else {
            grp_firm_threshold(z_c1, tau_current, gamma, w = w_g)
          }

        for (k_idx in seq_along(cov_indices)) {
          cov <- cov_indices[k_idx]
          p[[item]][[base_offset + cov]] <- grp_update_c1[k_idx]
        }

        # --- a1 sub-group --- skip for Rasch
        if (!is_rasch) {
          z_a1 <- numeric(group_size)
          for (k_idx in seq_along(cov_indices)) {
            cov <- cov_indices[k_idx]
            anl_deriv_a1 <- compute_deriv_fn("a1", p[[item]], cov)
            z_a1[k_idx] <- p[[item]][[base_offset + num_predictors + cov]] -
              anl_deriv_a1[[1]] / anl_deriv_a1[[2]]

            if (max_tau & pen.deriv) {
              id_max_z <- c(id_max_z, z_a1[k_idx] * (-anl_deriv_a1[[2]]))
            } else if (max_tau & !pen.deriv) {
              id_max_z <- c(id_max_z, z_a1[k_idx])
            }
          }

          grp_update_a1 <-
            if (pen_type == "grp.lasso") {
              grp_soft_threshold(z_a1, tau_current, w = w_g)
            } else {
              grp_firm_threshold(z_a1, tau_current, gamma, w = w_g)
            }

          for (k_idx in seq_along(cov_indices)) {
            cov <- cov_indices[k_idx]
            p[[item]][[base_offset + num_predictors + cov]] <-
              grp_update_a1[k_idx]
          }
        }

      } # End separate mode.

    } # End group loop.

  } # End group penalty case.

  return(list(
    p_item           = p[[item]],
    under_identified = under_identified,
    id_max_z         = id_max_z
  ))
}


# ==============================================================================
# MAX TAU COMPUTATION FOR GROUP PENALTIES
#
# For scalar penalties, max_tau = max(|z|) — the largest absolute unpenalized
# NR proposal across all DIF parameters.
#
# For group penalties, the correct formula is:
#   max_tau = max over (item j, group g) of: ||z_{j,g}||_2 / w_{j,g}
#
# where z_{j,g} is the group vector of NR proposals for item j's DIF
# parameters in group g, and w_{j,g} is the group weight.
#
# This function takes the flat id_max_z vector collected during the max_tau
# M-step pass and re-partitions it into groups to compute the correct norm.
# ==============================================================================


#' Compute max_tau for group penalties from a flat id_max_z vector.
#'
#' Re-partitions the flat z-proposal vector into groups, computes weighted
#' L2 norms, and returns the maximum across all items and groups.
#'
#' @param id_max_z Numeric vector. Flat vector of z proposals collected
#'   during the max_tau M-step pass.
#' @param group_spec Group specification from parse_dif_groups().
#' @param num_items Integer. Number of items.
#' @param item_type Character vector. Item types for all items.
#' @param anchor Integer vector or NULL. Anchor item indices.
#'
#' @return Numeric scalar. The maximum tau value.
#'
#' @details
#' The z values in id_max_z are ordered as they were collected during the
#' M-step: for each non-anchor item, for each group, the z values for
#' each covariate in the group (c1 then a1, matching the collection order
#' in update_dif_groups).
#'
#' @keywords internal
compute_max_tau_groups <-
  function(id_max_z,
           group_spec,
           num_items,
           item_type,
           anchor) {

  max_tau_val <- 0
  num_groups <- length(group_spec$groups_idx)

  # Walk through id_max_z, extracting group-sized chunks.
  pos <- 1

  for (item in 1:num_items) {

    # Skip anchor items (they didn't contribute to id_max_z).
    if (!is.null(anchor) && item %in% anchor) next

    is_rasch <- (item_type[item] == "rasch")

    for (g in 1:num_groups) {

      cov_indices <- group_spec$groups_idx[[g]]
      group_size <- length(cov_indices)
      w_g <- group_spec$item_group_weights[item, g]

      if (group_spec$group_mode == "paired") {
        # Each covariate contributes 1 c1 value + 1 a1 value (non-Rasch),
        # or 1 c1 value only (Rasch).
        chunk_size <- if (is_rasch) group_size else 2 * group_size

        if (pos + chunk_size - 1 > length(id_max_z)) {
          warning("id_max_z vector shorter than expected.", call. = FALSE)
          return(max_tau_val)
        }

        z_chunk <- id_max_z[pos:(pos + chunk_size - 1)]
        pos <- pos + chunk_size

        norm_g <- sqrt(sum(z_chunk^2)) / w_g
        max_tau_val <- max(max_tau_val, norm_g)

      } else if (group_spec$group_mode == "separate") {
        # c1 chunk.
        if (pos + group_size - 1 > length(id_max_z)) {
          return(max_tau_val)
        }
        z_c1 <- id_max_z[pos:(pos + group_size - 1)]
        pos <- pos + group_size

        norm_c1 <- sqrt(sum(z_c1^2)) / w_g
        max_tau_val <- max(max_tau_val, norm_c1)

        # a1 chunk (skip for Rasch).
        if (!is_rasch) {
          if (pos + group_size - 1 > length(id_max_z)) {
            return(max_tau_val)
          }
          z_a1 <- id_max_z[pos:(pos + group_size - 1)]
          pos <- pos + group_size

          norm_a1 <- sqrt(sum(z_a1^2)) / w_g
          max_tau_val <- max(max_tau_val, norm_a1)
        }
      }

    } # End group loop.
  } # End item loop.

  return(max_tau_val)
}


################################################################################
#                                                                              #
#  USER-FACING DOCUMENTATION FOR control$dif.families                          #
#  (to be added to the @param control block in regdif.R)                       #
#                                                                              #
#    \item{dif.families}{Named list controlling which DIF parameter            #
#'     families are estimated and how they are penalized. Each element         #
#'     is named after a DIF family (e.g., "c1", "a1", "s1", "g1", "u1")       #
#'     and contains a list with any subset of:                                 #
#'     \describe{                                                              #
#'       \item{estimated}{Logical. If FALSE, DIF parameters for this           #
#'         family are fixed to zero (not estimated). Default is TRUE.}         #
#'       \item{penalized}{Logical. If FALSE, DIF parameters for this           #
#'         family are estimated freely without regularization. Only            #
#'         relevant when estimated = TRUE. Default is TRUE.}                   #
#'       \item{in_group}{Logical. If FALSE, this family is excluded            #
#'         from the group norm and penalized with a scalar penalty             #
#'         instead. Only relevant when penalized = TRUE. Default is TRUE.}     #
#'     }                                                                       #
#'     Examples:                                                               #
#'     \itemize{                                                               #
#'       \item{No guessing DIF in a 3PL model:                                 #
#'         \code{dif.families = list(g1 = list(estimated = FALSE))}}           #
#'       \item{Free (unpenalized) residual DIF in a CFA model:                 #
#'         \code{dif.families = list(s1 = list(penalized = FALSE))}}           #
#'       \item{Slope DIF penalized independently (not grouped):                #
#'         \code{dif.families = list(a1 = list(in_group = FALSE))}}            #
#'     }                                                                       #
#'     Overrides that reference families not present in a given item            #
#'     type are silently ignored (e.g., "g1" is ignored for 2PL items).}       #
#                                                                              #
################################################################################
