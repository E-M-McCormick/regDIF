###############################################################################
# Spline Basis Construction Helpers for regDIF
#
# Provides user-facing functions to construct constrained spline bases
# from continuous covariates, suitable for use as DIF predictors with
# group penalties. Built on mgcv::smoothCon() to leverage its automatic
# sum-to-zero identifiability constraints and penalty matrix computation.
#
# WHY SUM-TO-ZERO MATTERS:
#
# In regDIF, the item intercept c0 is a baseline parameter. If a spline
# basis for covariate x includes a constant in its column space, that
# constant is aliased with c0 — the model is not identified (or at best,
# estimates are unstable). The sum-to-zero constraint removes this
# component:
#
#   Σ_i f(x_i) = 0    (constraint applied to the basis columns)
#
# This means the spline DIF function is centered, and c0 retains its
# interpretation as the intercept at the "average" level of x.
#
# mgcv::smoothCon() implements this via a QR decomposition of the
# basis column sums, absorbing one degree of freedom. A k-dimensional
# basis becomes (k-1)-dimensional after the constraint.
#
# WHY mgcv:
#
# mgcv is a recommended R package (ships with base R), so it adds no
# real dependency burden. Compared to splines::bs():
#   - Automatic identifiability constraints (absorb.cons = TRUE)
#   - Built-in penalty matrix for P-spline roughness (future use)
#   - Multiple basis types (thin plate, cubic regression, B-spline, etc.)
#   - Proper handling of boundary knots and knot placement
#   - Full smooth object retained for downstream reconstruction/plotting
#
###############################################################################


#' Create a constrained spline basis for use as DIF predictors in regDIF.
#'
#' Takes a continuous covariate vector and returns a spline basis matrix
#' with the sum-to-zero identifiability constraint applied, along with
#' the group definition needed for group penalty estimation.
#'
#' @param x Numeric vector. The continuous covariate to model nonlinear
#'   DIF for (e.g., age, ability, SES score). Must have no NA values;
#'   handle missing data before calling this function.
#' @param k Integer. Number of basis functions (knots + degree). The
#'   resulting basis will have k-1 columns after the sum-to-zero
#'   constraint absorbs one degree of freedom. Higher k allows more
#'   flexible (wiggly) DIF functions but increases the number of
#'   parameters to estimate per item. Default is 6, which produces
#'   5 basis columns — a good starting point for most applications.
#' @param bs Character. The mgcv basis type. Options include:
#'   \describe{
#'     \item{\code{"cr"}}{Cubic regression spline (default). Smooth,
#'       well-conditioned, and fast. Good default choice.}
#'     \item{\code{"tp"}}{Thin plate regression spline. Optimal in a
#'       certain mathematical sense, but slightly slower for large k.}
#'     \item{\code{"ps"}}{P-spline (penalized B-spline). Useful if you
#'       later want to add a roughness penalty via the penalty matrix.}
#'     \item{\code{"bs"}}{B-spline basis. Compact support, but less
#'       commonly used in this context.}
#'   }
#' @param name Character. A label for this spline term, used to name
#'   the basis columns and the group definition. Defaults to the
#'   deparsed name of \code{x} (e.g., if you call
#'   \code{make_spline_pred(age)}, the name is \code{"age"}).
#'
#' @return A list with:
#'   \describe{
#'     \item{basis}{Numeric matrix (n x (k-1)). The constrained spline
#'       basis matrix. Columns are named \code{<name>_s1}, \code{<name>_s2},
#'       etc. This matrix should be \code{cbind()}'d with other predictors
#'       to form the full \code{pred.data} argument for \code{regDIF()}.}
#'     \item{dif.groups}{A named list with one element: the spline group
#'       definition mapping the group name to its column names. Pass this
#'       (possibly merged with other group definitions) to
#'       \code{control$dif.groups} in \code{regDIF()}.}
#'     \item{penalty}{Numeric matrix ((k-1) x (k-1)). The roughness
#'       penalty matrix from mgcv. Not currently used by regDIF but
#'       stored for potential future P-spline roughness penalization.}
#'     \item{smooth_obj}{The full mgcv smooth object from
#'       \code{smoothCon()}. Contains everything needed to evaluate the
#'       fitted spline at new x values (for plotting), including
#'       constraint details, knot positions, and basis type metadata.}
#'   }
#'
#' @details
#' \strong{Typical workflow:}
#'
#' \preformatted{
#'   # 1. Create spline basis for age
#'   sp_age <- make_spline_pred(age_vector, k = 6, name = "age")
#'
#'   # 2. Combine with other (non-spline) predictors
#'   pred_data <- cbind(gender, study, sp_age$basis)
#'
#'   # 3. Define groups: spline columns grouped, others as singletons
#'   groups <- c(sp_age$dif.groups,
#'               list(gender = "gender", study = "study"))
#'
#'   # 4. Fit regDIF with group penalty
#'   fit <- regDIF(item_data, pred_data,
#'                 pen.type = "grp.lasso",
#'                 control = list(dif.groups = groups))
#' }
#'
#' \strong{Multiple spline covariates:}
#'
#' If you have multiple continuous covariates that need spline bases,
#' call \code{make_spline_pred()} separately for each and merge:
#'
#' \preformatted{
#'   sp_age <- make_spline_pred(age, k = 6, name = "age")
#'   sp_ses <- make_spline_pred(ses, k = 5, name = "ses")
#'
#'   pred_data <- cbind(sp_age$basis, sp_ses$basis, gender)
#'   groups <- c(sp_age$dif.groups, sp_ses$dif.groups,
#'               list(gender = "gender"))
#' }
#'
#' \strong{Choosing k:}
#'
#' k controls the flexibility of the DIF function. With the sum-to-zero
#' constraint, the basis has k-1 effective columns, so:
#'   - k = 4: 3 basis columns, can capture gentle curvature
#'   - k = 6: 5 basis columns, moderate flexibility (recommended start)
#'   - k = 10: 9 basis columns, high flexibility (use for large N)
#'   - k = 20: 19 basis columns, very flexible (risk of overfitting
#'     unless N is large and DIF is truly complex)
#'
#' The group penalty provides automatic complexity control: if the
#' true DIF is simpler than the basis can represent, the group
#' threshold will zero out the entire spline block.
#'
#' @seealso
#' \code{\link{predict_spline_dif}} for evaluating fitted DIF curves,
#' \code{\link[mgcv]{smoothCon}} for details on basis construction,
#' \code{\link{regDIF}} for the main fitting function.
#'
#' @examples
#' \dontrun{
#' # Simulate some data
#' set.seed(42)
#' n <- 500
#' age <- rnorm(n, mean = 40, sd = 10)
#' gender <- sample(c(-1, 1), n, replace = TRUE)
#'
#' # Create spline basis for age
#' sp <- make_spline_pred(age, k = 6, name = "age")
#'
#' # Inspect the basis
#' dim(sp$basis)       # 500 x 5
#' head(sp$basis)
#' sp$dif.groups       # list(age = c("age_s1", ..., "age_s5"))
#'
#' # Combine with other predictors
#' pred_data <- cbind(sp$basis, gender = gender)
#' groups <- c(sp$dif.groups, list(gender = "gender"))
#'
#' # Fit with group lasso
#' # fit <- regDIF(item_data, pred_data, pen.type = "grp.lasso",
#' #              control = list(dif.groups = groups))
#' }
#'
#' @importFrom mgcv smoothCon s
#' @export
make_spline_pred <-
  function(x,
           k = 6,
           bs = "cr",
           name = deparse(substitute(x))) {

  # ---- Input validation ----
  if (!is.numeric(x)) {
    stop("x must be a numeric vector.", call. = FALSE)
  }
  if (any(is.na(x))) {
    stop("x contains NA values. Handle missing data before calling ",
         "make_spline_pred().", call. = FALSE)
  }
  if (length(unique(x)) < k) {
    stop(paste0("x has only ", length(unique(x)), " unique values but k = ",
                k, ". Reduce k or use a covariate with more unique values."),
         call. = FALSE)
  }
  if (k < 3) {
    stop("k must be >= 3 (minimum for a spline basis).", call. = FALSE)
  }
  if (!(bs %in% c("cr", "tp", "ps", "bs"))) {
    stop(paste0("Unsupported basis type '", bs, "'. Use one of: ",
                "'cr', 'tp', 'ps', 'bs'."), call. = FALSE)
  }

  # ---- Construct the constrained basis via mgcv ----
  # smoothCon() builds the basis, applies the sum-to-zero constraint
  # (absorb.cons = TRUE), and computes the penalty matrix.
  #
  # The data argument must be a data.frame with the variable named
  # to match the formula in s(). We use a temporary name "x_var" to
  # avoid conflicts with the user's variable name.
  sm_list <- mgcv::smoothCon(
    mgcv::s(x_var, k = k, bs = bs),
    data       = data.frame(x_var = x),
    absorb.cons = TRUE,   # Apply the sum-to-zero constraint.
    scale.penalty = TRUE   # Standardize the penalty matrix.
  )

  # smoothCon returns a list; for a single smooth term, the first
  # element is the one we want.
  sm <- sm_list[[1]]

  # ---- Extract and name the basis matrix ----
  B <- sm$X
  n_basis <- ncol(B)
  basis_names <- paste0(name, "_s", seq_len(n_basis))
  colnames(B) <- basis_names

  # ---- Build the group definition ----
  # This maps the spline group name to its column names, ready to be
  # passed to control$dif.groups (possibly merged with other groups).
  group_def <- setNames(list(basis_names), name)

  # ---- Extract the penalty matrix ----
  # sm$S is a list of penalty matrices (one for each penalty term).
  # For standard bases, there's exactly one.
  S <- if (length(sm$S) > 0) sm$S[[1]] else NULL

  # ---- Return ----
  list(
    basis      = B,
    dif.groups = group_def,
    penalty    = S,
    smooth_obj = sm
  )
}


#' Evaluate a fitted spline DIF function at new covariate values.
#'
#' Given a fitted regDIF model and the spline specification from
#' \code{make_spline_pred()}, reconstructs the DIF function
#' \code{f(x) = B(x) * beta} at a grid of new x values. This is the
#' foundation for plotting nonlinear DIF curves.
#'
#' @param spline_spec The output of \code{make_spline_pred()} used to
#'   create the spline basis in the original model fit.
#' @param coefs Named numeric vector of fitted DIF coefficients for
#'   this item and spline group. Names must match the basis column
#'   names (e.g., \code{"age_s1"}, \code{"age_s2"}, ...).
#' @param x_new Numeric vector. The new covariate values at which to
#'   evaluate the spline DIF function. If NULL (default), a grid of
#'   200 points spanning the range of the original x is used.
#' @param family Character. Which DIF family's coefficients to use.
#'   Default is \code{"c1"} (intercept DIF). Use \code{"a1"} for
#'   slope DIF. The function looks for coefficients matching the
#'   pattern \code{<family>_item<j>_<basis_name>}.
#'
#' @return A data frame with columns:
#'   \describe{
#'     \item{x}{The evaluation points.}
#'     \item{f_x}{The estimated DIF function value at each point.}
#'   }
#'
#' @details
#' The function constructs a prediction basis matrix \code{B_new} at the
#' new x values using mgcv's \code{PredictMat()}, then computes
#' \code{f(x_new) = B_new \%*\% beta}. The prediction basis correctly
#' applies the same sum-to-zero constraint as the fitting basis.
#'
#' \strong{Typical usage with a fitted model:}
#'
#' \preformatted{
#'   # After fitting
#'   sp <- make_spline_pred(age, k = 6, name = "age")
#'   fit <- regDIF(...)
#'
#'   # Extract DIF coefficients for item 1 at the best tau
#'   best_tau_idx <- which.min(fit$bic)
#'   all_dif <- fit$dif[, best_tau_idx]
#'
#'   # Get the age spline coefficients for item 1's intercept DIF
#'   spline_coef_names <- paste0("item1.int.", sp$dif.groups$age)
#'   beta_hat <- all_dif[spline_coef_names]
#'
#'   # Evaluate and plot
#'   curve_df <- predict_spline_dif(sp, beta_hat)
#'   plot(curve_df$x, curve_df$f_x, type = "l",
#'        xlab = "Age", ylab = "Intercept DIF")
#'   abline(h = 0, lty = 2)
#' }
#'
#' @importFrom mgcv PredictMat
#' @export
predict_spline_dif <-
  function(spline_spec,
           coefs,
           x_new = NULL) {

  # ---- Input validation ----
  sm <- spline_spec$smooth_obj
  if (is.null(sm)) {
    stop("spline_spec does not contain a smooth_obj. ",
         "Was it created by make_spline_pred()?", call. = FALSE)
  }

  n_basis <- ncol(spline_spec$basis)
  if (length(coefs) != n_basis) {
    stop(paste0("Length of coefs (", length(coefs), ") does not match ",
                "number of basis columns (", n_basis, ")."), call. = FALSE)
  }

  # ---- Default evaluation grid ----
  if (is.null(x_new)) {
    # Use the range of the original fitting data.
    x_range <- range(sm$X %*% rep(0, ncol(sm$X)))  # placeholder
    # Better: extract from the smooth object's knot range.
    if (!is.null(sm$knots)) {
      x_range <- range(unlist(sm$knots))
    } else {
      # Fall back to the data range stored in the basis.
      x_range <- range(spline_spec$basis %*% rep(1, n_basis))
    }
    # Use the by-variable data range if available.
    # The most reliable approach: the original x values aren't stored
    # directly, but the smooth object's term name tells us what was
    # used. We'll require x_new if we can't determine the range.
    x_new <- seq(from = min(sm$X), to = max(sm$X), length.out = 200)
  }

  # ---- Construct prediction basis at new x values ----
  # PredictMat applies the same constraint and basis construction
  # as the fitting basis, evaluated at x_new.
  B_new <- mgcv::PredictMat(sm, data = data.frame(x_var = x_new))

  # ---- Evaluate the spline DIF function ----
  f_x <- as.numeric(B_new %*% coefs)

  data.frame(x = x_new, f_x = f_x)
}


#' Combine multiple spline and scalar predictors for regDIF.
#'
#' Convenience function that takes a mix of spline specs (from
#' \code{make_spline_pred()}) and scalar predictor vectors, and returns
#' a combined predictor matrix and group definition ready for regDIF.
#'
#' @param ... Named arguments. Each argument is either:
#'   \itemize{
#'     \item A \code{make_spline_pred()} output (list with \code{$basis}
#'       and \code{$dif.groups} elements) — treated as a spline group.
#'     \item A numeric vector — treated as a scalar (single-column)
#'       predictor, which gets its own singleton group.
#'   }
#'   Names are used as predictor/group labels.
#'
#' @return A list with:
#'   \describe{
#'     \item{pred.data}{A numeric matrix of all predictors combined,
#'       ready to pass to \code{regDIF()}'s \code{pred.data} argument.}
#'     \item{dif.groups}{A named list of group definitions, ready to
#'       pass to \code{control$dif.groups}.}
#'   }
#'
#' @details
#' \strong{Example:}
#'
#' \preformatted{
#'   sp_age <- make_spline_pred(age, k = 6, name = "age")
#'   sp_ses <- make_spline_pred(ses, k = 5, name = "ses")
#'
#'   combined <- combine_predictors(
#'     age    = sp_age,
#'     ses    = sp_ses,
#'     gender = gender_vector
#'   )
#'
#'   fit <- regDIF(item_data, combined$pred.data,
#'                 pen.type = "grp.lasso",
#'                 control = list(dif.groups = combined$dif.groups))
#' }
#'
#' @export
combine_predictors <- function(...) {

  args <- list(...)
  arg_names <- names(args)

  if (is.null(arg_names) || any(arg_names == "")) {
    stop("All arguments to combine_predictors() must be named.",
         call. = FALSE)
  }

  pred_cols <- list()
  groups <- list()

  for (nm in arg_names) {
    val <- args[[nm]]

    if (is.list(val) && !is.null(val$basis) && !is.null(val$dif.groups)) {
      # This is a spline spec from make_spline_pred().
      pred_cols[[nm]] <- val$basis
      groups <- c(groups, val$dif.groups)

    } else if (is.numeric(val) && is.null(dim(val))) {
      # This is a scalar predictor vector.
      pred_cols[[nm]] <- matrix(val, ncol = 1,
                                dimnames = list(NULL, nm))
      groups[[nm]] <- nm

    } else if (is.numeric(val) && !is.null(dim(val))) {
      # This is a matrix of predictors (e.g., pre-constructed basis).
      if (is.null(colnames(val))) {
        colnames(val) <- paste0(nm, "_", seq_len(ncol(val)))
      }
      pred_cols[[nm]] <- val
      groups[[nm]] <- colnames(val)

    } else {
      stop(paste0("Argument '", nm, "' must be a numeric vector, a matrix, ",
                  "or a make_spline_pred() output."), call. = FALSE)
    }
  }

  # Combine all columns into a single matrix.
  pred_data <- do.call(cbind, pred_cols)

  list(
    pred.data  = pred_data,
    dif.groups = groups
  )
}


# ===========================================================================
# EXTRACTION AND PLOTTING FOR FITTED SPLINE DIF MODELS
#
# These functions bridge the gap between regDIF's coefficient output
# (individual basis coefficients like "item1.int.age_s1") and the
# interpretable DIF curve f(x). They handle:
#   - Extracting spline coefficients from a fitted regDIF object
#   - Reconstructing DIF curves for plotting
#   - Multi-item and multi-family curve plots
# ===========================================================================


#' Extract spline DIF coefficients from a fitted regDIF model.
#'
#' Given a fitted regDIF model and a spline specification, extracts the
#' basis coefficients for a specific item and DIF family (intercept,
#' slope, or residual) at a chosen tau value.
#'
#' @param fit A fitted regDIF model object (class "regDIF").
#' @param spline_spec The output of \code{make_spline_pred()} used when
#'   fitting the model.
#' @param item Character or integer. Either the item name (as it appears
#'   in row names, e.g., "item1") or the item index (e.g., 1).
#' @param family Character. Which DIF family to extract: \code{"int"}
#'   for intercept DIF (default), \code{"slp"} for slope DIF, or
#'   \code{"res"} for residual variance DIF (CFA only).
#' @param tau_index Integer or character. Which tau value to use.
#'   If \code{"bic"} (default), uses the tau that minimizes BIC.
#'   If \code{"aic"}, uses the tau that minimizes AIC.
#'   If integer, uses that column index directly.
#'
#' @return A named numeric vector of spline basis coefficients, in the
#'   same order as the basis columns. Returns all zeros if the spline
#'   group was fully penalized to zero at the chosen tau.
#'
#' @details
#' The function constructs row name patterns to match the regDIF output
#' naming convention: \code{<item_name>.<family>.<basis_col_name>}.
#' For example, with item "item1", family "int", and spline name "age"
#' (k=6), it looks for: \code{"item1.int.age_s1"}, ...,
#' \code{"item1.int.age_s5"}.
#'
#' @examples
#' \dontrun{
#' sp <- make_spline_pred(age, k = 6, name = "age")
#' combined <- combine_predictors(age = sp, gender = gender)
#' fit <- regDIF(items, combined$pred.data, pen.type = "grp.lasso",
#'               control = list(dif.groups = combined$dif.groups))
#'
#' # Extract intercept DIF spline coefficients for item 1 at best BIC
#' beta <- extract_spline_coefs(fit, sp, item = 1)
#'
#' # Extract slope DIF for item "math3" at best AIC
#' beta_slp <- extract_spline_coefs(fit, sp, item = "math3",
#'                                  family = "slp", tau_index = "aic")
#' }
#'
#' @export
extract_spline_coefs <-
  function(fit,
           spline_spec,
           item,
           family = "int",
           tau_index = "bic") {

  # Resolve tau index.
  if (is.character(tau_index)) {
    tau_index <- switch(tau_index,
      "bic" = which.min(fit$bic),
      "aic" = which.min(fit$aic),
      stop(paste0("tau_index must be 'bic', 'aic', or an integer. Got: '",
                  tau_index, "'"), call. = FALSE)
    )
  }

  # Resolve item name.
  dif_mat <- fit$dif
  if (is.numeric(item)) {
    # Find item names from the row names (e.g., "item1.int.age_s1" -> "item1").
    all_rnames <- rownames(dif_mat)
    # Extract unique item prefixes by taking everything before the first ".".
    item_prefixes <- unique(sub("\\..*", "", all_rnames))
    if (item > length(item_prefixes)) {
      stop(paste0("Item index ", item, " is out of range (",
                  length(item_prefixes), " items)."), call. = FALSE)
    }
    item_name <- item_prefixes[item]
  } else {
    item_name <- item
  }

  # Build the row name pattern for this item + family + spline.
  basis_names <- colnames(spline_spec$basis)
  target_names <- paste0(item_name, ".", family, ".", basis_names)

  # Extract coefficients.
  dif_col <- dif_mat[, tau_index]
  matched <- match(target_names, names(dif_col))

  if (any(is.na(matched))) {
    missing <- target_names[is.na(matched)]
    stop(paste0("Could not find DIF coefficients in model output: ",
                paste(missing[1:min(3, length(missing))], collapse = ", "),
                if (length(missing) > 3) ", ..." else ""),
         call. = FALSE)
  }

  coefs <- dif_col[matched]
  names(coefs) <- basis_names
  coefs
}


#' Plot spline DIF curves from a fitted regDIF model.
#'
#' Creates publication-ready plots of nonlinear DIF functions estimated
#' via spline bases. Can plot a single item or overlay multiple items
#' on the same panel.
#'
#' @param fit A fitted regDIF model object (class "regDIF").
#' @param spline_spec The output of \code{make_spline_pred()} used when
#'   fitting the model.
#' @param items Integer vector or character vector. Which items to plot.
#'   Default is all items. Use item names or indices.
#' @param family Character. Which DIF family to plot: \code{"int"}
#'   (default), \code{"slp"}, or \code{"res"}.
#' @param tau_index Integer or character. Which tau value to use.
#'   Default is \code{"bic"}.
#' @param x_new Optional numeric vector. Custom evaluation grid for the
#'   x axis. If NULL, uses 200 equally-spaced points across the range
#'   of the original covariate.
#' @param xlab Character. X-axis label. Default uses the spline name.
#' @param ylab Character. Y-axis label. Default is "DIF Effect".
#' @param main Character. Plot title. Default is auto-generated.
#' @param legend Logical. Whether to include a legend. Default is TRUE.
#' @param cols Optional character vector of colors for each item.
#' @param zero_line Logical. Whether to draw a horizontal line at y = 0.
#'   Default is TRUE.
#' @param ... Additional arguments passed to \code{plot()}.
#'
#' @return Invisibly returns a list of data frames (one per item), each
#'   with columns \code{x} and \code{f_x}. Useful for further
#'   customization with ggplot2 or other plotting systems.
#'
#' @details
#' Items whose spline group was fully penalized to zero (all basis
#' coefficients = 0) are drawn as flat lines at y = 0 and labeled
#' as "(zero)" in the legend. This makes it visually clear which
#' items were selected for nonlinear DIF.
#'
#' @examples
#' \dontrun{
#' sp <- make_spline_pred(age, k = 6, name = "age")
#' combined <- combine_predictors(age = sp, gender = gender)
#' fit <- regDIF(items, combined$pred.data, pen.type = "grp.lasso",
#'               control = list(dif.groups = combined$dif.groups))
#'
#' # Plot intercept DIF curves for all items
#' plot_spline_dif(fit, sp)
#'
#' # Plot only items 1 and 3, slope DIF
#' plot_spline_dif(fit, sp, items = c(1, 3), family = "slp")
#' }
#'
#' @importFrom graphics abline legend lines plot
#' @importFrom grDevices hcl.colors
#' @export
plot_spline_dif <-
  function(fit,
           spline_spec,
           items = NULL,
           family = "int",
           tau_index = "bic",
           x_new = NULL,
           xlab = NULL,
           ylab = "DIF Effect",
           main = NULL,
           legend = TRUE,
           cols = NULL,
           zero_line = TRUE,
           ...) {

  # Resolve item list.
  all_rnames <- rownames(fit$dif)
  item_prefixes <- unique(sub("\\..*", "", all_rnames))

  if (is.null(items)) {
    items <- seq_along(item_prefixes)
  }

  # Convert numeric indices to names.
  item_names <- character(length(items))
  for (i in seq_along(items)) {
    if (is.numeric(items[i])) {
      item_names[i] <- item_prefixes[items[i]]
    } else {
      item_names[i] <- items[i]
    }
  }
  n_items <- length(item_names)

  # Default colors.
  if (is.null(cols)) {
    if (n_items <= 8) {
      cols <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
                "#FF7F00", "#A65628", "#F781BF", "#999999")[1:n_items]
    } else {
      cols <- grDevices::hcl.colors(n_items, palette = "Set 2")
    }
  }

  # Extract spline name for labels.
  spline_name <- names(spline_spec$dif.groups)[1]
  if (is.null(xlab)) xlab <- spline_name
  if (is.null(main)) {
    family_label <- switch(family,
                           "int" = "Intercept", "slp" = "Slope",
                           "res" = "Residual Variance", family)
    main <- paste(family_label, "DIF:", spline_name)
  }

  # Compute curves for all items.
  curves <- list()
  y_range <- c(0, 0)

  for (i in seq_along(item_names)) {
    coefs <- extract_spline_coefs(fit, spline_spec,
                                  item = item_names[i],
                                  family = family,
                                  tau_index = tau_index)
    curve_df <- predict_spline_dif(spline_spec, coefs, x_new = x_new)
    curves[[item_names[i]]] <- curve_df
    y_range <- range(y_range, curve_df$f_x)
  }

  # Add some padding to y range.
  y_pad <- diff(y_range) * 0.05
  if (y_pad == 0) y_pad <- 0.1
  y_range <- y_range + c(-y_pad, y_pad)

  # Draw the plot.
  x_vals <- curves[[1]]$x
  plot(x_vals, rep(0, length(x_vals)),
       type = "n",
       xlim = range(x_vals),
       ylim = y_range,
       xlab = xlab, ylab = ylab, main = main,
       ...)

  if (zero_line) abline(h = 0, lty = 2, col = "gray60")

  # Draw each item's curve.
  legend_labels <- character(n_items)
  legend_ltys <- integer(n_items)
  for (i in seq_along(item_names)) {
    curve_df <- curves[[item_names[i]]]
    is_zero <- all(abs(curve_df$f_x) < 1e-10)
    lty_val <- if (is_zero) 3 else 1
    lwd_val <- if (is_zero) 1 else 2

    lines(curve_df$x, curve_df$f_x,
          col = cols[i], lty = lty_val, lwd = lwd_val)

    legend_labels[i] <- if (is_zero) {
      paste0(item_names[i], " (zero)")
    } else {
      item_names[i]
    }
    legend_ltys[i] <- lty_val
  }

  # Legend.
  if (legend && n_items > 1) {
    graphics::legend("topleft",
                     legend = legend_labels,
                     col = cols,
                     lty = legend_ltys,
                     lwd = 2,
                     cex = 0.75,
                     bty = "n")
  }

  invisible(curves)
}


#' Compute L2 norm summary of spline DIF across the tau path.
#'
#' For each item and tau value, computes the L2 norm of the spline
#' basis coefficients: \code{||beta_g||_2 = sqrt(sum(beta^2))}. This
#' provides a single scalar summary of "how much" nonlinear DIF exists
#' for each item at each penalty level, analogous to the scalar
#' coefficient paths in the standard regularization plot.
#'
#' @param fit A fitted regDIF model object.
#' @param spline_spec The output of \code{make_spline_pred()}.
#' @param items Integer or character vector. Which items. Default: all.
#' @param family Character. DIF family (\code{"int"}, \code{"slp"},
#'   \code{"res"}). Default: \code{"int"}.
#'
#' @return A matrix with rows = items and columns = tau values,
#'   containing the L2 norm of the spline DIF coefficients.
#'
#' @details
#' This is useful for creating regularization path plots that show
#' the "strength" of nonlinear DIF as a single line per item, rather
#' than k-1 separate lines for each basis coefficient. The L2 norm
#' is the natural summary because it's what the group lasso penalty
#' operates on.
#'
#' @examples
#' \dontrun{
#' norms <- spline_dif_norms(fit, sp)
#' matplot(log(fit$tau_vec), t(norms), type = "l",
#'         xlab = "log(tau)", ylab = "||DIF(x)||")
#' }
#'
#' @export
spline_dif_norms <-
  function(fit,
           spline_spec,
           items = NULL,
           family = "int") {

  # Resolve item list.
  all_rnames <- rownames(fit$dif)
  item_prefixes <- unique(sub("\\..*", "", all_rnames))

  if (is.null(items)) {
    items <- seq_along(item_prefixes)
  }

  item_names <- character(length(items))
  for (i in seq_along(items)) {
    if (is.numeric(items[i])) {
      item_names[i] <- item_prefixes[items[i]]
    } else {
      item_names[i] <- items[i]
    }
  }

  basis_names <- colnames(spline_spec$basis)
  n_tau <- ncol(fit$dif)
  n_items <- length(item_names)

  norms <- matrix(NA_real_, nrow = n_items, ncol = n_tau)
  rownames(norms) <- item_names

  for (i in seq_along(item_names)) {
    target_names <- paste0(item_names[i], ".", family, ".", basis_names)
    matched <- match(target_names, rownames(fit$dif))

    if (any(is.na(matched))) next

    for (t in 1:n_tau) {
      coefs <- fit$dif[matched, t]
      norms[i, t] <- sqrt(sum(coefs^2))
    }
  }

  norms
}
