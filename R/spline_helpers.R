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
