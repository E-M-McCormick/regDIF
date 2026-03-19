###############################################################################
# Thresholding Functions                                                      #
#                                                                             #
# Penalty-specific thresholding operators used to regularize DIF parameters   #
# during the M-step of the penalized EM algorithm. Each function implements   #
# a proximal operator that shrinks or zeros out DIF effects based on the      #
# current tuning parameter (tau).                                             #
#                                                                             #
# Supported penalties:                                                        #
#   - LASSO (soft thresholding): continuous shrinkage toward zero             #
#   - MCP (firm thresholding): reduced bias for large effects via tapering    #
#   - Group LASSO: joint selection of intercept + slope DIF per covariate     #
#   - Group MCP: group version with MCP bias correction                       #
###############################################################################


# ==============================================================================
# SCALAR PENALTIES
# ==============================================================================

#' Lasso (soft-thresholding) penalty for a single scalar parameter.
#'
#' Implements the elastic net soft-thresholding operator:
#'   S(z; alpha, tau) = sign(z) * max(|z/(1 + tau*(1-alpha))| - tau*alpha/(1 + tau*(1-alpha)), 0)
#'
#' When alpha = 1 (pure lasso), this simplifies to:
#'   S(z; tau) = sign(z) * max(|z| - tau, 0)
#'
#' When alpha = 0 (pure ridge), there is no thresholding — just shrinkage:
#'   S(z; tau) = z / (1 + tau)
#'
#' @param z Numeric scalar. The Newton-Raphson proposal (unpenalized update).
#' @param alpha Numeric scalar in [0, 1]. Elastic net mixing parameter.
#'   alpha = 1 is lasso, alpha = 0 is ridge.
#' @param tau Numeric scalar >= 0. The penalty tuning parameter.
#'
#' @return Numeric scalar. The penalized parameter estimate.
#'
#' @keywords internal
#'
# Elastic net soft-thresholding operator (LASSO penalty).
#
# Applies the proximal operator for the elastic net penalty, which combines
# L1 (LASSO) and L2 (ridge) penalties. When alpha = 1, this reduces to
# pure LASSO; when alpha = 0, it reduces to ridge regression.
#
# The update rule is:
#   p_new = sign(z) * max(|z / (1 + tau*(1-alpha))| - tau*alpha / (1 + tau*(1-alpha)), 0)
#
# @param z Numeric scalar: unpenalized Newton-Raphson update for the parameter.
# @param alpha Numeric in [0,1]: elastic net mixing parameter (1 = LASSO, 0 = ridge).
# @param tau Numeric >= 0: penalty tuning parameter controlling shrinkage strength.
# @return Numeric scalar: the penalized (shrunken) parameter estimate.
soft_threshold <-
  function(z,
           alpha,
           tau) {

  # -----------------------------------------------------------------------
  # The denominator (1 + tau*(1-alpha)) handles the ridge (L2) component.
  # The numerator subtraction (tau*alpha) handles the lasso (L1) component.
  # Together they implement the elastic net proximal operator.
  # -----------------------------------------------------------------------
    p_new <- sign(z) * max(abs(z / (1 + tau * (1 - alpha))) -
                             (tau * alpha) / (1 + tau * (1 - alpha)), 0)

  return(p_new)

  }

#' MCP (firm-thresholding) penalty for a single scalar parameter.
#'
#' Implements the minimax concave penalty thresholding operator. MCP reduces
#' the bias of lasso by "tapering" the penalty: for small parameters, it acts
#' like a scaled lasso; for large parameters (|z| > gamma * tau), it applies
#' no shrinkage at all — just the ridge component.
#'
#' The gamma parameter controls how quickly the penalty tapers off:
#'   - gamma -> Inf: MCP approaches lasso
#'   - gamma = 1: undefined (must be > 1)
#'   - Small gamma (e.g., 2): fast tapering, less bias, but possibly unstable
#'   - Large gamma (e.g., 10): slow tapering, more bias, more stable
#'
#' @param z Numeric scalar. The Newton-Raphson proposal.
#' @param alpha Numeric scalar in [0, 1]. Elastic net mixing parameter.
#' @param tau Numeric scalar >= 0. The penalty tuning parameter.
#' @param gamma Numeric scalar > 1. The MCP concavity parameter.
#'
#' @return Numeric scalar. The penalized parameter estimate.
#'
#' @keywords internal
#'
firm_threshold <-
  function(z,
           alpha,
           tau,
           gamma) {
    # -----------------------------------------------------------------------
    # Two regimes:
    #   (a) |z/(1+tau*(1-alpha))| <= gamma*tau: apply scaled soft-threshold
    #       The scaling factor gamma/(gamma-1) reduces the shrinkage bias
    #       relative to pure lasso.
    #   (b) |z/(1+tau*(1-alpha))| > gamma*tau: no L1 shrinkage at all,
    #       only the ridge component z/(1+tau*(1-alpha)) remains.
    # -----------------------------------------------------------------------

    if (abs(z / (1 + tau * (1 - alpha))) <= gamma * tau) {
      p_new <- (gamma / (gamma - 1)) * soft_threshold(z, alpha, tau)
    } else {
      p_new <- z / (1 + tau * (1 - alpha))
    }

    return(p_new)
  }



# ==============================================================================
# GROUP PENALTIES (generalized to arbitrary group sizes + optional weights)
# ==============================================================================

#' Group lasso (soft-thresholding) penalty for a vector of parameters.
#'
#' Implements the proximal operator for the group lasso penalty. The group
#' lasso penalizes the L2 norm of a group of coefficients, encouraging the
#' entire group to be set to zero (or not) together.
#'
#' The proximal operator is:
#'   prox(z; tau, w) =
#'     if ||z||_2 <= tau*w:  0  (entire group shrunk to zero)
#'     else:                 (1 - tau*w / ||z||_2) * z  (shrink toward zero)
#'
#' This is the multivariate analog of the scalar soft-thresholding operator.
#' The key property is that when the L2 norm of the proposal vector z is small
#' enough (below the threshold tau*w), ALL elements are set to zero
#' simultaneously. Otherwise, the entire vector is scaled down uniformly.
#'
#' CHANGES FROM ORIGINAL:
#'   - Now returns rep(0, length(z)) instead of c(0,0) when shrunk to zero.
#'     This allows groups of any size, not just intercept-slope pairs.
#'   - Added weight parameter w (default = 1) for group-size-adjusted penalties.
#'     Standard practice: w = sqrt(|group|) so larger groups aren't over-penalized.
#'   - Both changes are backward compatible: calling with a 2-vector and w = 1
#'     gives exactly the same result as the original function.
#'
#' @param z Numeric vector of any length. The Newton-Raphson proposals for all
#'   parameters in this group. Could be length 2 (original c1/a1 pair), length 1
#'   (Rasch item with intercept DIF only), or length > 2 (spline basis group).
#' @param tau Numeric scalar >= 0. The penalty tuning parameter.
#' @param w Numeric scalar > 0. Group weight for scaling the penalty.
#'   Default is 1 (no rescaling). Typical choice: sqrt(length(z)) so that
#'   the penalty is proportional to the square root of group size.
#'
#' @return Numeric vector of same length as z. The penalized parameter estimates.
#'   Either all zeros (group killed) or a uniformly shrunk version of z.
#'
#' @keywords internal
#'

# Group LASSO soft-thresholding operator.
#
# Applies the proximal operator for the group LASSO penalty, which selects
# intercept and slope DIF effects jointly for each covariate. Uses the
# L2 norm of the parameter vector to determine whether the entire group
# is shrunk to zero or scaled proportionally.
#
# @param z Numeric vector of length 2: unpenalized updates for the
#   intercept and slope DIF parameters (grouped together per covariate).
# @param tau Numeric >= 0: penalty tuning parameter.
# @return Numeric vector of length 2: the penalized group parameter estimates.
#   Returns c(0, 0) if the L2 norm of z is below the threshold.

grp_soft_threshold <-
  function(z,
           tau,
           w = 1) {

    # -----------------------------------------------------------------------
    # Step 1: Compute the L2 (Euclidean) norm of the proposal vector.
    #         This is the "size" of the group's effect in aggregate.
    # -----------------------------------------------------------------------
    l2_norm_z <- sqrt(sum(z**2))

    # -----------------------------------------------------------------------
    # Step 2: Scale tau by the group weight. Larger groups (more parameters)
    #         get a proportionally larger "budget" before being zeroed out.
    #         With the default w = 1, this is equivalent to the original code.
    # -----------------------------------------------------------------------
    tau_w <- tau * w

    # -----------------------------------------------------------------------
    # Step 3: Apply the group soft-thresholding rule.
    #
    #   If ||z||_2 <= tau*w: the entire group's effect is too small to survive
    #     the penalty, so set ALL coefficients to zero. This is where the old
    #     code returned c(0,0) — now we return rep(0, length(z)) so this works
    #     for groups of any size.
    #
    #   If ||z||_2 > tau*w: shrink the vector toward zero by the factor
    #     (1 - tau*w / ||z||_2). Note this factor is always in (0, 1),
    #     so the direction of z is preserved but its magnitude decreases.
    #     This is exactly analogous to scalar soft-thresholding but in
    #     multiple dimensions.
    # -----------------------------------------------------------------------
    if (l2_norm_z <= tau_w){
      p_new <- rep(0, length(z))
    } else {
      p_new <- (1 - tau_w / l2_norm_z) * z
    }

  }

#' Group MCP (firm-thresholding) penalty for a vector of parameters.
#'
#' Implements the proximal operator for the group MCP penalty. Like scalar MCP,
#' this reduces the shrinkage bias of the group lasso by tapering the penalty
#' for large effects.
#'
#' Two regimes:
#'   ||z||_2 <= gamma * tau * w:
#'     Apply group soft-threshold with a gamma/(gamma-1) scaling factor.
#'     This reduces bias relative to group lasso.
#'   ||z||_2 > gamma * tau * w:
#'     No shrinkage at all — return z as-is. The effect is "large enough"
#'     that MCP considers it a genuine signal and applies no penalty.
#'
#' CHANGES FROM ORIGINAL:
#'   - Added weight parameter w (default = 1) for group-size-adjusted penalties.
#'   - Inherits the arbitrary-length fix from grp_soft_threshold (which it calls
#'     internally), so this automatically works for any group size.
#'
#' @param z Numeric vector of any length. The Newton-Raphson proposals.
#' @param tau Numeric scalar >= 0. The penalty tuning parameter.
#' @param gamma Numeric scalar > 1. The MCP concavity parameter.
#' @param w Numeric scalar > 0. Group weight. Default is 1.
#'
#' @return Numeric vector of same length as z. The penalized parameter estimates.
#'
#' @keywords internal
grp_firm_threshold <-
  function(z,
           tau,
           gamma,
           w = 1) {

    # -----------------------------------------------------------------------
    # Step 1: Compute L2 norm to determine which MCP regime we're in.
    # -----------------------------------------------------------------------
    l2_norm_z <- sqrt(sum(z^2))

    # -----------------------------------------------------------------------
    # Step 2: Scale the regime boundary by the group weight.
    # -----------------------------------------------------------------------
    tau_w <- tau * w

    # -----------------------------------------------------------------------
    # Step 3: Apply the two-regime MCP rule.
    #
    #   Small effects (||z||_2 <= gamma * tau * w):
    #     Use group soft-threshold with a bias-correcting multiplier.
    #     The factor gamma/(gamma-1) > 1 partially "undoes" the shrinkage,
    #     reducing bias compared to group lasso while still allowing
    #     zero-selection. Note we pass tau_w to grp_soft_threshold (not tau),
    #     since grp_soft_threshold will apply w=1 internally — we've already
    #     incorporated the weight into tau_w.
    #
    #   Large effects (||z||_2 > gamma * tau * w):
    #     Return z unchanged. MCP's key advantage: genuinely large effects
    #     get zero penalty, eliminating shrinkage bias entirely.
    # -----------------------------------------------------------------------
    if (l2_norm_z <= gamma * tau_w) {
      # Note: we pass tau_w (already weight-adjusted) and w = 1 to avoid
      # double-counting the weight inside grp_soft_threshold.
      p_new <- (gamma / (gamma - 1)) * grp_soft_threshold(z, tau_w, w = 1)
    } else {
      p_new <- z
    }

    return(p_new)
  }
