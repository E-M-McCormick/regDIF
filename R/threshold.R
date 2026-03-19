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

  # Apply elastic net soft-thresholding formula.
  # Denominator (1 + tau*(1-alpha)) accounts for the ridge (L2) component.
  # Numerator subtraction (tau*alpha) accounts for the LASSO (L1) component.
  p_new <- sign(z)*max(abs(z/(1+tau*(1-alpha))) -
                         (tau*alpha)/(1+tau*(1-alpha)), 0)

  return(p_new)

  }

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
           tau) {

    # Compute L2 norm of the grouped parameter vector.
    l2_norm_z <- sqrt(sum(z**2))

    # Apply group soft-thresholding: shrink proportionally if above threshold,
    # otherwise set entire group to zero.
    p_new <-
      if(l2_norm_z > tau) {
        (l2_norm_z - tau)*(z/l2_norm_z)
      } else if (l2_norm_z <= tau) {
        c(0,0)
      }

    return(p_new)

  }


# MCP (minimax concave penalty) firm-thresholding operator.
#
# Applies the proximal operator for the MCP, which reduces the bias
# introduced by LASSO for large effects. The gamma parameter controls
# the degree of tapering: when |z| > gamma*tau, the parameter is
# unpenalized (no shrinkage). For smaller |z|, the MCP inflates the
# soft-threshold estimate by gamma/(gamma-1) to reduce bias.
#
# @param z Numeric scalar: unpenalized Newton-Raphson update for the parameter.
# @param alpha Numeric in [0,1]: elastic net mixing parameter.
# @param tau Numeric >= 0: penalty tuning parameter.
# @param gamma Numeric > 1: MCP concavity parameter controlling bias correction.
#   Larger gamma = faster tapering (less bias, possibly less stable).
# @return Numeric scalar: the penalized parameter estimate.
firm_threshold <-
  function(z,
           alpha,
           tau,
           gamma) {

  # For small effects (within the MCP penalty region), apply scaled
  # soft-thresholding with bias correction factor gamma/(gamma-1).
  if(abs(z/(1+tau*(1-alpha))) <= gamma*tau){
    p_new <- (gamma/(gamma-1))*soft_threshold(z,alpha,tau)
  }else{
    # For large effects (beyond the MCP penalty region), return the
    # unpenalized estimate (only ridge component remains).
    p_new <- z/(1+tau*(1-alpha))
  }

  return(p_new)
  }

# Group MCP firm-thresholding operator.
#
# Applies the proximal operator for the group MCP penalty, combining
# group selection (joint intercept + slope) with MCP bias correction.
#
# @param z Numeric vector of length 2: unpenalized updates for the
#   intercept and slope DIF parameters.
# @param tau Numeric >= 0: penalty tuning parameter.
# @param gamma Numeric > 1: MCP concavity parameter.
# @return Numeric vector of length 2: the penalized group parameter estimates.
grp_firm_threshold <-
  function(z,
           tau,
           gamma) {

    # Compute L2 norm for the group threshold comparison.
    l2_norm_z <- sqrt(sum(z**2))

    # For small group effects, apply scaled group soft-thresholding.
    if(l2_norm_z <= gamma*tau){
      p_new <- (gamma/(gamma-1))*grp_soft_threshold(z,tau)
    }else{
      # For large group effects, return unpenalized estimates.
      p_new <- z
    }

    return(p_new)
  }
