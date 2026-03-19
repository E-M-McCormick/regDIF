###############################################################################
###                                                                         ###
###   Analytic Derivatives for the Penalized EM Algorithm (M-Step)          ###
###                                                                         ###
###############################################################################
#
# This file contains all analytic first- and second-derivative (gradient and
# Hessian) functions used in the M-step of the penalized EM algorithm
# implemented in the regDIF package.
#
# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
# In an EM algorithm for latent variable models, the M-step maximizes the
# expected complete-data log-likelihood with respect to model parameters.
# regDIF uses Newton-Raphson updates in the M-step:
#
#   p_new = p_old - gradient / hessian
#
# Each derivative function in this file returns a two-element list:
#   [[1]] = first derivative  (gradient, scalar or vector)
#   [[2]] = second derivative (Hessian diagonal element, scalar, or full
#           Hessian matrix for block/multivariate updates)
#
# ---------------------------------------------------------------------------
# Function families
# ---------------------------------------------------------------------------
#
# 1. IMPACT (LATENT DISTRIBUTION) DERIVATIVES
#    - d_alpha / d_alpha_proxy:
#        Univariate derivatives for latent mean (alpha) parameters.
#        The latent mean is modeled as alpha_i = X_mean %*% beta_mean,
#        i.e., a linear function of person-level covariates.
#    - d_phi / d_phi_proxy:
#        Univariate derivatives for latent variance (phi) parameters.
#        The latent variance is modeled on the log scale:
#        phi_i = exp(X_var %*% beta_var) to ensure positivity.
#    - d_impact_block / d_impact_block_proxy:
#        Multivariate (block) derivatives that jointly update all mean and
#        variance impact parameters, returning the full gradient vector and
#        Hessian matrix including cross-derivatives.
#
# 2. BERNOULLI (BINARY ITEM) DERIVATIVES
#    - d_bernoulli / d_bernoulli_proxy:
#        Univariate derivatives for 2PL/Rasch binary item parameters.
#        Uses the logistic ICC where P = 1/(1+exp(-eta)) and the key
#        derivative identity involves P*(1-P) terms.
#    - d_bernoulli_itemblock / d_bernoulli_itemblock_proxy:
#        Multivariate (block) derivatives for all parameters of a single
#        binary item, returning the full gradient vector and Hessian matrix.
#
# 3. CATEGORICAL (ORDINAL/GRADED RESPONSE) DERIVATIVES
#    - d_categorical / d_categorical_proxy:
#        Univariate derivatives for graded response model (GRM) parameters.
#        Uses cumulative logistic probabilities; derivatives for slope/DIF
#        parameters involve differences of cumulative probabilities, while
#        threshold derivatives use ratio forms.
#    - d_categorical_itemblock:
#        Multivariate (block) derivatives for all parameters of a single
#        ordinal item.
#
# 4. GAUSSIAN (CONTINUOUS ITEM / CFA) DERIVATIVES
#    - d_mu_gaussian / d_mu_gaussian_proxy:
#        Univariate derivatives for the mean (location) parameters of
#        continuous items under a Gaussian measurement model (CFA-like).
#        Derivatives involve (y - mu)/sigma^2 residual terms.
#    - d_sigma_gaussian / d_sigma_gaussian_proxy:
#        Univariate derivatives for the variance (scale) parameters of
#        continuous items. The variance is parameterized on the log scale
#        with a baseline s0 and DIF effects s1.
#    - d_gaussian_itemblock / d_gaussian_itemblock_proxy:
#        Multivariate (block) derivatives for all parameters (location,
#        scale, and DIF effects) of a single continuous item.
#
# ---------------------------------------------------------------------------
# Proxy vs. quadrature variants
# ---------------------------------------------------------------------------
# Functions with the "_proxy" suffix replace quadrature-based integration
# over the latent variable with observed proxy scores (e.g., sum scores).
# This avoids the num_quad loop and yields simpler, faster computations
# but introduces approximation error.
#
# ---------------------------------------------------------------------------
# E-table usage
# ---------------------------------------------------------------------------
# The "etable" (or "etable_item") is the matrix of posterior weights from
# the E-step, with dimensions (samp_size x num_quad). Each entry w_{iq}
# represents the posterior probability that person i has latent trait value
# theta_q, given the observed data. The derivatives are weighted sums over
# these posterior probabilities.
#
# ---------------------------------------------------------------------------
# Parameter naming conventions
# ---------------------------------------------------------------------------
#   c0 = baseline intercept            a0 = baseline slope (discrimination)
#   c1 = DIF effect on intercept       a1 = DIF effect on slope
#   s0 = baseline residual variance    s1 = DIF effect on residual variance
#   (for Gaussian items only)
#
###############################################################################


###############################################################################
### SECTION 1: Impact (Latent Distribution) Parameter Derivatives           ###
###############################################################################
#
# The latent variable theta ~ N(alpha_i, phi_i) has a person-specific mean
# alpha_i and variance phi_i determined by covariates:
#
#   alpha_i = X_mean_i %*% beta_mean     (latent mean equation)
#   phi_i   = exp(X_var_i %*% beta_var)  (latent variance, log-link)
#
# The complete-data log-likelihood contribution for person i at quad point q:
#   log f(theta_q | alpha_i, phi_i) = -0.5*log(phi_i)
#                                     - 0.5*(theta_q - alpha_i)^2 / phi_i
#
# Derivatives w.r.t. mean parameters (alpha) involve:
#   d/d(beta_mean_k) = X_mean[i,k] / phi_i * (theta_q - alpha_i)
#
# Derivatives w.r.t. variance parameters (phi) involve:
#   d/d(beta_var_k) = 0.5 * X_var[i,k] * [(theta_q-alpha_i)^2/phi_i - 1]
#   (using the chain rule through the exp link)
#
###############################################################################

#' Partial derivatives for mean impact equation.
#'
#' Computes the gradient and Hessian diagonal element for a single covariate
#' in the latent mean equation, alpha_i = X_mean %*% beta_mean. Used in
#' coordinate descent where one parameter is updated at a time via
#' univariate Newton-Raphson.
#'
#' @param p_impact Vector of impact parameters.
#' @param etable E-table for impact (samp_size x num_quad matrix of posterior
#'   weights from the E-step).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param cov Covariate being maximized (column index in mean_predictors).
#' @param samp_size Sample size in data set.
#' @param num_items Number of items in data set.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for mean impact equation (to
#' use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_alpha <-
  function(p_impact,
           etable,
           theta,
           mean_predictors,
           var_predictors,
           cov,
           samp_size,
           num_items,
           num_quad) {

  # Get latent mean and variance vectors.
  # alpha_i = X_mean_i %*% beta_mean for each person i.
  alpha <- mean_predictors %*% p_impact[grep("mean",names(p_impact),fixed=T)]
  # phi_i = exp(X_var_i %*% beta_var); exp link ensures positivity.
  phi <- exp(var_predictors %*% p_impact[grep("var",names(p_impact),fixed=T)])

  # First derivative of the normal log-likelihood w.r.t. one mean parameter:
  #   d/d(beta_mean_k) = X_mean[i,k] / phi_i * (theta_q - alpha_i)
  # Evaluated at each quadrature point for each person.
  d1_trace <- vapply(1:num_quad,
                       function(x) {
                         mean_predictors[,cov]/phi*(theta[x]-alpha)
                         },numeric(samp_size))
  # Second derivative (always negative for the normal distribution):
  #   d2/d(beta_mean_k)^2 = -X_mean[i,k]^2 / phi_i
  d2_trace <- vapply(1:num_quad,
                       function(x) {
                         -mean_predictors[,cov]**2/phi
                         },numeric(samp_size))

  # Weight by posterior probabilities (E-table) and sum over all persons
  # and quadrature points.
  d1 <- sum(etable*d1_trace, na.rm = TRUE)
  d2 <- sum(etable*d2_trace, na.rm = TRUE)

  dlist <- list(d1,d2)

}

#' Partial derivatives for mean impact equation using proxy data.
#'
#' Proxy variant of \code{d_alpha}. Instead of integrating over the latent
#' variable using quadrature, this function substitutes observed proxy scores
#' (e.g., standardized sum scores) directly for theta. This collapses the
#' samp_size x num_quad matrix to a samp_size vector, greatly simplifying
#' and speeding up computation.
#'
#' @param p_impact Vector of impact parameters.
#' @param prox_data Matrix of observed proxy scores (samp_size x 1),
#'   substituted for the latent variable theta.
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param cov Covariate being maximized.
#' @param samp_size Sample size in data set.
#' @param num_items Number of items in data set.
#'
#' @return a \code{"list"} of first and second partial derivatives for mean impact equation (to
#' use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_alpha_proxy <-
  function(p_impact,
           prox_data,
           mean_predictors,
           var_predictors,
           cov,
           samp_size,
           num_items) {

    # Get latent mean and variance vectors.
    alpha <- mean_predictors %*% p_impact[grep("mean",names(p_impact),fixed=T)]
    phi <- exp(var_predictors %*% p_impact[grep("var",names(p_impact),fixed=T)])

    # Same formula as d_alpha but with prox_data replacing theta[x].
    # No loop over quadrature points needed -- one value per person.
    d1_trace <- mean_predictors[,cov]/phi*(prox_data-alpha)
    d2_trace <- -mean_predictors[,cov]**2/phi

    # Sum over all persons (no E-table weighting needed with proxy scores).
    d1 <- sum(d1_trace, na.rm = TRUE)
    d2 <- sum(d2_trace, na.rm = TRUE)

    dlist <- list(d1,d2)

  }


#' Partial derivatives for variance impact equation.
#'
#' Computes the gradient and Hessian diagonal element for a single covariate
#' in the latent variance equation, phi_i = exp(X_var %*% beta_var). Used
#' in coordinate descent with univariate Newton-Raphson updates. Because
#' phi is parameterized via the exp link, the chain rule introduces
#' sqrt(phi) and phi^(-3/2) terms in the derivatives.
#'
#' @param p_impact Vector of impact parameters.
#' @param etable E-table for impact (samp_size x num_quad posterior weights).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param cov Covariate being maximized (column index in var_predictors).
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for variance impact equation (to
#' use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_phi <-
  function(p_impact,
           etable,
           theta,
           mean_predictors,
           var_predictors,
           cov,
           samp_size,
           num_items,
           num_quad) {

  # Get latent mean and variance vectors
  alpha <- mean_predictors %*% p_impact[grep("mean",names(p_impact),fixed=T)]
  phi <- exp(var_predictors %*% p_impact[grep("var",names(p_impact),fixed=T)])

  # Chain rule factor for the exp-link on phi.
  # d(sqrt(phi))/d(beta_var_k) = 0.5 * sqrt(phi) * X_var[i,k]
  # These are the "eta_d" terms that multiply the base derivative of the
  # normal log-likelihood w.r.t. the standard deviation sqrt(phi).
  eta_d1 <- .5*sqrt(phi)*var_predictors[,cov]
  eta_d2 <- .5*sqrt(phi)*var_predictors[,cov]**2

  # First derivative of normal LL w.r.t. beta_var_k:
  #   eta_d1 * [(theta_q - alpha_i)^2 / phi^(3/2) - 1/sqrt(phi)]
  # The two terms arise from d/d(sqrt(phi))[-0.5*log(phi) - 0.5*(theta-alpha)^2/phi].
  d1_trace <- vapply(1:num_quad,
                       function(x) {
                         eta_d1*((theta[x]-alpha)**2/phi**(3/2) -
                                      1/sqrt(phi))
                         },numeric(samp_size))
  # Second derivative (Fisher-information-like approximation):
  #   -eta_d2 * phi^(-3/2) * (theta_q - alpha_i)^2
  d2_trace <- vapply(1:num_quad,
                       function(x) {
                         -eta_d2*(phi**(-3/2)*(theta[x]-alpha)**2)
                         },numeric(samp_size))

  # Weight by E-table and sum across persons and quadrature points.
  d1 <- sum(etable*d1_trace, na.rm = TRUE)
  d2 <- sum(etable*d2_trace, na.rm = TRUE)

  dlist <- list(d1,d2)

  }

#' Partial derivatives for variance impact equation using proxy data.
#'
#' Proxy variant of \code{d_phi}. Substitutes observed proxy scores for the
#' quadrature-based latent variable, eliminating the loop over quadrature
#' points and E-table weighting.
#'
#' @param p_impact Vector of impact parameters.
#' @param prox_data Matrix of observed proxy scores (samp_size x 1),
#'   substituted for the latent variable theta.
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param cov Covariate being maximized.
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for variance impact equation (to
#' use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_phi_proxy <-
  function(p_impact,
           prox_data,
           mean_predictors,
           var_predictors,
           cov,
           samp_size,
           num_items) {

    # Get latent mean and variance vectors
    alpha <- mean_predictors %*% p_impact[grep("mean",names(p_impact),fixed=T)]
    phi <- exp(var_predictors %*% p_impact[grep("var",names(p_impact),fixed=T)])

    # Chain rule factors for exp-link (same as d_phi).
    eta_d1 <- .5*sqrt(phi)*var_predictors[,cov]
    eta_d2 <- .5*sqrt(phi)*var_predictors[,cov]**2

    # Same derivative formulas as d_phi but with prox_data replacing theta[x].
    d1_trace <- eta_d1*((prox_data-alpha)**2/phi**(3/2) -
                                   1/sqrt(phi))
    d2_trace <- -eta_d2*(phi**(-3/2)*(prox_data-alpha)**2)

    # Sum directly over persons (no E-table weighting with proxy scores).
    d1 <- sum(d1_trace, na.rm = TRUE)
    d2 <- sum(d2_trace, na.rm = TRUE)

    dlist <- list(d1,d2)

  }

#' Partial derivatives for mean and variance impact equation (block update).
#'
#' Computes the full gradient vector and Hessian matrix for all mean and
#' variance impact parameters simultaneously. This enables a multivariate
#' Newton-Raphson update: p_new = p_old - H^{-1} %*% g, which accounts
#' for correlations between parameters and typically converges faster than
#' coordinate descent.
#'
#' The gradient vector d1 has length (num_mean_parms + num_var_parms).
#' The Hessian matrix d2 is the corresponding square matrix, with diagonal
#' blocks for mean-mean and variance-variance second derivatives, and
#' off-diagonal blocks for mean-variance cross-derivatives.
#'
#' @param p_mean Vector of mean impact parameters.
#' @param p_var Vector of variance impact parameters.
#' @param etable E-table for impact (samp_size x num_quad posterior weights).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param samp_size Sample size in data set.
#' @param num_items Number of items in data set.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#' @param num_predictors Number of predictors in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for impact equation (to use
#' with multivariate Newton-Raphson)
#'
#' @keywords internal
#'
d_impact_block <-
  function(p_mean,
           p_var,
           etable,
           theta,
           mean_predictors,
           var_predictors,
           samp_size,
           num_items,
           num_quad,
           num_predictors) {

    # Obtain number of impact parameters.
    # The parameter vector is [beta_mean_1,...,beta_mean_p, beta_var_1,...,beta_var_q].
    num_impact_parms <- length(p_mean) + length(p_var)

    # Make space for first and second derivatives.
    # d1 is the gradient vector; d2 is the full (lower-triangular) Hessian matrix.
    d1 <- matrix(0,nrow=num_impact_parms,ncol=1)
    d2 <- matrix(0,nrow=num_impact_parms,ncol=num_impact_parms)

    # Get latent mean and variance vectors.
    alpha <- mean_predictors %*% p_mean
    phi <- exp(var_predictors %*% p_var)

    # --- Diagonal block: mean-mean derivatives ---
    # Same formulas as d_alpha, computed for all mean covariates.
    for(cov in 1:ncol(mean_predictors)) {
      d1_trace_mean <- vapply(1:num_quad,
                              function(x) {
                                mean_predictors[,cov]/phi*(theta[x]-alpha)
                              },numeric(samp_size))
      d2_trace_mean <- vapply(1:num_quad,
                              function(x) {
                                -mean_predictors[,cov]**2/phi
                              },numeric(samp_size))

      d1[cov,1] <- sum(etable*d1_trace_mean, na.rm = TRUE)
      d2[cov,cov] <- sum(etable*d2_trace_mean, na.rm = TRUE)
    }

    # --- Diagonal block: variance-variance derivatives ---
    # Uses the simplified form for the log-link parameterization of phi.
    # d/d(beta_var_k) = 0.5 * X_var[i,k] * [(theta-alpha)^2/phi - 1]
    for(cov in 1:ncol(var_predictors)) {
      d1_trace_var <-
        vapply(1:num_quad,
               function(x) {
                 .5*var_predictors[,cov]*(
                   (theta[x]-alpha)**2/phi - 1)
                 },numeric(samp_size))
      d2_trace_var <-
        vapply(1:num_quad,
               function(x) {
                 -.5*var_predictors[,cov]**2*(theta[x]-alpha)**2/phi
                 },numeric(samp_size))
      d1[ncol(mean_predictors)+cov,1] <-
        sum(etable*d1_trace_var, na.rm = TRUE)
      d2[ncol(mean_predictors)+cov,ncol(mean_predictors)+cov] <-
        sum(etable*d2_trace_var, na.rm = TRUE)
    }

    # --- Off-diagonal block: mean-variance cross-derivatives ---
    # d2/(d(beta_mean_j) d(beta_var_k)) captures how the mean and variance
    # parameters interact in the normal log-likelihood.
    for(cov in 1:ncol(var_predictors)) {
      for(cov2 in 1:ncol(mean_predictors)) {

        # Cross-derivative between variance parameter cov and mean parameter cov2.
        d2_trace_cross <-
          vapply(1:num_quad,
                 function(x) {
                   var_predictors[,cov]*mean_predictors[,cov2]/phi*(
                     alpha-theta[x])
                 },numeric(samp_size))
        d2[ncol(mean_predictors)+cov,cov2] <-
          sum(etable*d2_trace_cross, na.rm = TRUE)

        if(cov2 > 1 && cov < cov2) {


          # Cross derivatives for mean parameters with different predictors.
          # d2/(d(beta_mean_cov) d(beta_mean_cov2)) for off-diagonal mean block.
          d2_trace_cross_mean <-
            vapply(1:num_quad,
                   function(x) {
                     -mean_predictors[,cov]*mean_predictors[,cov2]/phi
                   },numeric(samp_size))
          d2[cov2,cov] <-
            sum(etable*d2_trace_cross_mean,
                na.rm = TRUE)

          if(cov2 <= length(p_var)) {
          # Cross derivatives for variance parameters with different predictors.
          # d2/(d(beta_var_cov) d(beta_var_cov2)) for off-diagonal variance block.
          d2_trace_cross_var <-
            vapply(1:num_quad,
                   function(x) {
                     -.5*var_predictors[,cov]*var_predictors[,cov2]*(
                       theta[x]-alpha)**2/phi
                   },numeric(samp_size))
          d2[ncol(mean_predictors)+cov2,ncol(mean_predictors)+cov] <-
            sum(etable*d2_trace_cross_var,
                na.rm = TRUE)
          }
        }


      }
    }

    dlist <- list(d1,d2)

  }


#' Partial derivatives for mean and variance impact equation using observed score proxy.
#'
#' Proxy variant of \code{d_impact_block}. Computes the full gradient vector
#' and Hessian matrix for all impact parameters using observed proxy scores
#' instead of quadrature-based integration. Same block structure as
#' \code{d_impact_block} but without the vapply loops over quadrature points.
#'
#' @param p_mean Vector of mean impact parameters.
#' @param p_var Vector of variance impact parameters.
#' @param prox_data Vector of observed proxy scores.
#' @param mean_predictors Possibly different matrix of predictors for the mean
#' impact equation.
#' @param var_predictors Possibly different matrix of predictors for the
#' variance impact equation.
#' @param samp_size Sample size in data set.
#' @param num_items Number of items in data set.
#' @param num_predictors Number of predictors in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for impact equation (to
#' use with multivariate Newton-Rapshon and observed proxy scores)
#'
#' @keywords internal
#'
d_impact_block_proxy <-
  function(p_mean,
           p_var,
           prox_data,
           mean_predictors,
           var_predictors,
           samp_size,
           num_items,
           num_quad,
           num_predictors) {

    # Obtain number of impact parameters.
    num_impact_parms <- length(p_mean) + length(p_var)

    # Make space for first and second derivatives.
    d1 <- matrix(0,nrow=num_impact_parms,ncol=1)
    d2 <- matrix(0,nrow=num_impact_parms,ncol=num_impact_parms)

    # Get latent mean and variance vectors.
    alpha <- mean_predictors %*% p_mean
    phi <- exp(var_predictors %*% p_var)

    # --- Diagonal block: mean-mean derivatives (proxy version) ---
    for(cov in 1:ncol(mean_predictors)) {
      d1_trace_mean <- mean_predictors[,cov]/phi*(prox_data-alpha)
      d2_trace_mean <- -mean_predictors[,cov]**2/phi

      d1[cov,1] <- sum(d1_trace_mean, na.rm = TRUE)
      d2[cov,cov] <- sum(d2_trace_mean, na.rm = TRUE)
    }

    # --- Diagonal block: variance-variance derivatives (proxy version) ---
    for(cov in 1:ncol(var_predictors)) {
      d1_trace_var <- .5*var_predictors[,cov]*((prox_data-alpha)**2/phi - 1)
      d2_trace_var <- -.5*var_predictors[,cov]**2*(prox_data-alpha)**2/phi
      d1[ncol(mean_predictors)+cov,1] <- sum(d1_trace_var, na.rm = TRUE)
      d2[ncol(mean_predictors)+cov,ncol(mean_predictors)+cov] <- sum(d2_trace_var, na.rm = TRUE)
    }

    # --- Off-diagonal block: mean-variance cross-derivatives (proxy version) ---
    for(cov in 1:ncol(var_predictors)) {
      for(cov2 in 1:ncol(mean_predictors)) {

        # Cross-derivative between variance parameter cov and mean parameter cov2.
        d2_trace_cross <- var_predictors[,cov]*mean_predictors[,cov2]/phi*(alpha-prox_data)
        d2[ncol(mean_predictors)+cov,cov2] <- sum(d2_trace_cross, na.rm = TRUE)

        if(cov2 > 1 && cov < cov2) {


          # Cross derivatives for mean parameters with different predictors.
          d2_trace_cross_mean <- -mean_predictors[,cov]*mean_predictors[,cov2]/phi
          d2[cov2,cov] <- sum(d2_trace_cross_mean, na.rm = TRUE)

          if(cov2 <= length(p_var)) {
            # Cross derivatives for variance parameters with different predictors.
            d2_trace_cross_var <-
              -.5*var_predictors[,cov]*var_predictors[,cov2]*(prox_data-alpha)**2/phi
            d2[ncol(mean_predictors)+cov2,ncol(mean_predictors)+cov] <- sum(d2_trace_cross_var,
                                                                            na.rm = TRUE)
          }
        }


      }
    }

    dlist <- list(d1,d2)

  }


###############################################################################
### SECTION 2: Bernoulli (Binary Item) Derivatives                         ###
###############################################################################
#
# Binary items are modeled with the 2PL (or Rasch) IRT model using the
# logistic item characteristic curve (ICC):
#
#   P(Y=1 | theta, x) = 1 / (1 + exp(-eta))
#
# where the linear predictor eta is:
#   eta = (c0 + x'c1) + (a0 + x'a1) * theta
#
# c0 = baseline intercept, c1 = DIF effects on intercept,
# a0 = baseline slope (discrimination), a1 = DIF effects on slope.
#
# Key derivative identities for the Bernoulli log-likelihood:
#   dP/d(eta) = P * (1 - P)
#   d(log L)/d(eta) = [y/P - (1-y)/(1-P)] * P*(1-P) = y - P
#   d2(log L)/d(eta)^2 = -P * (1 - P)  (always negative => concave)
#
# The first derivative (gradient) for parameter beta_k is:
#   d1 = sum_i sum_q eta_d[i,q] * [y_i * (1-P) - (1-y_i) * P] * w_{iq}
# where eta_d = d(eta)/d(beta_k) and w_{iq} is the E-table entry.
#
# The second derivative (Hessian) for parameter beta_k is:
#   d2 = sum_i sum_q eta_d[i,q]^2 * [-P*(1-P)] * w_{iq}
#
###############################################################################

#' Partial derivatives for binary items.
#'
#' Computes the gradient and Hessian diagonal element for a single parameter
#' of a binary (2PL/Rasch) item. The parameter to differentiate with respect
#' to is specified by \code{parm}: "c0" (baseline intercept), "a0" (baseline
#' slope), "c1" (DIF intercept effect), or "a1" (DIF slope effect). Used in
#' coordinate descent with univariate Newton-Raphson updates.
#'
#' @param parm Item parameter being maximized (one of "c0", "a0", "c1", "a1").
#' @param p_item Vector of item parameters.
#' @param etable_item E-table for item (list of 2 matrices: one per response
#'   category, each samp_size x num_quad).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param cov Covariate being maximized (column index in pred_data).
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for Bernoulli item likelihood (to
#' use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_bernoulli <-
  function(parm,
           p_item,
           etable_item,
           theta,
           pred_data,
           cov,
           samp_size,
           num_items,
           num_quad) {

  # Compute eta_d = d(eta)/d(parm), the derivative of the linear predictor
  # with respect to the parameter being optimized. This is a samp_size x
  # num_quad matrix (constant across quad points for intercept terms,
  # theta-dependent for slope terms).
  if(parm == "c0"){
    # d(eta)/d(c0) = 1 for all persons and quadrature points.
    eta_d <- matrix(1, nrow = samp_size, ncol = num_quad)
  } else if(parm == "a0"){
    # d(eta)/d(a0) = theta (the latent variable at each quadrature point).
    eta_d <- t(matrix(theta,ncol=samp_size,nrow=num_quad))
  } else if(parm == "c1"){
    # d(eta)/d(c1_k) = x_k (the covariate value for predictor k).
    eta_d <- matrix(pred_data[,cov],
                    ncol = num_quad,
                    nrow = samp_size)
  } else if(parm == "a1"){
    # d(eta)/d(a1_k) = x_k * theta (interaction of covariate and latent trait).
    eta_d <- matrix(pred_data[,cov],
                    ncol = num_quad,
                    nrow = samp_size)*t(matrix(theta,
                                               ncol=samp_size,
                                               nrow=num_quad))
  }

  # Compute P(Y=1 | theta_q, x_i) = logistic ICC at each (person, quad point).
  traceline <- bernoulli_traceline_pts(p_item,
                                       theta,
                                       pred_data,
                                       samp_size)

  # First derivative of the Bernoulli LL:
  #   d1 = sum eta_d * P * [(y/P) - 1] * w_{iq}
  # where etable_item[[2]] = w_{iq} for Y=1 responses, etable_item[[1]] = w_{iq}
  # for Y=0 responses. The expression simplifies to:
  #   eta_d * P * [etable_item[[2]]/P - etable_item[[2]] - etable_item[[1]]]
  d1 <- sum(eta_d*traceline*(etable_item[[2]]/traceline -
                               etable_item[[2]] -
                               etable_item[[1]]),
            na.rm = TRUE)
  # Second derivative of the Bernoulli LL:
  #   d2 = sum eta_d^2 * [-P*(1-P)] * [total E-table weight]
  # The term (-P + P^2) = -P*(1-P) ensures the Hessian is negative (concave).
  d2 <- sum(eta_d**2*(-traceline + traceline**2)*(etable_item[[1]] +
                                                    etable_item[[2]]),
            na.rm = TRUE)

  dlist <- list(d1,d2)

  }

#' Partial derivatives for binary items with proxy data.
#'
#' Proxy variant of \code{d_bernoulli}. Uses observed proxy scores instead of
#' quadrature to approximate the latent variable. Since there is no E-table,
#' the first derivative is computed directly from observed responses
#' (item_data_current) and the ICC evaluated at the proxy score.
#'
#' @param parm Item parameter being maximized (one of "c0", "a0", "c1", "a1").
#' @param p_item Vector of item parameters.
#' @param prox_data Vector of observed proxy scores (samp_size x 1).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param item_data_current Vector of current item responses (coded 1 = no,
#'   2 = yes for the binary response).
#' @param cov Covariate being maximized.
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for Bernoulli item likelihood (to
#' use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_bernoulli_proxy <-
  function(parm,
           p_item,
           prox_data,
           pred_data,
           item_data_current,
           cov,
           samp_size,
           num_items) {

    # Compute eta_d = d(eta)/d(parm) using proxy scores for theta.
    # Matrices collapse to samp_size x 1 (single "quadrature point" per person).
    if(parm == "c0"){
      eta_d <- matrix(1, nrow = samp_size, ncol = 1)
    } else if(parm == "a0"){
      eta_d <- prox_data
    } else if(parm == "c1"){
      eta_d <- matrix(pred_data[,cov],
                      ncol = 1,
                      nrow = samp_size)
    } else if(parm == "a1"){
      eta_d <- matrix(pred_data[,cov],
                      ncol = 1,
                      nrow = samp_size)*prox_data
    }

    # Compute P(Y=1) at proxy scores.
    traceline <- bernoulli_traceline_pts_proxy(p_item,
                                               prox_data,
                                               pred_data)

    # Construct the base first derivative: (y_i - P_i) for each person.
    # item_data_current == 1 means Y=0 (coded as category 1), contributing -P.
    # item_data_current == 2 means Y=1 (coded as category 2), contributing (1-P).
    d1_base <- matrix(0, nrow = nrow(traceline), ncol = 1)
    d1_base[item_data_current == 1,] <- -traceline[item_data_current == 1,]
    d1_base[item_data_current == 2,] <- (1 - traceline)[item_data_current == 2,]

    # Gradient: weighted by eta_d.
    d1 <- sum(eta_d*d1_base,
              na.rm = TRUE)

    # Hessian diagonal: -P*(1-P) * eta_d^2, always negative.
    d2 <- sum(eta_d**2*(-traceline + traceline**2),
              na.rm = TRUE)

    dlist <- list(d1,d2)

  }

#' Partial derivatives for binary items by item-blocks (multivariate update).
#'
#' Computes the full gradient vector and Hessian matrix for all parameters
#' of a single binary item simultaneously. This enables a multivariate
#' Newton-Raphson update for the item: p_new = p_old - H^{-1} g.
#'
#' The parameter vector is ordered as: [c0, a0, c1_1,...,c1_K, a1_1,...,a1_K]
#' where K = num_predictors. The gradient d1 has length (2 + 2*K) and the
#' Hessian d2 is the corresponding square matrix.
#'
#' The "d1_base" and "d2_base" quantities are the derivative contributions
#' before multiplying by the parameter-specific eta_d terms. This factored
#' approach avoids redundant traceline computations.
#'
#' @param p_item Vector of item parameters.
#' @param etable E-table for item (samp_size x num_quad posterior weights).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param item_data_current Vector of current item responses (1 or 2).
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_predictors Number of predictors in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for Bernoulli item likelihood (to
#' use with multivariate Newton-Raphson)
#'
#' @keywords internal
#'
d_bernoulli_itemblock <-
  function(p_item,
           etable,
           theta,
           pred_data,
           item_data_current,
           samp_size,
           num_items,
           num_predictors,
           num_quad) {

    # Make space for first and second derivatives.
    d1 <- matrix(0,nrow=length(p_item),ncol=1)
    d2 <- matrix(0,nrow=length(p_item),ncol=length(p_item))

    # First derivative for linear predictor w.r.t. theta.
    eta_d_a0 <- t(matrix(theta,
                         ncol=samp_size,
                         nrow=num_quad))

    # Get item response function.
    traceline <- bernoulli_traceline_pts(p_item,
                                         theta,
                                         pred_data,
                                         samp_size)

    # Get posterior probabilities for each response.
    etable_item <- lapply(1:2, function(x) etable)

    # Obtain E-tables for each response category.
      for(resp in 1:2) {
        etable_item[[resp]][which(
          !(item_data_current == resp)), ] <- 0
      }

    # Calculate first and second base derivatives (shared across all parameters).
    # d1_base = P * [w_{iq}^{(Y=1)}/P - w_{iq}^{(Y=1)} - w_{iq}^{(Y=0)}]
    #         = w_{iq}^{(Y=1)} * (1-P) - w_{iq}^{(Y=0)} * P  (i.e., E[Y-P])
    d1_base <- traceline*(etable_item[[2]]/traceline -
                            etable_item[[2]] -
                            etable_item[[1]])
    # d2_base = -P*(1-P) * total_weight, always negative (concave LL).
    d2_base <- (-traceline + traceline**2)*etable

    # First and second derivative for c0.
    d1[1,1] <- sum(d1_base, na.rm = TRUE) #d1
    d2[1,1] <- sum(d2_base, na.rm = TRUE) #d2

    # First and second derivative for a0.
    d1[2,1] <- sum(eta_d_a0*d1_base, na.rm = TRUE) #d1
    d2[2,2] <- sum(eta_d_a0**2*d2_base, na.rm = TRUE) #d2

    # Cross derivative for c0 and a0.
    d2[2,1] <- sum(eta_d_a0*d2_base, na.rm = TRUE) #d2

    # Cycle through predictors (outer cycle).
    for(cov in 1:num_predictors) {

      # First derivative for linear predictor w.r.t. covariate.
      cov_matrix <- matrix(pred_data[,cov],
                           ncol = num_quad,
                           nrow = samp_size)

      # First and second derivatives for c1.
      d1[2+cov,1] <-
        sum(cov_matrix*d1_base,
            na.rm = TRUE) #d1
      d2[2+cov,2+cov] <-
        sum(cov_matrix**2*d2_base,
            na.rm = TRUE) #d2

      # First and second derivatives for a1.
      d1[2+num_predictors+cov,1] <-
        sum(cov_matrix*eta_d_a0*d1_base,
            na.rm = TRUE) #d1
      d2[2+num_predictors+cov,2+num_predictors+cov] <-
        sum((cov_matrix*eta_d_a0)**2*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for c0 and c1.
      d2[2+cov,1] <-
        sum(cov_matrix*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for c0 and a1, as well as a0 and c1.
      d2[2+num_predictors+cov,1] <- d2[2+cov,2] <-
        sum(cov_matrix*eta_d_a0*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for a0 and a1.
      d2[2+num_predictors+cov,2] <-
        sum(cov_matrix*eta_d_a0**2*d2_base,
            na.rm = TRUE) #d2

      # Cycle through predictors (inner cycle).
      for(cov2 in 1:num_predictors) {


        if(cov == cov2) {

          # Cross derivatives with same predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix**2*eta_d_a0*d2_base,
                na.rm = TRUE) #d2

        } else {

          # First derivatives for linear predictor w.r.t. second covariate.
          cov2_matrix <- matrix(pred_data[,cov2],
                                ncol = num_quad,
                                nrow = samp_size)

          # Cross derivatives with different predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix*cov2_matrix*eta_d_a0*d2_base,
                na.rm = TRUE) #d2

          if(cov2 > 1 && cov < cov2) {

            # Cross derivatives with different predictor for c1 and c1.
            d2[2+cov2,2+cov] <-
              sum(cov_matrix*cov2_matrix*d2_base, #d2
                  na.rm = TRUE)

            # Cross derivatives with different predictor for a1 and a1.
            d2[2+num_predictors+cov2,2+num_predictors+cov] <-
              sum(cov_matrix*cov2_matrix*eta_d_a0**2*d2_base, #a1a1
                  na.rm = TRUE)
          }
        }

      }

    }

    dlist <- list(d1,d2)

  }


#' Partial derivatives for binary items by item-blocks using observed score proxy.
#'
#' Proxy variant of \code{d_bernoulli_itemblock}. Computes the full gradient
#' vector and Hessian matrix for all parameters of a single binary item using
#' observed proxy scores instead of quadrature-based integration. The structure
#' is identical to \code{d_bernoulli_itemblock} but with samp_size x 1 matrices
#' instead of samp_size x num_quad.
#'
#' @param p_item Vector of item parameters.
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param item_data_current Vector of current item responses (1 or 2).
#' @param prox_data Vector of observed proxy scores (samp_size x 1).
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_predictors Number of predictors in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for Bernoulli item likelihood (to
#' use with multivariate Newton-Raphson and observed proxy scores)
#'
#' @keywords internal
#'
d_bernoulli_itemblock_proxy <-
  function(p_item,
           pred_data,
           item_data_current,
           prox_data,
           samp_size,
           num_items,
           num_predictors,
           num_quad) {

    # Make space for first and second derivatives.
    d1 <- matrix(0,nrow=length(p_item),ncol=1)
    d2 <- matrix(0,nrow=length(p_item),ncol=length(p_item))

    # First derivative for linear predictor w.r.t. theta.
    eta_d_a0 <- prox_data

    # Get item response function.
    traceline <- bernoulli_traceline_pts_proxy(p_item,
                                               prox_data,
                                               pred_data)

    # Calculate first and second base derivatives (proxy version).
    # d1_base = (y_i - P_i): negative P for Y=0 responses, positive (1-P) for Y=1.
    d1_base <- matrix(0, nrow = nrow(traceline), ncol = 1)
    d1_base[item_data_current == 1,] <- -traceline[item_data_current == 1,]
    d1_base[item_data_current == 2,] <- (1 - traceline)[item_data_current == 2,]

    # d2_base = -P*(1-P), always negative (concave LL). No E-table needed.
    d2_base <- -traceline + traceline**2

    # First and second derivative for c0.
    d1[1,1] <- sum(d1_base, na.rm = TRUE) #d1
    d2[1,1] <- sum(d2_base, na.rm = TRUE) #d2

    # First and second derivative for a0.
    d1[2,1] <- sum(eta_d_a0*d1_base, na.rm = TRUE) #d1
    d2[2,2] <- sum(eta_d_a0**2*d2_base, na.rm = TRUE) #d2

    # Cross derivative for c0 and a0.
    d2[2,1] <- sum(eta_d_a0*d2_base, na.rm = TRUE) #d2

    # Cycle through predictors (outer cycle).
    for(cov in 1:num_predictors) {

      # First derivative for linear predictor w.r.t. covariate.
      cov_matrix <- pred_data[,cov]

      # First and second derivatives for c1.
      d1[2+cov,1] <-
        sum(cov_matrix*d1_base,
            na.rm = TRUE) #d1
      d2[2+cov,2+cov] <-
        sum(cov_matrix**2*d2_base,
            na.rm = TRUE) #d2

      # First and second derivatives for a1.
      d1[2+num_predictors+cov,1] <-
        sum(cov_matrix*eta_d_a0*d1_base,
            na.rm = TRUE) #d1
      d2[2+num_predictors+cov,2+num_predictors+cov] <-
        sum((cov_matrix*eta_d_a0)**2*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for c0 and c1.
      d2[2+cov,1] <-
        sum(cov_matrix*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for c0 and a1, as well as a0 and c1.
      d2[2+num_predictors+cov,1] <- d2[2+cov,2] <-
        sum(cov_matrix*eta_d_a0*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for a0 and a1.
      d2[2+num_predictors+cov,2] <-
        sum(cov_matrix*eta_d_a0**2*d2_base,
            na.rm = TRUE) #d2

      # Cycle through predictors (inner cycle).
      for(cov2 in 1:num_predictors) {


        if(cov == cov2) {

          # Cross derivatives with same predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix**2*eta_d_a0*d2_base,
                na.rm = TRUE) #d2

        } else {

          # First derivatives for linear predictor w.r.t. second covariate.
          cov2_matrix <- pred_data[,cov2]

          # Cross derivatives with different predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix*cov2_matrix*eta_d_a0*d2_base,
                na.rm = TRUE) #d2

          if(cov2 > 1 && cov < cov2) {

            # Cross derivatives with different predictor for c1 and c1.
            d2[2+cov2,2+cov] <-
              sum(cov_matrix*cov2_matrix*d2_base, #d2
                  na.rm = TRUE)

            # Cross derivatives with different predictor for a1 and a1.
            d2[2+num_predictors+cov2,2+num_predictors+cov] <-
              sum(cov_matrix*cov2_matrix*eta_d_a0**2*d2_base, #a1a1
                  na.rm = TRUE)
          }
        }

      }

    }

    dlist <- list(d1,d2)


  }

###############################################################################
### SECTION 3: Categorical (Ordinal / Graded Response) Derivatives         ###
###############################################################################
#
# Ordinal items are modeled with the Graded Response Model (GRM), which uses
# cumulative logistic probabilities:
#
#   P*(Y >= k | theta, x) = 1 / (1 + exp(-(c0_k + x'c1 + (a0 + x'a1)*theta)))
#
# The category response probability is the difference of adjacent cumulative
# probabilities:
#   P(Y = k | theta, x) = P*(Y >= k) - P*(Y >= k+1)
#
# Derivatives for slope and DIF parameters (thr < 0, i.e., non-threshold
# parameters) involve sums over all response categories of terms like:
#   d(P(Y=k))/d(eta) = -P*_k*(1-P*_k) + P*_{k-1}*(1-P*_{k-1})
#
# Threshold derivatives (thr >= 1) use ratio forms involving the cumulative
# probabilities for adjacent categories, because each threshold only affects
# two adjacent category probabilities.
#
# Parameter indexing for ordinal items:
#   Positions 1..(num_responses-2): threshold parameters (c0_2,...,c0_{K-1})
#   Position (num_responses-1):     baseline slope a0
#   Followed by c1 and a1 DIF parameters.
#
###############################################################################

#' Partial derivatives for ordinal items.
#'
#' Computes the gradient and Hessian diagonal element for a single parameter
#' of an ordinal (GRM) item. When \code{thr < 0}, the parameter is a
#' non-threshold parameter (slope or DIF effect) and the derivative sums
#' over all response categories. When \code{thr >= 1}, the parameter is a
#' specific threshold and only two adjacent categories contribute.
#'
#' @param parm Item parameter being maximized (one of "c0", "a0", "c1", "a1").
#' @param p_item Vector of item parameters.
#' @param etable_item E-table for item (list of num_responses matrices, each
#'   samp_size x num_quad, zeroed out for non-matching response categories).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param thr Threshold value being maximized. Negative values indicate
#'   non-threshold parameters (slope/DIF); positive values indicate the
#'   specific threshold index.
#' @param cov Covariate being maximized (column index in pred_data).
#' @param samp_size Sample size in dataset.
#' @param num_responses_item Number of responses for item.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for categorical item likelihood
#' (to use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_categorical <-
  function(parm,
           p_item,
           etable_item,
           theta,
           pred_data,
           thr,
           cov,
           samp_size,
           num_responses_item,
           num_items,
           num_quad) {

    if(parm == "c0"){
      eta_d <- matrix(1, nrow = samp_size, ncol = num_quad)
    } else if(parm == "a0"){
      eta_d <- t(matrix(theta,
                        ncol=samp_size,
                        nrow=num_quad))
    } else if(parm == "c1"){
      eta_d <- matrix(pred_data[,cov],
                      ncol = num_quad,
                      nrow = samp_size)
    } else if(parm == "a1"){
      eta_d <- matrix(pred_data[,cov],
                      ncol = num_quad,
                      nrow = samp_size)*t(matrix(theta,
                                                 ncol=samp_size,
                                                 nrow=num_quad))
    }

    # Compute cumulative traceline P*(Y >= k | theta_q, x_i) for all thresholds.
    # Returns a list indexed by threshold, each element a samp_size x num_quad matrix.
    cum_traceline <- cumulative_traceline_pts(p_item,
                                              theta,
                                              pred_data,
                                              samp_size,
                                              num_responses_item,
                                              num_quad)

    # Non-threshold derivatives (slope and DIF parameters affect all categories).
    # When thr < 0, we are differentiating w.r.t. a0, c1, or a1 (not a threshold).
    if(thr < 0){
      # Initialize with contributions from the first and last response categories.
      # Category 1: P(Y=1) = 1 - P*(Y >= 2) = 1 - cum_traceline[[1]]
      #   d(P(Y=1))/d(eta) = -P*_1 * (1 - P*_1)
      d1 <- eta_d*(-etable_item[[1]]*cum_traceline[[1]] +
                     etable_item[[num_responses_item]]*(
                       1 - cum_traceline[[num_responses_item-1]]))
      d2 <- eta_d**2*(-etable_item[[1]]*(cum_traceline[[1]]*(
        1-cum_traceline[[1]])) +
          etable_item[[num_responses_item]]*(
            -cum_traceline[[num_responses_item-1]]*(
              1-cum_traceline[[num_responses_item-1]])))

      # Add contributions from intermediate response categories (2 through K-1).
      # For category k: P(Y=k) = P*(Y >= k) - P*(Y >= k+1)
      #   d(P(Y=k))/d(eta) involves (1-P*_k) and P*_{k-1} terms.
      for(i in 2:(num_responses_item-1)){

        # Skip intermediate derivative calculations for constrained theshold.
        d1 <- d1 + eta_d*etable_item[[i]]*((1-cum_traceline[[i]]) -
                                             cum_traceline[[i-1]])
        d2 <- d2 + eta_d**2*etable_item[[i]]*(cum_traceline[[i-1]]**2 +
                                                cum_traceline[[i]]**2 -
                                                cum_traceline[[i-1]] -
                                                cum_traceline[[i]])
      }

      # Sum over all persons and quadrature points.
      d1 <- sum(d1, na.rm = TRUE)
      d2 <- sum(d2, na.rm = TRUE)

      # Threshold derivatives (only affect two adjacent categories).
      # When thr >= 1, the threshold parameter only appears in the boundary
      # between categories thr and thr+1.
    } else {
      # cat_traceline = P(Y = thr+1), the category probability for the
      # category just above the threshold.
      if(thr < (num_responses_item-1)) {
        cat_traceline <- (cum_traceline[[thr]] - cum_traceline[[thr+1]])
      } else{
        cat_traceline <- cum_traceline[[thr]]
      }
      # First derivative for threshold thr: involves ratio of P**(1-P*)
      # to the category probabilities above and below the threshold.
      d1 <-
        sum(-etable_item[[thr]]*cum_traceline[[thr]]*(
          1 - cum_traceline[[thr]]
        ) / (cum_traceline[[thr-1]] - cum_traceline[[thr]]), na.rm = TRUE) +
        sum(etable_item[[thr+1]]*cum_traceline[[thr]]*(
          1 - cum_traceline[[thr]]
        ) / cat_traceline, na.rm = TRUE)
      # Second derivative for threshold thr: more complex expression involving
      # the squared cumulative probability terms and their products.
      d2 <- sum(etable_item[[thr]] /
                  (cum_traceline[[thr-1]] -
                     cum_traceline[[thr]])*(
                       cum_traceline[[thr]]*(1 - cum_traceline[[thr]])**2 -
                         cum_traceline[[thr]]**2*(1 - cum_traceline[[thr]]) +
                         cum_traceline[[thr]]**2*(1 - cum_traceline[[thr]])**2 /
                         (cum_traceline[[thr-1]] - cum_traceline[[thr]])
                       ), na.rm = TRUE) -
        sum(etable_item[[thr+1]] / cat_traceline*(
          cum_traceline[[thr]]*(1 - cum_traceline[[thr]])**2 -
            cum_traceline[[thr]]**2*(1-cum_traceline[[thr]]) -
            cum_traceline[[thr]]**2*(1-cum_traceline[[thr]])**2 /
            cat_traceline
          ), na.rm = TRUE)

    }

    dlist <- list(d1,d2)

  }

#' Partial derivatives for ordinal items using proxy data.
#'
#' Proxy variant of \code{d_categorical}. Uses observed proxy scores instead
#' of quadrature to approximate the latent variable. Since there is no E-table,
#' response-category indicators (d1_base) are derived from observed responses
#' (item_data_current) rather than posterior weights.
#'
#' @param parm Item parameter being maximized (one of "c0", "a0", "c1", "a1").
#' @param p_item Vector of item parameters.
#' @param prox_data Vector of observed proxy scores (samp_size x 1).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param item_data_current Vector of current item responses (1 through
#'   num_responses_item).
#' @param thr Threshold value being maximized. Negative = non-threshold
#'   parameter; positive = threshold index.
#' @param cov Covariate being maximized.
#' @param samp_size Sample size in dataset.
#' @param num_responses_item Number of responses for item.
#' @param num_items Number of items in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for categorical item likelihood
#' (to use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_categorical_proxy <-
  function(parm,
           p_item,
           prox_data,
           pred_data,
           item_data_current,
           thr,
           cov,
           samp_size,
           num_responses_item,
           num_items) {

    if(parm == "c0"){
      eta_d <- matrix(1, nrow = samp_size, ncol = 1)
    } else if(parm == "a0"){
      eta_d <- prox_data
    } else if(parm == "c1"){
      eta_d <- matrix(pred_data[,cov],
                      ncol = 1,
                      nrow = samp_size)
    } else if(parm == "a1"){
      eta_d <- matrix(pred_data[,cov],
                      ncol = 1,
                      nrow = samp_size)*prox_data
    }

    # Compute cumulative traceline at proxy scores (one value per person per threshold).
    cum_traceline <- cumulative_traceline_pts_proxy(p_item,
                                                    prox_data,
                                                    pred_data,
                                                    samp_size,
                                                    num_responses_item)

    # Build response-category indicator vectors: d1_base[[k]] = 1 if person i
    # responded in category k, 0 otherwise. These replace the E-table weighting
    # used in the quadrature-based version.
    d1_base <- lapply(1:num_responses_item, function(x) matrix(1, nrow = samp_size, ncol = 1))
    for(resp in 1:num_responses_item) {
      d1_base[[resp]][!(item_data_current == resp)] <- 0
    }

    # Non-threshold derivatives.
    if(thr < 0){
      d1 <- eta_d*(-d1_base[[1]]*cum_traceline[[1]] +
                     d1_base[[num_responses_item]]*(
                       1 - cum_traceline[[num_responses_item-1]]))
      d2 <- eta_d**2*(-d1_base[[1]]*(cum_traceline[[1]]*(
        1-cum_traceline[[1]])) +
          d1_base[[num_responses_item]]*(
            -cum_traceline[[num_responses_item-1]]*(
              1-cum_traceline[[num_responses_item-1]])))

      for(i in 2:(num_responses_item-1)){

        # Skip intermediate derivative calculations for constrained theshold.
        d1 <- d1 + eta_d*d1_base[[i]]*((1-cum_traceline[[i]]) -
                                             cum_traceline[[i-1]])
        d2 <- d2 + eta_d**2*d1_base[[i]]*(cum_traceline[[i-1]]**2 +
                                                cum_traceline[[i]]**2 -
                                                cum_traceline[[i-1]] -
                                                cum_traceline[[i]])
      }


      d1 <- sum(d1, na.rm = TRUE)
      d2 <- sum(d2, na.rm = TRUE)

      # Threshold derivatives.
    } else {
      if(thr < (num_responses_item-1)) {
        cat_traceline <- (cum_traceline[[thr]] - cum_traceline[[thr+1]])
      } else{
        cat_traceline <- cum_traceline[[thr]]
      }
      d1 <-
        sum(-d1_base[[thr]]*cum_traceline[[thr]]*(
          1 - cum_traceline[[thr]]
        ) / (cum_traceline[[thr-1]] - cum_traceline[[thr]]), na.rm = TRUE) +
        sum(d1_base[[thr+1]]*cum_traceline[[thr]]*(
          1 - cum_traceline[[thr]]
        ) / cat_traceline, na.rm = TRUE)
      d2 <- sum(d1_base[[thr]] /
                  (cum_traceline[[thr-1]] -
                     cum_traceline[[thr]])*(
                       cum_traceline[[thr]]*(1 - cum_traceline[[thr]])**2 -
                         cum_traceline[[thr]]**2*(1 - cum_traceline[[thr]]) +
                         cum_traceline[[thr]]**2*(1 - cum_traceline[[thr]])**2 /
                         (cum_traceline[[thr-1]] - cum_traceline[[thr]])
                     ), na.rm = TRUE) -
        sum(d1_base[[thr+1]] / cat_traceline*(
          cum_traceline[[thr]]*(1 - cum_traceline[[thr]])**2 -
            cum_traceline[[thr]]**2*(1-cum_traceline[[thr]]) -
            cum_traceline[[thr]]**2*(1-cum_traceline[[thr]])**2 /
            cat_traceline
        ), na.rm = TRUE)

    }

    dlist <- list(d1,d2)

  }

#' Partial derivatives for ordinal items by item-blocks (multivariate update).
#'
#' Computes the full gradient vector and Hessian matrix for all parameters
#' of a single ordinal (GRM) item simultaneously. This enables a multivariate
#' Newton-Raphson update. The parameter vector is ordered as:
#' [c0 (=1, constrained), thresh_2,...,thresh_{K-1}, a0, c1_1,...,c1_P,
#'  a1_1,...,a1_P] where K = num_responses_item.
#'
#' Non-threshold derivatives (for c0, a0, c1, a1) sum over all response
#' categories. Threshold derivatives are computed in a separate loop at the
#' end, since each threshold only affects two adjacent category probabilities.
#'
#' @param parm Item parameter being maximized (unused in block version but
#'   retained for interface consistency).
#' @param p_item Vector of item parameters.
#' @param etable E-table for item (samp_size x num_quad posterior weights).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param item_data_current Vector of current item responses (1 through
#'   num_responses_item).
#' @param samp_size Sample size in dataset.
#' @param num_responses_item Number of responses for item.
#' @param num_items Number of items in dataset.
#' @param num_predictors Number of predictors in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for categorical item likelihood
#' (to use with multivariate Newton-Raphson)
#'
#' @keywords internal
#'
d_categorical_itemblock <-
  function(parm,
           p_item,
           etable,
           theta,
           pred_data,
           item_data_current,
           samp_size,
           num_responses_item,
           num_items,
           num_predictors,
           num_quad) {

    # Make space for first and second derivatives.
    d1 <- matrix(0,nrow=length(p_item),ncol=1)
    d2 <- matrix(0,nrow=length(p_item),ncol=length(p_item))

    # First derivative for linear predictor w.r.t. theta.
    eta_d_a0 <- t(matrix(theta,
                         ncol=samp_size,
                         nrow=num_quad))

    # Get item response function.
    cum_traceline <- cumulative_traceline_pts(p_item,
                                              theta,
                                              pred_data,
                                              samp_size,
                                              num_responses_item,
                                              num_quad)

    # Get posterior probabilities for each response.
    etable_item <- lapply(1:num_responses_item, function(x) etable)

    # Obtain E-tables for each response category.
    for(resp in 1:num_responses_item) {
      etable_item[[resp]][which(
        !(item_data_current == resp)), ] <- 0
    }


    # Calculate first and second base derivatives for non-threshold parameters.
    # These are the derivative contributions shared by c0, a0, c1, and a1,
    # accumulated over all response categories.
    d1_base <- d2_base <- 0
    for(resp in 1:num_responses_item) {

      if(resp == 1) {
        # Lowest category: P(Y=1) = 1 - P*(Y>=2).
        # d(P(Y=1))/d(eta) = -P*_1*(1-P*_1).
        d1_base <- d1_base +
          -etable_item[[1]]*cum_traceline[[1]]
        d2_base <- d2_base +
          -etable_item[[1]]*(cum_traceline[[1]]*(1-cum_traceline[[1]]))
      } else if(resp == num_responses_item) {
        # Highest category: P(Y=K) = P*(Y>=K).
        # d(P(Y=K))/d(eta) = P*_{K-1}*(1-P*_{K-1}).
        d1_base <- d1_base +
          etable_item[[num_responses_item]]*(
            1-cum_traceline[[num_responses_item-1]])
        d2_base <- d2_base +
          etable_item[[num_responses_item]]*(
            -cum_traceline[[num_responses_item-1]]*(
              1-cum_traceline[[num_responses_item-1]]))
      } else {
        # Intermediate categories: P(Y=k) = P*(Y>=k) - P*(Y>=k+1).
        d1_base <- d1_base +
          etable_item[[resp]]*((1-cum_traceline[[resp]]) -
                                 cum_traceline[[resp-1]])
        d2_base <- d2_base +
          etable_item[[resp]]*(cum_traceline[[resp-1]]**2 +
                              cum_traceline[[resp]]**2 -
                              cum_traceline[[resp-1]] -
                              cum_traceline[[resp]])
      }

    }

    # First and second derivative for c0.
    d1[1,1] <- sum(d1_base, na.rm = TRUE) #d1
    d2[1,1] <- sum(d2_base, na.rm = TRUE) #d2

    # First and second derivative for a0.
    d1[num_responses_item,1] <-
      sum(eta_d_a0*d1_base, na.rm = TRUE) #d1
    d2[num_responses_item,num_responses_item] <-
      sum(eta_d_a0**2*d2_base, na.rm = TRUE) #d2

    # Cross derivative for c0 and a0.
    d2[num_responses_item,1] <- sum(eta_d_a0*d2_base, na.rm = TRUE) #d2

    # Cycle through predictors (outer cycle).
    for(cov in 1:num_predictors) {

      # First derivative for linear predictor w.r.t. covariate.
      cov_matrix <- matrix(pred_data[,cov],
                           ncol = num_quad,
                           nrow = samp_size)

      # First and second derivatives for c1.
      d1[num_responses_item+cov,1] <-
        sum(cov_matrix*d1_base,
            na.rm = TRUE) #d1
      d2[num_responses_item+cov,num_responses_item+cov] <-
        sum(cov_matrix**2*d2_base,
            na.rm = TRUE) #d2

      # First and second derivatives for a1.
      d1[num_responses_item+num_predictors+cov,1] <-
        sum(cov_matrix*eta_d_a0*d1_base,
            na.rm = TRUE) #d1
      d2[num_responses_item+num_predictors+cov,
         num_responses_item+num_predictors+cov] <-
        sum((cov_matrix*eta_d_a0)**2*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for c0 and c1.
      d2[num_responses_item+cov,1] <-
        sum(cov_matrix*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for c0 and a1, as well as a0 and c1.
      d2[num_responses_item+num_predictors+cov,1] <-
        d2[num_responses_item+cov,num_responses_item] <-
        sum(cov_matrix*eta_d_a0*d2_base,
            na.rm = TRUE) #d2

      # Cross derivatives for a0 and a1.
      d2[num_responses_item+num_predictors+cov,num_responses_item] <-
        sum(cov_matrix*eta_d_a0**2*d2_base,
            na.rm = TRUE) #d2

      # Cycle through predictors (inner cycle).
      for(cov2 in 1:num_predictors) {


        if(cov == cov2) {

          # Cross derivatives with same predictor for c1 and a1.
          d2[num_responses_item+num_predictors+cov,num_responses_item+cov2] <-
            sum(cov_matrix**2*eta_d_a0*d2_base,
                na.rm = TRUE) #d2

        } else {

          # First derivatives for linear predictor w.r.t. second covariate.
          cov2_matrix <- matrix(pred_data[,cov2],
                                ncol = num_quad,
                                nrow = samp_size)

          # Cross derivatives with different predictor for c1 and a1.
          d2[num_responses_item+num_predictors+cov,num_responses_item+cov2] <-
            sum(cov_matrix*cov2_matrix*eta_d_a0*d2_base,
                na.rm = TRUE) #d2

          if(cov2 > 1 && cov < cov2) {

            # Cross derivatives with different predictor for c1 and c1.
            d2[num_responses_item+cov2,num_responses_item+cov] <-
              sum(cov_matrix*cov2_matrix*d2_base, #d2
                  na.rm = TRUE)

            # Cross derivatives with different predictor for a1 and a1.
            d2[num_responses_item+num_predictors+cov2,
               num_responses_item+num_predictors+cov] <-
              sum(cov_matrix*cov2_matrix*eta_d_a0**2*d2_base, #a1a1
                  na.rm = TRUE)
          }
        }

      }

    }

    # Threshold derivatives. Each threshold thr only affects two adjacent
    # category probabilities: P(Y=thr) and P(Y=thr+1). The base derivative
    # is P*_thr * (1 - P*_thr), the logistic derivative of the cumulative
    # probability at threshold thr.
    for(thr in 2:(num_responses_item-1)) {
      # d(P*_thr)/d(c0_thr) = P*_thr * (1 - P*_thr) -- logistic sigmoid derivative.
      d1_base_thr <- cum_traceline[[thr]]*(1-cum_traceline[[thr]])
      # Second base derivative: d2(P*_thr)/d(c0_thr)^2.
      d2_base_thr <- d1_base_thr*(1 - 2*cum_traceline[[thr]])

      cat_traceline1 <- cum_traceline[[thr-1]] - cum_traceline[[thr]]
      cat_traceline2 <- cum_traceline[[thr]]
      if(thr < (num_responses_item-1)) {
        cat_traceline2 <- cat_traceline2 - cum_traceline[[thr+1]]
      }

      d1[thr,1] <-
        sum(d1_base_thr*(-etable_item[[thr]] / cat_traceline1 +
                           etable_item[[thr+1]] / cat_traceline2),
            na.rm = TRUE)

      d2[thr,thr] <-
        sum(etable_item[[thr]] / cat_traceline1 *
              (d2_base_thr + d1_base_thr**2/cat_traceline1),
            na.rm = TRUE) -
        sum(etable_item[[thr+1]] / cat_traceline2 *
              (d2_base_thr - d1_base_thr**2/cat_traceline2),
            na.rm = TRUE)

      d2[thr,1] <-
        sum(etable_item[[thr]])


      # for(thr2 in 3:(num_responses_item-1))
      #
      # for(cov in 1:num_predictors) {
      #   # First derivative for linear predictor w.r.t. covariate.
      #   cov_matrix <- matrix(pred_data[,cov],
      #                        ncol = num_quad,
      #                        nrow = samp_size)
      # }

    }


    dlist <- list(d1,d2)

  }


###############################################################################
### SECTION 4: Gaussian (Continuous Item / CFA) Derivatives                ###
###############################################################################
#
# Continuous items are modeled with a Gaussian measurement model (CFA-like):
#
#   Y_i | theta ~ N(mu_i(theta), sigma_i^2)
#
# where the conditional mean is a linear function of the latent trait:
#   mu_i(theta) = (c0 + x'c1) + (a0 + x'a1) * theta
#
# and the residual variance is parameterized on the log scale:
#   sigma_i^2 = s0 * exp(x' s1)
#
# This ensures sigma^2 > 0 and allows DIF in the residual variance.
#
# The Gaussian log-likelihood for person i at quadrature point q is:
#   log f(y_i | theta_q) = -0.5*log(sigma_i^2) - 0.5*(y_i - mu_i)^2 / sigma_i^2
#
# Derivatives for mean parameters (c0, a0, c1, a1) involve:
#   d1 = eta_d / sigma^2 * (y - mu)     (residual/variance)
#   d2 = -eta_d^2 / sigma^2             (always negative)
#
# Derivatives for variance parameters (s0, s1) are more complex because
# sigma appears in both the normalizing constant and the quadratic form,
# and the log-link introduces additional chain rule factors.
#
###############################################################################

#' Partial derivatives for mean parameter of continuous items.
#'
#' Computes the gradient and Hessian diagonal element for a single mean
#' (location) parameter of a continuous (Gaussian) item. The derivative
#' involves the residual (y - mu) scaled by the inverse variance.
#'
#' @param parm Item parameter being maximized (one of "c0", "a0", "c1", "a1").
#' @param p_item Vector of item parameters.
#' @param etable E-table (samp_size x num_quad posterior weights).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param responses_item Vector of item responses (continuous).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param cov Covariate being maximized (column index in pred_data).
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for mean value of Gaussian item
#' likelihood (to use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_mu_gaussian <-
  function(parm,
           p_item,
           etable,
           theta,
           responses_item,
           pred_data,
           cov,
           samp_size,
           num_items,
           num_quad) {

  # Compute eta_d = d(mu)/d(parm), the derivative of the conditional mean
  # with respect to the parameter being optimized. Same pattern as Bernoulli.
  if(parm == "c0"){
    # d(mu)/d(c0) = 1.
    eta_d <- matrix(1, nrow = samp_size, ncol = num_quad)
  } else if(parm == "a0"){
    # d(mu)/d(a0) = theta.
    eta_d <- t(replicate(n=samp_size, theta))
  } else if(parm == "c1"){
    # d(mu)/d(c1_k) = x_k.
    eta_d <- matrix(rep(pred_data[,cov], num_quad),
                    ncol = num_quad,
                    nrow = samp_size)
  } else if(parm == "a1"){
    # d(mu)/d(a1_k) = x_k * theta.
    eta_d <- matrix(rep(pred_data[,cov], num_quad),
                    ncol = num_quad,
                    nrow = samp_size)*t(replicate(n=samp_size, theta))
  }


  # Get conditional mean mu_i(theta_q) and residual standard deviation sigma_i
  # for each person i at each quadrature point q.
  # mu_i(theta_q) = (c0 + x'c1) + (a0 + x'a1)*theta_q.
  mu <- sapply(theta,
              function(x) {
                (p_item[grep("c0",names(p_item),fixed=T)] +
                   pred_data %*%
                   p_item[grep("c1",names(p_item),fixed=T)]) +
                  (p_item[grep("a0",names(p_item),fixed=T)] +
                     pred_data %*%
                     p_item[grep("a1",names(p_item),fixed=T)])*x
                })
  # sigma_i = sqrt(s0 * exp(x' s1)), the person-specific residual SD.
  sigma <- sqrt(p_item[grep("s0",names(p_item))][1]*exp(
    pred_data %*% p_item[grep("s1",names(p_item))]
    ))


  # First derivative of Gaussian LL w.r.t. mean parameter:
  #   d1 = eta_d / sigma^2 * (y - mu)  (residual scaled by inverse variance).
  d1_trace <- t(sapply(1:samp_size,
                       function(x) {
                         eta_d[x,]/sigma[x]**2*(responses_item[x] - mu[x,])
                         }))
  # Second derivative: d2 = -eta_d^2 / sigma^2 (always negative).
  d2_trace <- t(sapply(1:samp_size,
                       function(x) {
                         -eta_d[x,]**2 / sigma[x]**2
                         }))

  # Weight by E-table and sum across persons and quadrature points.
  d1 <- sum(etable*d1_trace, na.rm = TRUE)
  d2 <- sum(etable*d2_trace, na.rm = TRUE)

  dlist <- list(d1,d2)

  }

#' Partial derivatives for mean parameter of continuous items with proxy data.
#'
#' Proxy variant of \code{d_mu_gaussian}. Uses observed proxy scores instead
#' of quadrature. The derivative formulas are identical but evaluated at the
#' proxy score rather than integrated over quadrature points.
#'
#' @param parm Item parameter being maximized (one of "c0", "a0", "c1", "a1").
#' @param p_item Vector of item parameters.
#' @param prox_data Vector of observed proxy scores (samp_size x 1).
#' @param responses_item Vector of item responses (continuous).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param cov Covariate being maximized.
#' @param samp_size Sample size in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for mean value of Gaussian item
#' likelihood (to use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_mu_gaussian_proxy <-
  function(parm,
           p_item,
           prox_data,
           responses_item,
           pred_data,
           cov,
           samp_size) {

    if(parm == "c0"){
      eta_d <- matrix(1, nrow = samp_size, ncol = 1)
    } else if(parm == "a0"){
      eta_d <- prox_data
    } else if(parm == "c1"){
      eta_d <- as.matrix(pred_data[,cov])
    } else if(parm == "a1"){
      eta_d <- as.matrix(pred_data[,cov]*prox_data)
    }


    # Get latent mean and variance vectors.
    mu <- (p_item[grep("c0",names(p_item),fixed=T)] +
                      pred_data %*%
                      p_item[grep("c1",names(p_item),fixed=T)]) +
                     (p_item[grep("a0",names(p_item),fixed=T)] +
                        pred_data %*%
                        p_item[grep("a1",names(p_item),fixed=T)])*prox_data
    sigma <- sqrt(p_item[grep("s0",names(p_item))][1]*exp(
      pred_data %*% p_item[grep("s1",names(p_item))]
    ))


    d1_trace <- t(sapply(1:samp_size,
                         function(x) {
                           eta_d[x,]/sigma[x]**2*(responses_item[x] - mu[x,])
                         }))
    d2_trace <- t(sapply(1:samp_size,
                         function(x) {
                           -eta_d[x,]**2 / sigma[x]**2
                         }))

    d1 <- sum(d1_trace, na.rm = TRUE)
    d2 <- sum(d2_trace, na.rm = TRUE)

    dlist <- list(d1,d2)

  }

#' Partial derivatives for variance parameter of continuous items.
#'
#' Computes the gradient and Hessian diagonal element for a single variance
#' (scale) parameter of a continuous (Gaussian) item. The residual variance
#' is parameterized as sigma^2 = s0 * exp(x' s1), so the chain rule through
#' the exp link and the sqrt introduces additional complexity compared to
#' the mean parameter derivatives.
#'
#' For s0: eta_d1 involves exp(x's1) / (2*sigma), i.e., d(sigma)/d(s0).
#' For s1: eta_d1 involves sigma * x_k / 2, i.e., d(sigma)/d(s1_k).
#'
#' @param parm Item parameter being maximized (one of "s0" or "s1").
#' @param p_item Vector of item parameters.
#' @param etable E-table for impact (samp_size x num_quad posterior weights).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param responses_item Vector of item responses (continuous).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param cov Covariate being maximized (column index in pred_data, for "s1").
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#'
#' @return a \code{"list"} of first and second partial derivatives for variance value of Gaussian
#' item likelihood (to use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_sigma_gaussian <-
  function(parm,
           p_item,
           etable,
           theta,
           responses_item,
           pred_data,
           cov,
           samp_size,
           num_items,
           num_quad) {

  # Compute sigma_i = sqrt(s0 * exp(x' s1)) and mu_i(theta_q).
  sigma <- sqrt(p_item[grep("s0",names(p_item))][1]*exp(
    pred_data %*% p_item[grep("s1",names(p_item))]))
  mu <- sapply(theta,
              function(x) {
                (p_item[grep("c0",names(p_item),fixed=T)] +
                   pred_data %*%
                   p_item[grep("c1",names(p_item),fixed=T)]) +
                  (p_item[grep("a0",names(p_item),fixed=T)] +
                     pred_data %*%
                     p_item[grep("a1",names(p_item),fixed=T)])*x
                })

  # Compute chain rule factors for the variance parameterization.
  # eta_d1 = d(sigma)/d(parm), eta_d2 = d2(sigma)/d(parm)^2.
  if(parm == "s0") {
    # d(sigma)/d(s0) = exp(x's1) / (2*sigma).
    eta_d1 <- sapply(1:samp_size,
                     function(x) {
                       exp(pred_data[x,] %*%
                             p_item[grep("s1",names(p_item))]) / (2*sigma[x])
                       })
    # d2(sigma)/d(s0)^2 = -exp(x's1)^2 / (4*sigma^3).
    eta_d2 <- sapply(1:samp_size,
                     function(x) {
                       -exp(pred_data[x,] %*%
                              p_item[grep("s1",names(p_item))])**2 /
                         (4*sigma[x]**3)
                       })
  } else if(parm == "s1") {
    # d(sigma)/d(s1_k) = sigma * x_k / 2 (chain rule through exp link).
    eta_d1 <- sapply(1:samp_size,
                     function(x) {
                       sigma[x]*pred_data[x,cov] / 2
                       })
    # d2(sigma)/d(s1_k)^2 = sigma * x_k^2 / 4.
    eta_d2 <- sapply(1:samp_size,
                     function(x) {
                       sigma[x]*pred_data[x,cov]**2 / 4
                       })
  }


  # First derivative of Gaussian LL w.r.t. sigma parameter:
  #   d1 = eta_d1 * [(y-mu)^2 / sigma^3 - 1/sigma]
  # This comes from d/d(sigma)[-0.5*log(sigma^2) - 0.5*(y-mu)^2/sigma^2].
  d1_trace <- t(sapply(1:samp_size,
                       function(x) {
                         eta_d1[x]*((responses_item[x]-mu[x,])**2 /
                                      sigma[x]**3 -
                                      1/sigma[x])
                         }))

  # Second derivative depends on the parameterization:
  if(parm == "s0") {
    # For s0: full second derivative including both the (d(sigma)/d(s0))^2
    # term and the (d2(sigma)/d(s0)^2) term.
    d2_trace <- t(sapply(1:samp_size,
                         function(x) {
                           eta_d1[x]**2*(1 / sigma[x]**2 -
                                           3*(responses_item[x] - mu[x,])**2 /
                                           sigma[x]**4) +
                             eta_d2[x]*((responses_item[x] - mu[x,])**2 /
                                          sigma[x]**3 - 1/sigma[x])
                           }))
  } else if(parm == "s1") {
    # For s1: simplified form using the log-link structure.
    d2_trace <- t(sapply(1:samp_size,
                         function(x) {
                           -2*eta_d2[x]*(sigma[x]**(-3)*(responses_item[x] -
                                                           mu[x,])**2)
                           }))
  }

  # Weight by E-table and sum.
  d1 <- sum(etable*d1_trace, na.rm = TRUE)
  d2 <- sum(etable*d2_trace, na.rm = TRUE)

  dlist <- list(d1,d2)

}

#' Partial derivatives for variance parameter of continuous items with proxy data.
#'
#' Proxy variant of \code{d_sigma_gaussian}. Uses observed proxy scores
#' instead of quadrature. Same derivative formulas evaluated at the proxy
#' score rather than integrated over quadrature points.
#'
#' @param parm Item parameter being maximized (one of "s0" or "s1").
#' @param p_item Vector of item parameters.
#' @param prox_data Vector of observed proxy scores (samp_size x 1).
#' @param responses_item Vector of item responses (continuous).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param cov Covariate being maximized.
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for variance value of Gaussian
#' item likelihood (to use with coordinate descent and univariate Newton-Raphson)
#'
#' @keywords internal
#'
d_sigma_gaussian_proxy <-
  function(parm,
           p_item,
           prox_data,
           responses_item,
           pred_data,
           cov,
           samp_size,
           num_items) {

    sigma <- sqrt(p_item[grep("s0",names(p_item))][1]*exp(
      pred_data %*% p_item[grep("s1",names(p_item))]))
    mu <- (p_item[grep("c0",names(p_item),fixed=T)] +
                      pred_data %*%
                      p_item[grep("c1",names(p_item),fixed=T)]) +
                     (p_item[grep("a0",names(p_item),fixed=T)] +
                        pred_data %*%
                        p_item[grep("a1",names(p_item),fixed=T)])*prox_data

    if(parm == "s0") {
      eta_d1 <- sapply(1:samp_size,
                       function(x) {
                         exp(pred_data[x,] %*%
                               p_item[grep("s1",names(p_item))]) / (2*sigma[x])
                       })
      eta_d2 <- sapply(1:samp_size,
                       function(x) {
                         -exp(pred_data[x,] %*%
                                p_item[grep("s1",names(p_item))])**2 /
                           (4*sigma[x]**3)
                       })
    } else if(parm == "s1") {
      eta_d1 <- sapply(1:samp_size,
                       function(x) {
                         sigma[x]*pred_data[x,cov] / 2
                       })
      eta_d2 <- sapply(1:samp_size,
                       function(x) {
                         sigma[x]*pred_data[x,cov]**2 / 4
                       })
    }


    d1_trace <- t(sapply(1:samp_size,
                         function(x) {
                           eta_d1[x]*((responses_item[x]-mu[x,])**2 /
                                        sigma[x]**3 -
                                        1/sigma[x])
                         }))

    if(parm == "s0") {
      d2_trace <- t(sapply(1:samp_size,
                           function(x) {
                             eta_d1[x]**2*(1 / sigma[x]**2 -
                                             3*(responses_item[x] - mu[x,])**2 /
                                             sigma[x]**4) +
                               eta_d2[x]*((responses_item[x] - mu[x,])**2 /
                                            sigma[x]**3 - 1/sigma[x])
                           }))
    } else if(parm == "s1") {
      d2_trace <- t(sapply(1:samp_size,
                           function(x) {
                             -2*eta_d2[x]*(sigma[x]**(-3)*(responses_item[x] -
                                                             mu[x,])**2)
                           }))
    }

    d1 <- sum(d1_trace, na.rm = TRUE)
    d2 <- sum(d2_trace, na.rm = TRUE)

    dlist <- list(d1,d2)

  }

#' Partial derivatives for continuous items by item-blocks (multivariate update).
#'
#' Computes the full gradient vector and Hessian matrix for all parameters
#' of a single continuous (Gaussian) item simultaneously. The parameter
#' vector is ordered as:
#' [c0, a0, c1_1,...,c1_P, a1_1,...,a1_P, s0, s1_1,...,s1_P]
#' where P = num_predictors.
#'
#' This function handles three groups of parameters (mean, slope, variance)
#' and all their cross-derivatives. The "d1_base_mu" and "d1_base_sigma"
#' quantities factor the derivative into shared components multiplied by
#' parameter-specific eta_d terms.
#'
#' @param p_item Vector of item parameters.
#' @param etable E-table (samp_size x num_quad posterior weights).
#' @param theta Matrix of adaptive theta values (quadrature points).
#' @param responses_item Vector of item responses (continuous).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#' @param num_predictors Number of predictors in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for all Gaussian item
#' parameters (to use with multivariate Newton-Raphson)
#'
#' @keywords internal
#'
d_gaussian_itemblock <-
  function(p_item,
           etable,
           theta,
           responses_item,
           pred_data,
           samp_size,
           num_items,
           num_quad,
           num_predictors) {

    # Make space for first and second derivatives.
    # The parameter vector has length 2 + 3*num_predictors + 1:
    # [c0, a0, c1_1..c1_P, a1_1..a1_P, s0, s1_1..s1_P].
    d1 <- matrix(0,nrow=length(p_item),ncol=1)
    d2 <- matrix(0,nrow=length(p_item),ncol=length(p_item))


    # Get conditional mean mu_i(theta_q) and residual SD sigma_i.
    # Parameters are accessed by index (not name) for efficiency.
    # p_item[1] = c0, p_item[2] = a0,
    # p_item[3:(2+P)] = c1, p_item[(3+P):(2+2P)] = a1,
    # p_item[(3+2P)] = s0, p_item[(4+2P):(3+3P)] = s1.
    mu <- sapply(theta,
                 function(x) {
                   (p_item[1] +
                      pred_data %*%
                      p_item[3:(2+num_predictors)]) +
                     (p_item[2] +
                        pred_data %*%
                        p_item[(3+num_predictors):(2+num_predictors*2)])*x
                 })
    sigma <- sqrt(p_item[(3+num_predictors*2)]*exp(
      pred_data %*% p_item[(4+num_predictors*2):(3+num_predictors*3)]
    ))

    # Pre-compute eta_d terms for each parameter type.
    # eta_d_mu_base = d(mu)/d(c0) = 1 for all (person, quad) combinations.
    eta_d_mu_base <- matrix(1, nrow = samp_size, ncol = num_quad)
    # eta_d_mu_a0 = d(mu)/d(a0) = theta_q.
    eta_d_mu_a0 <- t(matrix(theta,
                            ncol=samp_size,
                            nrow=num_quad))
    # eta_d_sigma0_base = d(sigma)/d(s0) = exp(x's1) / (2*sigma).
    eta_d_sigma0_base <-
      as.vector(exp(pred_data %*% p_item[(4+num_predictors*2):(3+num_predictors*3)]) / (2*sigma))
    # eta_d_sigma1_base = d(sigma)/d(s1_k) factor = sigma / 2.
    eta_d_sigma1_base <- as.vector(sigma / 2)

    # Pre-compute base derivative quantities shared across parameters.
    # These are already weighted by the E-table where appropriate.

    # d1_base_mu: base gradient for mean parameters = (y-mu)/sigma^2 * E-table.
    d1_base_mu <-
      sapply(1:num_quad,
             function(x) {
               1/sigma**2*(responses_item - mu[,x])
               })*etable
    # d2_base_mu: base Hessian for mean parameters = -1/sigma^2 * E-table.
    d2_base_mu <- sapply(1:num_quad,
                         function(x) {
                           - 1 / sigma**2 * etable[,x]
                         })
    # d1_base_sigma: base gradient for variance parameters =
    #   [(y-mu)^2/sigma^3 - 1/sigma] * E-table.
    d1_base_sigma <- sapply(1:num_quad,
                            function(x) {
                              ((responses_item-mu[,x])**2 /
                                 sigma**3 -
                                 1/sigma)
                            })*etable
    # d2_base_sigma0: base Hessian for s0 (includes both first- and second-order
    # chain rule terms for d(sigma)/d(s0)).
    d2_base_sigma0 <- sapply(1:num_quad,
                             function(x) {
                               eta_d_sigma0_base*(1 / sigma**2 -
                                                    3*(responses_item - mu[,x])**2 /
                                                    sigma**4) +
                                 (-1 / (2*sigma**2))*((responses_item - mu[,x])**2 /
                                                        sigma**3 - 1/sigma)
                             })*etable
    # d2_base_sigma1: base Hessian for s1 (simplified using log-link structure).
    d2_base_sigma1 <- sapply(1:num_quad,
                             function(x) {
                               -2*eta_d_sigma1_base*(sigma**(-3)*(responses_item -
                                                                    mu[,x])**2)
                             })*etable

    # First and second derivative for c0.
    d1[1,1] <- sum(d1_base_mu, na.rm = TRUE) #d1
    d2[1,1] <- sum(d2_base_mu, na.rm = TRUE) #d2

    # First and second derivative for a0.
    d1[2,1] <- sum(eta_d_mu_a0*d1_base_mu, na.rm = TRUE) #d1
    d2[2,2] <- sum(eta_d_mu_a0**2*d2_base_mu, na.rm = TRUE) #d2

    # First and second derivative for s0.
    d1[(3+num_predictors*2),1] <- sum(eta_d_sigma0_base*d1_base_sigma, na.rm = TRUE) #d1
    d2[(3+num_predictors*2),(3+num_predictors*2)] <-
      sum(eta_d_sigma0_base*d2_base_sigma0, na.rm = TRUE) #d2

    # Cross derivative for c0 and a0.
    d2[2,1] <- sum(eta_d_mu_a0*d2_base_mu, na.rm = TRUE) #d2

    # Cross derivative for c0 and s0.
    d2[(3+num_predictors*2),1] <- sum(-eta_d_sigma0_base*d1_base_mu*eta_d_sigma1_base**(-1),
                                      na.rm = TRUE) #d2

    # Cross derivative for a0 and s0.
    d2[(3+num_predictors*2),2] <-
      sum(-eta_d_sigma0_base*eta_d_mu_a0*d1_base_mu*eta_d_sigma1_base**(-1),
          na.rm = TRUE) #d2

    # Cycle through predictors (outer cycle).
    for(cov in 1:num_predictors) {

      # First derivative for linear predictor w.r.t. covariate.
      cov_matrix <- pred_data[,cov]

      # First and second derivatives for c1.
      d1[2+cov,1] <-
        sum(cov_matrix*d1_base_mu,
            na.rm = TRUE) #d1
      d2[2+cov,2+cov] <-
        sum(cov_matrix**2*d2_base_mu,
            na.rm = TRUE) #d2

      # First and second derivatives for a1.
      d1[2+num_predictors+cov,1] <-
        sum(cov_matrix*eta_d_mu_a0*d1_base_mu,
            na.rm = TRUE) #d1
      d2[2+num_predictors+cov,2+num_predictors+cov] <-
        sum((cov_matrix*eta_d_mu_a0)**2*d2_base_mu,
            na.rm = TRUE) #d2

      # First and second derivatives for s1.
      d1[(3+num_predictors*2+cov),1] <-
        sum(cov_matrix*eta_d_sigma1_base*d1_base_sigma,
            na.rm = TRUE) #d1
      d2[(3+num_predictors*2+cov),(3+num_predictors*2+cov)] <-
        sum(cov_matrix**2/2*d2_base_sigma1,
            na.rm = TRUE) #d2

      # # Cross derivatives for c0 and c1.
      d2[2+cov,1] <- sum(cov_matrix*d2_base_mu, na.rm = TRUE) #d2

      # # Cross derivatives for c0 and a1, as well as a0 and c1.
      d2[2+num_predictors+cov,1] <- d2[2+cov,2] <-
        sum(cov_matrix*eta_d_mu_a0*d2_base_mu,
            na.rm = TRUE) #d2

      # # Cross derivatives for a0 and a1.
      d2[2+num_predictors+cov,2] <-
        sum(cov_matrix*eta_d_mu_a0**2*d2_base_mu,
            na.rm = TRUE) #d2

      # # Cross derivatives for c0 and s1
      d2[(3+num_predictors*2+cov),1] <- sum(-cov_matrix*d1_base_mu,
                                             na.rm = TRUE) #d2

      # # Cross derivatives for a0 and s1
      d2[(3+num_predictors*2+cov),2] <- sum(-cov_matrix*eta_d_mu_a0*d1_base_mu,
                                            na.rm = TRUE) #d2

      # # Cross derivatives for s0 and c1
      d2[(3+num_predictors*2),2+cov] <-
        sum(-cov_matrix*eta_d_sigma0_base*d1_base_mu*eta_d_sigma1_base**(-1),
            na.rm = TRUE) #d2

      # # Cross derivatives for s0 and a1
      d2[(3+num_predictors*2),2+num_predictors+cov] <-
        sum(-cov_matrix*eta_d_mu_a0*eta_d_sigma0_base*d1_base_mu*eta_d_sigma1_base**(-1),
            na.rm = TRUE) #d2

      # # Cross derivatives for s0 and s1
      d2[(3+num_predictors*2+cov),(3+num_predictors*2)] <-
        sum(cov_matrix*eta_d_sigma0_base*d2_base_sigma1*(1/eta_d_sigma1_base/2),
            na.rm = TRUE) #d2

      # # Cycle through predictors (inner cycle).
      for(cov2 in 1:num_predictors) {


        if(cov == cov2) {

          # Cross derivatives with same predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix**2*eta_d_mu_a0*d2_base_mu,
                na.rm = TRUE) #d2

          # Cross derivatives with same predictor for c1 and s1.
          d2[(3+num_predictors*2+cov),2+cov2] <- sum(-cov_matrix**2*d1_base_mu,
                                                     na.rm = TRUE) #d2

          # Cross derivatives with same predictor for a1 and s1.
          d2[(3+num_predictors*2+cov),2+num_predictors+cov2] <-
            sum(-cov_matrix**2*eta_d_mu_a0*d1_base_mu,
                na.rm = TRUE) #d2

        } else {

          # First derivatives for linear predictor w.r.t. second covariate.
          cov2_matrix <- pred_data[,cov2]

          # Cross derivatives with different predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix*cov2_matrix*eta_d_mu_a0*d2_base_mu,
                na.rm = TRUE) #d2

          # Cross derivatives with different predictor for c1 and s1.
          d2[(3+num_predictors*2+cov),2+cov2] <-
            sum(-cov_matrix*cov2_matrix*d1_base_mu,
                na.rm = TRUE) #d2

          # Cross derivatives with different predictor for a1 and s1.
          d2[(3+num_predictors*2+cov),2+num_predictors+cov2] <-
            sum(-cov_matrix*cov2_matrix*eta_d_mu_a0*d1_base_mu,
                na.rm = TRUE) #d2

          if(cov2 > 1 && cov < cov2) {

            # Cross derivatives with different predictor for c1 and c1.
            d2[2+cov2,2+cov] <-
              sum(cov_matrix*cov2_matrix*d2_base_mu, #d2
                  na.rm = TRUE)

            # Cross derivatives with different predictor for a1 and a1.
            d2[2+num_predictors+cov2,2+num_predictors+cov] <-
              sum(cov_matrix*cov2_matrix*eta_d_mu_a0**2*d2_base_mu, #a1a1
                  na.rm = TRUE)

            # Cross derivatives with different predictor for s1 and s1.
            d2[(3+num_predictors*2+cov2),(3+num_predictors*2+cov)] <-
              sum(cov_matrix*cov2_matrix/2*d2_base_sigma1,
                  na.rm = TRUE) #d2
          }
        }

      }

    }


    dlist <- list(d1,d2)

  }

#' Partial derivatives for continuous items by item-blocks using proxy data.
#'
#' Proxy variant of \code{d_gaussian_itemblock}. Computes the full gradient
#' vector and Hessian matrix for all parameters of a single continuous item
#' using observed proxy scores. Same structure and parameter ordering as
#' \code{d_gaussian_itemblock} but without quadrature loops.
#'
#' @param p_item Vector of item parameters.
#' @param prox_data Vector of observed proxy scores (samp_size x 1).
#' @param responses_item Vector of item responses (continuous).
#' @param pred_data Matrix or dataframe of DIF and/or impact predictors.
#' @param samp_size Sample size in dataset.
#' @param num_items Number of items in dataset.
#' @param num_quad Number of quadrature points used for approximating the
#' latent variable.
#' @param num_predictors Number of predictors in dataset.
#'
#' @return a \code{"list"} of first and second partial derivatives for all Gaussian item
#' parameters (to use with multivariate Newton-Raphson and observed proxy scores)
#'
#' @keywords internal
#'
d_gaussian_itemblock_proxy <-
  function(p_item,
           prox_data,
           responses_item,
           pred_data,
           samp_size,
           num_items,
           num_quad,
           num_predictors) {


    # Make space for first and second derivatives.
    d1 <- matrix(0,nrow=length(p_item),ncol=1)
    d2 <- matrix(0,nrow=length(p_item),ncol=length(p_item))


    # Get conditional mean and residual SD (proxy version: no quadrature loop).
    # mu_i = (c0 + x'c1) + (a0 + x'a1) * prox_data_i.
    mu <- (p_item[1] + pred_data %*% p_item[3:(2+num_predictors)]) +
      (p_item[2] + pred_data %*% p_item[(3+num_predictors):(2+num_predictors*2)])*prox_data
    # sigma_i = sqrt(s0 * exp(x' s1)).
    sigma <- sqrt(p_item[(3+num_predictors*2)]*exp(
      pred_data %*% p_item[(4+num_predictors*2):(3+num_predictors*3)]
    ))

    # Pre-compute eta_d terms (proxy version: samp_size x 1 instead of samp_size x num_quad).
    eta_d_mu_base <- matrix(1, nrow = samp_size, ncol = 1)
    eta_d_mu_a0 <- prox_data
    # d(sigma)/d(s0) = exp(x's1) / (2*sigma).
    eta_d_sigma0_base <-
      exp(pred_data %*% p_item[(4+num_predictors*2):(3+num_predictors*3)]) / (2*sigma)
    # d(sigma)/d(s1_k) base factor = sigma / 2.
    eta_d_sigma1_base <- sigma / 2

    # Pre-compute base derivative quantities (proxy version, no E-table weighting).
    # Base gradient for mean parameters: (y - mu) / sigma^2.
    d1_base_mu <- 1/sigma**2*(responses_item - mu)
    # Base Hessian for mean parameters: -1 / sigma^2.
    d2_base_mu <- - 1 / sigma**2
    # Base gradient for variance parameters: [(y-mu)^2/sigma^3 - 1/sigma].
    d1_base_sigma <- ((responses_item-mu)**2 / sigma**3 - 1/sigma)
    # Base Hessian for s0 parameter.
    d2_base_sigma0 <-
      eta_d_sigma0_base*(1 / sigma**2 - 3*(responses_item - mu)**2 / sigma**4) +
      (-1 / (2*sigma**2))*((responses_item - mu)**2 / sigma**3 - 1/sigma)
    # Base Hessian for s1 parameters.
    d2_base_sigma1 <- -2*eta_d_sigma1_base*(sigma**(-3)*(responses_item - mu)**2)

    # First and second derivative for c0.
    d1[1,1] <- sum(d1_base_mu, na.rm = TRUE) #d1
    d2[1,1] <- sum(d2_base_mu, na.rm = TRUE) #d2

    # First and second derivative for a0.
    d1[2,1] <- sum(eta_d_mu_a0*d1_base_mu, na.rm = TRUE) #d1
    d2[2,2] <- sum(eta_d_mu_a0**2*d2_base_mu, na.rm = TRUE) #d2

    # First and second derivative for s0.
    d1[(3+num_predictors*2),1] <- sum(eta_d_sigma0_base*d1_base_sigma, na.rm = TRUE) #d1
    d2[(3+num_predictors*2),(3+num_predictors*2)] <-
      sum(eta_d_sigma0_base*d2_base_sigma0, na.rm = TRUE) #d2

    # Cross derivative for c0 and a0.
    d2[2,1] <- sum(eta_d_mu_a0*d2_base_mu, na.rm = TRUE) #d2

    # Cross derivative for c0 and s0.
    d2[(3+num_predictors*2),1] <- sum(-eta_d_sigma0_base*d1_base_mu*eta_d_sigma1_base**(-1),
                                      na.rm = TRUE) #d2

    # Cross derivative for a0 and s0.
    d2[(3+num_predictors*2),2] <-
      sum(-eta_d_sigma0_base*eta_d_mu_a0*d1_base_mu*eta_d_sigma1_base**(-1),
          na.rm = TRUE) #d2

    # Cycle through predictors (outer cycle).
    for(cov in 1:num_predictors) {

      # First derivative for linear predictor w.r.t. covariate.
      cov_matrix <- pred_data[,cov]

      # First and second derivatives for c1.
      d1[2+cov,1] <-
        sum(cov_matrix*d1_base_mu,
            na.rm = TRUE) #d1
      d2[2+cov,2+cov] <-
        sum(cov_matrix**2*d2_base_mu,
            na.rm = TRUE) #d2

      # First and second derivatives for a1.
      d1[2+num_predictors+cov,1] <-
        sum(cov_matrix*eta_d_mu_a0*d1_base_mu,
            na.rm = TRUE) #d1
      d2[2+num_predictors+cov,2+num_predictors+cov] <-
        sum((cov_matrix*eta_d_mu_a0)**2*d2_base_mu,
            na.rm = TRUE) #d2

      # First and second derivatives for s1.
      d1[(3+num_predictors*2+cov),1] <-
        sum(cov_matrix*eta_d_sigma1_base*d1_base_sigma,
            na.rm = TRUE) #d1
      d2[(3+num_predictors*2+cov),(3+num_predictors*2+cov)] <-
        sum(cov_matrix**2/2*d2_base_sigma1,
            na.rm = TRUE) #d2

      # # Cross derivatives for c0 and c1.
      d2[2+cov,1] <- sum(cov_matrix*d2_base_mu, na.rm = TRUE) #d2

      # # Cross derivatives for c0 and a1, as well as a0 and c1.
      d2[2+num_predictors+cov,1] <- d2[2+cov,2] <-
        sum(cov_matrix*eta_d_mu_a0*d2_base_mu,
            na.rm = TRUE) #d2

      # # Cross derivatives for a0 and a1.
      d2[2+num_predictors+cov,2] <-
        sum(cov_matrix*eta_d_mu_a0**2*d2_base_mu,
            na.rm = TRUE) #d2

      # # Cross derivatives for c0 and s1
      d2[(3+num_predictors*2+cov),1] <- sum(-cov_matrix*d1_base_mu,
                                            na.rm = TRUE) #d2

      # # Cross derivatives for a0 and s1
      d2[(3+num_predictors*2+cov),2] <- sum(-cov_matrix*eta_d_mu_a0*d1_base_mu,
                                            na.rm = TRUE) #d2

      # # Cross derivatives for s0 and c1
      d2[(3+num_predictors*2),2+cov] <-
        sum(-cov_matrix*eta_d_sigma0_base*d1_base_mu*eta_d_sigma1_base**(-1),
            na.rm = TRUE) #d2

      # # Cross derivatives for s0 and a1
      d2[(3+num_predictors*2),2+num_predictors+cov] <-
        sum(-cov_matrix*eta_d_mu_a0*eta_d_sigma0_base*d1_base_mu*eta_d_sigma1_base**(-1),
            na.rm = TRUE) #d2

      # # Cross derivatives for s0 and s1
      d2[(3+num_predictors*2+cov),(3+num_predictors*2)] <-
        sum(cov_matrix*eta_d_sigma0_base*d2_base_sigma1*(1/eta_d_sigma1_base/2),
            na.rm = TRUE) #d2

      # # Cycle through predictors (inner cycle).
      for(cov2 in 1:num_predictors) {


        if(cov == cov2) {

          # Cross derivatives with same predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix**2*eta_d_mu_a0*d2_base_mu,
                na.rm = TRUE) #d2

          # Cross derivatives with same predictor for c1 and s1.
          d2[(3+num_predictors*2+cov),2+cov2] <- sum(-cov_matrix**2*d1_base_mu,
                                                     na.rm = TRUE) #d2

          # Cross derivatives with same predictor for a1 and s1.
          d2[(3+num_predictors*2+cov),2+num_predictors+cov2] <-
            sum(-cov_matrix**2*eta_d_mu_a0*d1_base_mu,
                na.rm = TRUE) #d2

        } else {

          # First derivatives for linear predictor w.r.t. second covariate.
          cov2_matrix <- pred_data[,cov2]

          # Cross derivatives with different predictor for c1 and a1.
          d2[2+num_predictors+cov,2+cov2] <-
            sum(cov_matrix*cov2_matrix*eta_d_mu_a0*d2_base_mu,
                na.rm = TRUE) #d2

          # Cross derivatives with different predictor for c1 and s1.
          d2[(3+num_predictors*2+cov),2+cov2] <-
            sum(-cov_matrix*cov2_matrix*d1_base_mu,
                na.rm = TRUE) #d2

          # Cross derivatives with different predictor for a1 and s1.
          d2[(3+num_predictors*2+cov),2+num_predictors+cov2] <-
            sum(-cov_matrix*cov2_matrix*eta_d_mu_a0*d1_base_mu,
                na.rm = TRUE) #d2

          if(cov2 > 1 && cov < cov2) {

            # Cross derivatives with different predictor for c1 and c1.
            d2[2+cov2,2+cov] <-
              sum(cov_matrix*cov2_matrix*d2_base_mu, #d2
                  na.rm = TRUE)

            # Cross derivatives with different predictor for a1 and a1.
            d2[2+num_predictors+cov2,2+num_predictors+cov] <-
              sum(cov_matrix*cov2_matrix*eta_d_mu_a0**2*d2_base_mu, #a1a1
                  na.rm = TRUE)

            # Cross derivatives with different predictor for s1 and s1.
            d2[(3+num_predictors*2+cov2),(3+num_predictors*2+cov)] <-
              sum(cov_matrix*cov2_matrix/2*d2_base_sigma1,
                  na.rm = TRUE) #d2
          }
        }

      }

    }


    dlist <- list(d1,d2)

  }
